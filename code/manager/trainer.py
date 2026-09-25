import torch
import os
import numpy as np
import time
import logging
import random
import torch.distributed as dist
import signal
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sklearn import metrics
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl, move_to_device_dgl
import csv
import math

GLOBAL_SEED = 42
GLOBAL_WORKER_ID = None


def _timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out")


class Trainer(object):
    def __init__(self, params, model, train_data, valid_evaluator, test_evaluator, train_sampler=None):
        self.params = params
        self.train_data = train_data
        self.graph_classifier = model
        self.train_sampler = train_sampler
        self.world_size = params.world_size
        self.rank = params.rank

        # Preload molecular graphs to device
        self.molecular_graphs = move_to_device_dgl(self.train_data.molecular_graphs, self.params.device)

        self.valid_evaluator = valid_evaluator
        self.test_evaluator = test_evaluator

        self.batch_size = params.batch_size
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers
        self.updates_counter = 0
        self.early_stop = 0

        model_params = list(self.graph_classifier.parameters())
        if self.rank == 0:
            logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        # Build global entity ID to molecular graph batch index mapping
        mol_keys = sorted(self.train_data.molecular_graphs.keys())
        n_rows = max(int(self.train_data.num_entity), (int(mol_keys[-1]) + 1) if mol_keys else 0)
        self.entity2mol_idx = torch.full((n_rows,), -1, dtype=torch.long, device=params.device)
        for local_idx, global_eid in enumerate(mol_keys):
            eid_int = int(float(global_eid))
            if eid_int < n_rows:
                self.entity2mol_idx[eid_int] = local_idx
        params.entity2mol_idx = self.entity2mol_idx

        model_ref = self.graph_classifier.module if hasattr(self.graph_classifier, 'module') else self.graph_classifier

        # Inductive safety: mark unseen entities
        if getattr(params, 'inductive', False):
            unseen_ids = getattr(train_data, 'unseen_orig_ids', None)
            if unseen_ids is not None:
                model_ref.mark_unseen_entities(unseen_ids)
                if self.rank == 0:
                    logging.info(f'[InductiveSafe] marked {len(unseen_ids)} unseen (orig IDs)')

        # Optimizer and scheduler
        self.optimizer = Adam(self.graph_classifier.parameters(), lr=params.lr, weight_decay=params.weight_decay_rate)
        # Scheduler: Warmup + Cosine annealing (no restart)
        steps_per_epoch = len(train_data) // (params.batch_size * max(params.world_size, 1)) // max(params.accum_steps, 1)
        total_steps = steps_per_epoch * params.num_epochs
        warmup_steps = steps_per_epoch * 2

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.02, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if self.rank == 0:
            logging.info(f'Scheduler: Warmup({warmup_steps}) + Cosine({total_steps - warmup_steps}) | '
                         f'steps/epoch={steps_per_epoch}')

        self.move_batch_to_device = move_batch_to_device_dgl
        self.reset_training_state()
        self.test_result = {}
        self.val_result = {}

        # Gradient accumulation config
        self.accum_steps = getattr(params, 'accum_steps', 4)
        # Pre-move full graph to GPU
        self.kg_full_graph = self.params.global_graph.to(params.device)

        # Load physicochemical descriptors (187-dim)
        if getattr(model_ref, 'use_desc', False):
            _code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            desc_path = os.path.abspath(os.path.join(
                _code_dir, '..', 'data', self.params.dataset.split('/')[0], 'text_embeddings.pt'))
            if os.path.exists(desc_path):
                desc_dict = torch.load(desc_path, map_location='cpu')
                n_rows = max(int(k) for k in desc_dict.keys()) + 1
                desc_tensor = torch.zeros(n_rows, 187)
                missing = 0
                for i in range(n_rows):
                    if str(i) in desc_dict:
                        desc_tensor[i] = desc_dict[str(i)]
                    else:
                        missing += 1
                desc_tensor = desc_tensor.to(self.params.device)
                model_ref.desc_cache = desc_tensor
                if self.rank == 0:
                    logging.info(f'[Desc] Loaded: shape={desc_tensor.shape}, missing={missing}')
            else:
                if self.rank == 0:
                    logging.warning(f'[Desc] Not found: {desc_path}. Using zeros.')
                model_ref.desc_cache = None

        # Precompute neighbor indices (transductive/inductive branching)
        model_ref = self.graph_classifier.module if hasattr(self.graph_classifier, 'module') else self.graph_classifier
        if getattr(model_ref, 'use_pair_attention', False):
            if getattr(params, 'inductive', False):
                # Inductive
                import numpy as _np
                train_txt = self.params.file_paths['train']
                tri = _np.loadtxt(train_txt, dtype=_np.int64).reshape(-1, 3)

                n_drugs_orig = int(model_ref.num_drugs)

                src = _np.concatenate([tri[:, 0], tri[:, 1]])
                dst = _np.concatenate([tri[:, 1], tri[:, 0]])

                pairs = _np.unique(_np.stack([src, dst], axis=1), axis=0)

                pairs = pairs[(pairs[:, 0] < n_drugs_orig) &
                              (pairs[:, 1] < n_drugs_orig) &
                              (pairs[:, 0] != pairs[:, 1])]

                order = _np.argsort(pairs[:, 0], kind='stable')
                pairs = pairs[order]

                K = model_ref.max_neighbors
                neighbor_indices = torch.full((n_drugs_orig, K), -1, dtype=torch.long)
                neighbor_counts = torch.zeros(n_drugs_orig, dtype=torch.long)

                node_ids = _np.arange(n_drugs_orig)
                starts = _np.searchsorted(pairs[:, 0], node_ids)
                ends = _np.searchsorted(pairs[:, 0], node_ids, side='right')
                for node in range(n_drugs_orig):
                    s, e = int(starts[node]), int(ends[node])
                    if s == e:
                        continue
                    n = min(e - s, K)
                    neighbor_indices[node, :n] = torch.from_numpy(pairs[s:s + n, 1].copy())
                    neighbor_counts[node] = n

                neighbor_indices = neighbor_indices.to(params.device)
                neighbor_counts = neighbor_counts.to(params.device)
                model_ref.register_buffer('neighbor_indices', neighbor_indices)
                model_ref.register_buffer('neighbor_counts', neighbor_counts)
                if self.rank == 0:
                    logging.info(f'[PairAttention] Neighbors(orig-ID, inductive): '
                                 f'avg={neighbor_counts.float().mean():.1f}, '
                                 f'max={neighbor_counts.max().item()}')
            else:
                # Transductive
                neighbor_indices, neighbor_counts = model_ref._precompute_neighbor_indices(
                    self.params.global_graph, self.params.num_drugs, model_ref.max_neighbors)
                model_ref.register_buffer('neighbor_indices', neighbor_indices)
                model_ref.register_buffer('neighbor_counts', neighbor_counts)
                if self.rank == 0:
                    avg_n = neighbor_counts.float().mean().item()
                    max_n = neighbor_counts.max().item()
                    logging.info(f'[PairAttention] Neighbors: avg={avg_n:.1f}, max={max_n}')

    def train_batch(self):
        def init_fn(worker_id):
            global GLOBAL_WORKER_ID
            GLOBAL_WORKER_ID = worker_id
            seed = GLOBAL_SEED + worker_id
            np.random.seed(seed)
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False

        train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=self.train_sampler,
            worker_init_fn=init_fn
        )

        self.graph_classifier.train()
        model_ref = self.graph_classifier.module if hasattr(self.graph_classifier, 'module') else self.graph_classifier
        model_ref._current_epoch = self.epoch

        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=(self.rank != 0))

        epoch_loss = 0.0
        num_batches = 0
        all_labels = []
        all_scores = []

        self.optimizer.zero_grad()

        for b_idx, batch in bar:
            (heads,
             tails), frag_graphs, drug_pairs, relation_labels, _, heads_neg, tails_neg = self.move_batch_to_device(
                batch, self.params.device)

            # Guard against entity2mol_idx out-of-bounds
            max_valid_id = self.entity2mol_idx.size(0) - 1
            all_entities = torch.cat([heads, tails])
            if heads_neg is not None:
                all_entities = torch.cat([all_entities, heads_neg.flatten()])
            if tails_neg is not None:
                all_entities = torch.cat([all_entities, tails_neg.flatten()])

            invalid_mask = (all_entities < 0) | (all_entities > max_valid_id)
            if invalid_mask.any():
                invalid_ids = all_entities[invalid_mask].unique().cpu().tolist()
                raise RuntimeError(
                    f"[FATAL] Entity ID out of bounds for entity2mol_idx (size={self.entity2mol_idx.size(0)})! "
                    f"Max valid ID: {max_valid_id}. Invalid IDs found: {invalid_ids[:20]}..."
                )

            if torch.isnan(
                    relation_labels).any() or relation_labels.min() < 0 or relation_labels.max() >= self.params.num_rels:
                raise ValueError(
                    f"Batch {b_idx} contains invalid labels! min={relation_labels.min()}, max={relation_labels.max()}, n_rel={self.params.num_rels}"
                )

            scores, total_loss, loss_dict = self.graph_classifier(
                self.kg_full_graph, self.molecular_graphs, frag_graphs, drug_pairs,
                labels=relation_labels, mode='train',
                heads_neg=heads_neg, tails_neg=tails_neg
            )

            loss_val = total_loss.item()
            main_val = loss_dict.get("main", 0.0)
            kl_val = loss_dict.get("kl", 0.0)

            if torch.isfinite(total_loss):
                (total_loss / self.accum_steps).backward()
            else:
                raise RuntimeError(f"Batch {b_idx} loss is NaN/Inf, training aborted.")

            self.updates_counter += 1

            # Periodic optimizer update
            if (b_idx + 1) % self.accum_steps == 0 or (b_idx + 1) == len(train_dataloader):
                clip_grad_norm_(self.graph_classifier.parameters(), max_norm=2.0, norm_type=2)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            num_batches += 1

            bar.set_description(f'b:{b_idx + 1} | total:{loss_val:.4f} | main:{main_val:.3f} | kl:{kl_val:.3f}')

            with torch.no_grad():
                epoch_loss += loss_val
                all_labels.extend(relation_labels.cpu().numpy().flatten().tolist())
                all_scores.extend(torch.argmax(scores, dim=1).cpu().flatten().tolist())

            del scores, total_loss, loss_dict, heads, tails, frag_graphs, drug_pairs, relation_labels

            # Periodic evaluation
            if self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                result, save_dev_data = self.valid_evaluator.eval()
                logging.info(f'Eval Performance: {result} in {time.time() - tic:.2f}s')

                tic = time.time()
                test_result, save_test_data = self.test_evaluator.eval()
                logging.info(f'Test Performance: {test_result} in {time.time() - tic:.2f}s')

                self.graph_classifier.train()

                # Early stopping decision
                if self.world_size > 1:
                    stop_tensor = torch.tensor([0], device=self.params.device)
                    save_flag = torch.tensor([0], device=self.params.device)
                    if self.rank == 0:
                        setting = getattr(self.params, 'inductive_setting', None)
                        if setting == 'S5':
                            cur_score = (result['f1_score'] + result['acc'] + result['pr_auc'] + result['k']) / 4.0
                        else:
                            cur_score = result['f1_score']

                        if cur_score >= self.best_metric:
                            save_flag[0] = 1
                            self.best_metric = cur_score
                            self.test_best_metric = test_result['f1_score']
                            self.not_improved_count = 0
                            self.val_result = result
                            self.test_result = test_result
                            logging.info(f'Test Performance Per Class: {save_test_data}')
                            early_stop_flag = 0
                        else:
                            self.not_improved_count += 1
                            if self.not_improved_count >= self.params.early_stop_epoch:
                                early_stop_flag = 1
                            else:
                                early_stop_flag = 0
                        stop_tensor[0] = early_stop_flag
                    dist.broadcast(stop_tensor, src=0)
                    dist.broadcast(save_flag, src=0)
                    if save_flag.item() == 1:
                        if self.rank != 0:
                            setting = getattr(self.params, 'inductive_setting', None)
                            if setting == 'S5':
                                self.best_metric = 0.6 * result['f1_score'] + 0.4 * result['acc']
                            else:
                                self.best_metric = result['f1_score']
                        self.save_classifier()
                    if stop_tensor.item() == 1:
                        self.early_stop = 1
                        break
                else:
                    setting = getattr(self.params, 'inductive_setting', None)
                    if setting == 'S5':
                        cur_score = (result['f1_score'] + result['acc'] + result['pr_auc'] + result['k']) / 4.0
                    else:
                        cur_score = result['f1_score']

                    if cur_score >= self.best_metric:
                        self.save_classifier()
                        self.best_metric = cur_score
                        self.test_best_metric = test_result['f1_score']
                        self.not_improved_count = 0
                        self.val_result = result
                        self.test_result = test_result
                        logging.info(f'Test Performance Per Class: {save_test_data}')
                    else:
                        self.not_improved_count += 1
                        if self.not_improved_count >= self.params.early_stop_epoch:
                            self.early_stop = 1
                            break

                self.last_metric = result['f1_score']

        avg_loss = epoch_loss / max(num_batches, 1)
        acc = metrics.accuracy_score(all_labels, all_scores)
        f1_macro = metrics.f1_score(all_labels, all_scores, average='macro')
        f1_micro = metrics.f1_score(all_labels, all_scores, average='micro')
        f1_per_class = metrics.f1_score(all_labels, all_scores, average=None)

        return avg_loss, acc, f1_macro, f1_micro, f1_per_class

    def save_classifier(self):
        if self.world_size > 1:
            torch.cuda.synchronize()
            dist.barrier()
        if self.rank == 0:
            save_path = os.path.join(self.params.exp_dir, f'best_graph_classifier_{self.params.iFold}.pth')
            model_state = self.graph_classifier.module.state_dict() if self.world_size > 1 else self.graph_classifier.state_dict()
            torch.save(model_state, save_path)
            logging.info(f'Better model saved w.r.t F1. Path: {save_path}')
        if getattr(self.params, 'world_size', 1) > 1:
            dist.barrier()

    def reset_training_state(self):
        self.best_metric = 0
        self.test_best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train(self):
        self.reset_training_state()
        for epoch in range(1, self.params.num_epochs + 1):
            if self.params.dataset == 'DrugBank':
                epoch_timeout = 252000
            else:
                epoch_timeout = 72000

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(epoch_timeout)

            try:
                if self.early_stop:
                    logging.info("Early stopping triggered. Exiting training.")
                    break

                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)

                self.epoch = epoch
                time_start = time.time()
                loss, acc, f1_macro, f1_micro, f1_per_class = self.train_batch()
                time_elapsed = time.time() - time_start

                if self.world_size > 1:
                    torch.cuda.synchronize()
                    dist.barrier()
            finally:
                signal.alarm(0)

            if self.rank == 0:
                logging.info(
                    f'Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f} | '
                    f'F1_Macro: {f1_macro:.4f} | F1_Micro: {f1_micro:.4f} | Time: {time_elapsed:.2f}'
                )

        # Save results to CSV
        if self.rank == 0:
            csv_path = os.path.join(self.params.exp_dir, 'result.csv')
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.params.result_fieldnames)
                writer.writerow({
                    'Fold': self.params.iFold,
                    'Eval Accuracy': self.val_result.get('acc', 0),
                    'Eval F1 Score': self.val_result.get('f1_score', 0),
                    'Eval PR AUC': self.val_result.get('pr_auc', 0),
                    'Eval Kappa': self.val_result.get('k', 0),
                    'Test Accuracy': self.test_result.get('acc', 0),
                    'Test F1 Score': self.test_result.get('f1_score', 0),
                    'Test PR AUC': self.test_result.get('pr_auc', 0),
                    'Test Kappa': self.test_result.get('k', 0)
                })
