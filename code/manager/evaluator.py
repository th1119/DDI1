import os
import numpy as np
import torch
import random
import torch.nn.functional as F
import torch.distributed as dist
import signal
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score, accuracy_score
from tqdm import tqdm
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl, move_to_device_dgl

GLOBAL_SEED = 42
GLOBAL_WORKER_ID = None


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
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def _timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out")


class Evaluator_multiclass():
    """
    Multi-class evaluator (unconditioned propagation mode).
    """

    def __init__(self, params, classifier, data, is_test=False):
        self.params = params
        self.graph_classifier = classifier
        self.data = data

        self.move_batch_to_device = move_batch_to_device_dgl
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers
        self.is_test = is_test
        self.eval_times = 0
        self.current_epoch = 0
        self.rank = params.rank
        self.world_size = params.world_size

    def eval(self):
        world_size = self.world_size
        rank = self.rank

        # Timeout setting
        if self.params.dataset == 'DrugBank':
            timeout_seconds = 14400
        else:
            timeout_seconds = 7200
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            # Distributed DataLoader
            if world_size > 1:
                sampler = DistributedSampler(self.data, num_replicas=world_size, rank=rank, shuffle=False)
                dataloader = DataLoader(
                    self.data, batch_size=self.params.batch_size,
                    sampler=sampler, num_workers=self.num_workers,
                    collate_fn=self.collate_fn, worker_init_fn=init_fn)
            else:
                dataloader = DataLoader(
                    self.data, batch_size=self.params.batch_size,
                    shuffle=False, num_workers=self.num_workers,
                    collate_fn=self.collate_fn, worker_init_fn=init_fn)

            self.graph_classifier.eval()
            device = self.params.device

            # Get model object (DDP compatible)
            model = self.graph_classifier.module if hasattr(self.graph_classifier, 'module') else self.graph_classifier

            # Inductive compatibility: rebuild entity2mol_idx
            _saved_entity2mol_idx = model.entity2mol_idx
            if hasattr(self.data, 'num_entity'):
                mol_keys = sorted(self.data.molecular_graphs.keys())
                entity2mol = torch.full((self.data.num_entity,), -1, dtype=torch.long, device=device)
                for local_idx, eid in enumerate(mol_keys):
                    entity2mol[int(eid)] = local_idx
                model.entity2mol_idx = entity2mol

            # Load molecular graphs to GPU
            mol_graphs = move_to_device_dgl(self.data.molecular_graphs, device)

            local_labels, local_preds, local_probas = [], [], []
            local_loss_sum, local_batches = 0.0, 0

            with torch.no_grad():
                for batch in tqdm(dataloader, disable=(rank != 0)):
                    (heads, tails), frag_graphs, drug_pairs, r_labels, g_labels, heads_neg, tails_neg = \
                        self.move_batch_to_device(batch, device)
                    B, num_rels = heads.size(0), self.params.num_rels

                    # Determine which KG full graph to use
                    if hasattr(self.params,
                               'inference_global_graph') and self.params.inference_global_graph is not None:
                        kg_full_graph = self.params.inference_global_graph.to(device)
                    else:
                        kg_full_graph = self.params.global_graph.to(device)

                    # Unconditioned propagation + classification
                    logits_joint, _, _ = model(
                        kg_full_graph, mol_graphs, frag_graphs, drug_pairs,
                        labels=None, mode='test')
                    preds = torch.argmax(logits_joint, dim=1).cpu().numpy()
                    probas = F.softmax(logits_joint, dim=1).cpu().numpy()

                    loss_val = F.cross_entropy(logits_joint, r_labels).item()
                    local_loss_sum += loss_val * B
                    local_batches += B
                    local_labels.extend(r_labels.cpu().numpy().tolist())
                    local_preds.extend(preds.tolist())
                    local_probas.extend(probas)

            # Release molecular graphs
            del mol_graphs
            torch.cuda.empty_cache()

            # Multi-GPU aggregation
            if world_size > 1:
                gathered_labels = [None] * world_size
                gathered_preds = [None] * world_size
                gathered_probas = [None] * world_size
                dist.all_gather_object(gathered_labels, local_labels)
                dist.all_gather_object(gathered_preds, local_preds)
                dist.all_gather_object(gathered_probas, local_probas)
                loss_tensor = torch.tensor([local_loss_sum, local_batches], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = loss_tensor[0].item() / max(int(loss_tensor[1].item()), 1)
            else:
                gathered_labels = [local_labels]
                gathered_preds = [local_preds]
                gathered_probas = [local_probas]
                avg_loss = local_loss_sum / max(local_batches, 1)

            if world_size > 1:
                if rank != 0:
                    model.entity2mol_idx = _saved_entity2mol_idx
                    dist.barrier()
                    return {'loss': 0.0, 'acc': 0.0, 'f1_score': 0.0, 'pr_auc': 0.0, 'k': 0.0}, {
                        'f1': np.zeros(self.params.num_rels)}

            all_labels = np.concatenate([np.array(lst) for lst in gathered_labels])
            all_preds = np.concatenate([np.array(lst) for lst in gathered_preds])
            all_probas = np.concatenate([np.array(lst) for lst in gathered_probas])

            acc = accuracy_score(all_labels, all_preds)
            f1_macro = f1_score(all_labels, all_preds, average='macro')
            f1_per_class = f1_score(all_labels, all_preds, average=None)
            labels_onehot = np.eye(self.params.num_rels)[all_labels]
            pr_auc = average_precision_score(labels_onehot, all_probas, average="macro")
            kappa = cohen_kappa_score(all_labels, all_preds)

            if world_size > 1:
                dist.barrier()
            model.entity2mol_idx = _saved_entity2mol_idx
            return (
                {'loss': avg_loss, 'acc': acc, 'f1_score': f1_macro, 'pr_auc': pr_auc, 'k': kappa},
                {'f1': f1_per_class}
            )
        finally:
            signal.alarm(0)
