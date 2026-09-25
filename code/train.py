import os
import argparse
import random
import csv
import logging
import numpy as np
import torch
import torch.distributed as dist
import datetime
from warnings import simplefilter
from scipy.sparse import SparseEfficiencyWarning
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch_geometric.data import Data

from manager.trainer import Trainer
from manager.evaluator import Evaluator_multiclass
from model.Classifier_model import Classifier_model
from utils.initialization_utils import initialize_experiment, initialize_model
from data_processor.datasets import SubgraphDataset

if os.environ.get('DEBUG_CUDA', '0') == '1':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['OMP_NUM_THREADS'] = '4'
simplefilter('ignore', category=UserWarning)
simplefilter('ignore', category=SparseEfficiencyWarning)


def setup_distributed():
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['NCCL_BLOCKING_WAIT'] = '1'
        os.environ['NCCL_TIMEOUT'] = '14400'
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=14400)
        )
        return int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return 0, 0, 1


def set_seed(seed, rank=0):
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    os.environ['PYTHONHASHSEED'] = str(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False


def process_dataset(params):
    _code_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(_code_dir, '..', 'data'))

    if params.inductive:
        # ================= Inductive branch =================
        fold_dir = os.path.join(data_dir, params.dataset, params.iFold)
        params.file_paths = {
            'train': os.path.join(fold_dir, 'train.txt'),
            'valid': os.path.join(fold_dir, 'valid.txt'),
            'test':  os.path.join(fold_dir, 'test.txt')
        }

        train_data = SubgraphDataset(
            split='train',
            raw_data_paths=params.file_paths,
            dataset=params.dataset,
            data_root='data',
            mode='inductive',
            inductive_setting=params.inductive_setting,
            bkg_full_file=os.path.join(fold_dir, 'BKG_full.txt'),
            unseen_entities_file=os.path.join(fold_dir, 'unseen_entities.txt'),
            unseen_relations_file=os.path.join(fold_dir, 'unseen_relations.txt')
                if params.inductive_setting in ('S3', 'S4', 'S5') else None,
            support_file=os.path.join(fold_dir, 'graph_B_support.txt'),
            rel_graph_min_weight=params.rel_graph_min_weight,
            num_negatives=params.num_negatives,
            add_transpose_rels=params.add_transpose_rels
        )

        # Inject inference compact-to-original ID mapping for evaluation
        inf_kwargs = dict(
            ssp_graph=train_data.inference_ssp_graph,
            global_graph=train_data.inference_global_graph,
            train_relation_graph=train_data.inference_relation_graph,
            inference_relation_graph=train_data.inference_relation_graph,
            molecular_graphs=train_data.inference_molecular_graphs,
            frag_graphs=train_data.inference_frag_graphs,
            id2entity={c: o for o, c in train_data.inference_entity_map.items()},
            id2relation={c: o for o, c in train_data.inference_relation_map.items()},
            rel=train_data.num_rels,
            num_drugs=train_data.num_drugs,
            dataset=params.dataset,
            data_root='data',
            mode='inductive',
            num_negatives=params.num_negatives,
            add_transpose_rels=False,
            entity_map=train_data.inference_entity_map,
            relation_map=train_data.inference_relation_map
        )
        valid_data = SubgraphDataset(split='valid', raw_data_paths=params.file_paths, **inf_kwargs)
        test_data = SubgraphDataset(split='test',  raw_data_paths=params.file_paths, **inf_kwargs)

        inv_map_dict = train_data.inv_inference_entity_map
        n_inf = max(inv_map_dict.keys()) + 1
        inv_map = torch.full((n_inf,), -1, dtype=torch.long)
        for inf_id, orig_id in inv_map_dict.items():
            inv_map[inf_id] = orig_id
        valid_data.inv_map_active = inv_map
        test_data.inv_map_active = inv_map

        params.train_relation_graph = train_data.train_relation_graph
        params.inference_relation_graph = train_data.inference_relation_graph

        # Training global graph (Graph A)
        g_dgl = train_data.global_graph
        src, dst = g_dgl.edges()
        et = g_dgl.edata['type'].long()
        total_nodes_A = train_data.num_node
        total_nodes_B = train_data.inference_global_graph.num_nodes()

        params.global_graph = Data(
            edge_index=torch.stack([src, dst], dim=0).cpu(), edge_type=et.cpu(),
            fwd_mask=(et < train_data.num_base_relations).cpu(),
            bwd_mask=(et >= train_data.num_base_relations).cpu(),
            num_nodes=total_nodes_A,
            num_base_relations=train_data.num_base_relations
        )
        # Inference global graph (Graph B)
        inf_g = train_data.inference_global_graph
        inf_src, inf_dst = inf_g.edges()
        inf_et = inf_g.edata['type'].long()
        params.inference_global_graph = Data(
            edge_index=torch.stack([inf_src, inf_dst], dim=0).cpu(), edge_type=inf_et.cpu(),
            fwd_mask=(inf_et < train_data.inference_num_base_relations).cpu(),
            bwd_mask=(inf_et >= train_data.inference_num_base_relations).cpu(),
            num_nodes=total_nodes_B,
            num_base_relations=train_data.inference_num_base_relations
        )

        params.num_rels = train_data.num_rels
        params.aug_num_rels = train_data.aug_num_rels
        params.num_base_relations = train_data.num_base_relations
        params.kg_relation_dim = train_data.aug_num_rels
        params.subgraph_feature_num = None
        params.atom_feats_size = train_data.atom_feats_size
        params.frag_feats_size = train_data.frag_feats_size
        params.num_drugs = train_data.num_drugs
        params.entity_relation_adj = train_data.entity_relation_adj
        params.num_meta_relations = train_data.num_meta_relations

        params.num_entities = total_nodes_A
        params.num_nodes = total_nodes_A

        # Class weights
        counts = np.bincount(train_data.triplets[:, 2], minlength=params.num_rels).astype(np.float64)
        counts = np.maximum(counts, 1.0)

        weights = 1.0 / np.sqrt(counts)
        weights = weights / weights.sum() * params.num_rels
        weights = np.clip(weights, None, 5.0)

        params.class_weights = torch.tensor(weights, dtype=torch.float32).to(params.device)
        params.relation_counts = counts

        # Relation statistics (based on training graph)
        stats_path = os.path.join(data_dir, params.dataset, params.iFold, 'relation_stats.npy')
        if os.path.exists(stats_path) and not getattr(params, 'force_recompute_stats', False):
            relation_stats = np.load(stats_path)
        else:
            from utils.data_utils import compute_relation_stats
            relation_stats = compute_relation_stats(train_data, params.aug_num_rels)
            np.save(stats_path, relation_stats)
        params.relation_stats = relation_stats

        if getattr(params, 'rank', 0) == 0:
            logging.info(f"[INFO] num_entities: {params.num_entities} | num_base_relations: {params.num_base_relations}")
            logging.info(f"Device: {params.device} | # Nodes: {params.num_nodes} | # Relations: {params.num_rels}")
        return train_data, valid_data, test_data

    # ================= Transductive branch =================
    train_data = SubgraphDataset(split='train', raw_data_paths=params.file_paths,
                                 add_transpose_rels=params.add_transpose_rels, dataset=params.dataset,
                                 BKG_file_name=params.BKG_file_name, data_root='data',
                                 rel_graph_min_weight=params.rel_graph_min_weight,
                                 num_negatives=params.num_negatives)

    shared_kwargs = dict(ssp_graph=train_data.ssp_graph, molecular_graphs=train_data.molecular_graphs,
                         frag_graphs=train_data.frag_graphs, id2entity=train_data.id2entity,
                         id2relation=train_data.id2relation, rel=train_data.num_rels,
                         global_graph=train_data.global_graph, BKG_file_name=params.BKG_file_name,
                         dataset=params.dataset, data_root='data', add_transpose_rels=params.add_transpose_rels,
                         raw_data_paths=params.file_paths,
                         rel_graph_min_weight=params.rel_graph_min_weight,
                         train_relation_graph=train_data.train_relation_graph,
                         inference_relation_graph=train_data.inference_relation_graph,
                         num_negatives=params.num_negatives)
    valid_data = SubgraphDataset(split='valid', **shared_kwargs)
    test_data = SubgraphDataset(split='test', **shared_kwargs)

    g_dgl = train_data.global_graph
    src, dst = g_dgl.edges()
    edge_type = g_dgl.edata['type'].long()

    params.num_base_relations = train_data.num_rels + (train_data.aug_num_rels - train_data.num_rels) // 2
    params.aug_num_rels = train_data.aug_num_rels
    params.num_entities = max(train_data.num_entity, int(train_data.triplets.max()) + 1)
    params.kg_relation_dim = params.aug_num_rels

    params.global_graph = Data(edge_index=torch.stack([src, dst], dim=0).cpu(), edge_type=edge_type.cpu(),
                               fwd_mask=(edge_type < params.num_base_relations).cpu(),
                               bwd_mask=(edge_type >= params.num_base_relations).cpu(), num_nodes=params.num_entities,
                               num_base_relations=params.num_base_relations)

    params.num_rels = train_data.num_rels
    params.num_nodes = train_data.num_node
    params.subgraph_feature_num = None
    params.atom_feats_size = train_data.atom_feats_size
    params.frag_feats_size = train_data.frag_feats_size
    params.num_drugs = train_data.num_drugs
    params.train_relation_graph = train_data.train_relation_graph
    params.inference_relation_graph = getattr(train_data, 'inference_relation_graph', train_data.train_relation_graph)
    params.entity_relation_adj = train_data.entity_relation_adj
    params.num_meta_relations = train_data.num_meta_relations

    # Class weights
    counts = np.bincount(train_data.triplets[:, 2], minlength=params.num_rels).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * params.num_rels
    weights = np.clip(weights, None, 5.0)

    params.class_weights = torch.tensor(weights, dtype=torch.float32).to(params.device)
    params.relation_counts = counts

    if getattr(params, 'rank', 0) == 0:
        sorted_idx = np.argsort(weights)[::-1]
        logging.info(f'[Class Weights] min={weights.min():.4f}, max={weights.max():.4f}, '
                     f'ratio={weights.max() / weights.min():.1f}:1')
        logging.info(f'[Class Weights] count<10: {(counts < 10).sum()}, '
                     f'count<50: {(counts < 50).sum()}, count<100: {(counts < 100).sum()}')

    # Relation statistics
    stats_path = os.path.join(data_dir, params.dataset, params.iFold, 'relation_stats.npy')
    if os.path.exists(stats_path) and not getattr(params, 'force_recompute_stats', False):
        relation_stats = np.load(stats_path)
    else:
        from utils.data_utils import compute_relation_stats
        relation_stats = compute_relation_stats(train_data, params.aug_num_rels)
        np.save(stats_path, relation_stats)
    params.relation_stats = relation_stats

    if getattr(params, 'rank', 0) == 0:
        logging.info(f"[INFO] num_entities: {params.num_entities} | num_base_relations: {params.num_base_relations}")
        logging.info(f"Device: {params.device} | # Nodes: {params.num_nodes} | # Relations: {params.num_rels}")
    return train_data, valid_data, test_data


def main(params):
    rank, local_rank, world_size = setup_distributed()
    params.rank, params.local_rank, params.world_size = rank, local_rank, world_size
    params.device = torch.device(
        f'cuda:{local_rank}' if not params.disable_cuda and torch.cuda.is_available() else 'cpu')

    _code_dir = os.path.dirname(os.path.abspath(__file__))
    params.main_dir = os.path.abspath(os.path.join(_code_dir, '..'))
    params.exp_dir = os.path.abspath(os.path.join(_code_dir, '..', 'experiments', params.experiment_name))

    if params.inductive and params.inductive_setting:
        params.dataset = f'{params.dataset}/inductive_{params.inductive_setting}'

    if rank == 0:
        initialize_experiment(params, __file__)
        params.result_fieldnames = ['Fold', 'Eval Accuracy', 'Eval F1 Score', 'Eval PR AUC', 'Eval Kappa',
                                    'Test Accuracy', 'Test F1 Score', 'Test PR AUC', 'Test Kappa']
        os.makedirs(params.exp_dir, exist_ok=True)
        with open(os.path.join(params.exp_dir, 'result.csv'), mode='w', newline='') as f:
            csv.DictWriter(f, fieldnames=params.result_fieldnames).writeheader()
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
    if world_size > 1:
        dist.barrier()

    params.kg_num_layers = getattr(params, 'kg_num_layers', params.ultra_num_layers)
    data_dir = os.path.abspath(os.path.join(_code_dir, '..', 'data'))

    if params.fold_index is not None:
        fold_list = [params.fold_index - 1]
        params.Folds = 1
    else:
        fold_list = range(params.Folds)

    try:
        for iFold in fold_list:
            set_seed(params.seed, rank=rank)
            params.iFold = f'iFold_{iFold + 1}'
            if rank == 0: logging.info(f"{'=' * 40} iFold: {params.iFold} {'=' * 40}")

            params.file_paths = {
                'train': os.path.join(data_dir, params.dataset, params.iFold, f"{params.train_file}.txt"),
                'valid': os.path.join(data_dir, params.dataset, params.iFold, f"{params.valid_file}.txt"),
                'test': os.path.join(data_dir, params.dataset, params.iFold, f"{params.test_file}.txt")
            }

            train_data, valid_data, test_data = process_dataset(params)
            train_sampler = DistributedSampler(train_data, shuffle=True, drop_last=True) if world_size > 1 else None

            classifier = initialize_model(params, Classifier_model).to(params.device)
            if world_size > 1:
                classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
                classifier = DDP(classifier, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

            valid_evaluator = Evaluator_multiclass(params, classifier, valid_data)
            test_evaluator = Evaluator_multiclass(params, classifier, test_data, is_test=True)

            if rank == 0:
                logging.info(
                    f"Model Architecture Initialized. Total Params: {sum(p.numel() for p in classifier.parameters()):,}")
                logging.info('Start training...')

            trainer = Trainer(params, classifier, train_data, valid_evaluator, test_evaluator, train_sampler)

            # Sync entity2mol_idx from Trainer to model
            model_ref = classifier.module if hasattr(classifier, 'module') else classifier
            model_ref.entity2mol_idx = trainer.entity2mol_idx

            # Inductive safety: inject graph-space mapping
            if getattr(params, 'inductive', False):
                _mA = getattr(train_data, 'orig2trainA', None)
                _mB = getattr(train_data, 'orig2infB', None)
                if _mA is not None and _mB is not None:
                    model_ref.orig2trainA = _mA.to(params.device)
                    model_ref.orig2infB = _mB.to(params.device)
                    if rank == 0:
                        logging.info(f'[InductiveSafe] graph-space maps injected: '
                                     f'orig2trainA={tuple(_mA.shape)}, orig2infB={tuple(_mB.shape)}')

            trainer.train()
            if rank == 0: logging.info('Training finished.')

            if world_size > 1:
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.empty_cache()

    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="UnPairRelNet")

    # ================= Basic settings =================
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument("--experiment_name", "-e", type=str, default="default_v3_1")
    parser.add_argument('--Folds', type=int, default=1)
    parser.add_argument('--dataset', "-d", type=str, default='Ryu')
    parser.add_argument("--train_file", "-tf", type=str, default="train")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid")
    parser.add_argument("--test_file", "-ttf", type=str, default="test")
    parser.add_argument('--BKG_file_name', type=str, default='BKG_file')
    parser.add_argument('--add_transpose_rels', '-tr', action='store_true')
    parser.add_argument("--eval_every_iter", type=int, default=2000)
    parser.add_argument("--save_every_epoch", type=int, default=10)
    parser.add_argument("--early_stop_epoch", type=int, default=6)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--lr_decay_rate", type=float, default=0.93)
    parser.add_argument("--weight_decay_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", "-ne", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=14)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--emb_dim", "-dim", type=int, default=256)
    parser.add_argument("--gcn_dropout", type=float, default=0.2)  # [DEPRECATED] Unused, kept for compatibility
    parser.add_argument('--ultra_num_layers', type=int, default=2)
    parser.add_argument('--kg_emb_dim', type=int, default=64)
    parser.add_argument('--alpha_aux', type=float, default=0.08)
    parser.add_argument('--alpha_kl', type=float, default=0.0)
    parser.add_argument('--gamma_focal', type=float, default=2.5)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--rel_graph_min_weight', type=float, default=0.005,
                        help='Relation graph edge weight threshold (0-1), smaller = denser')
    parser.add_argument('--beta', type=float, default=0.999,
                        help='[DEPRECATED] Unused, kept for compatibility')
    parser.add_argument('--avg_opposite_rels', action='store_true',
                        help='Average forward/inverse DDI relation encodings')
    parser.add_argument('--inductive', action='store_true', help='Use inductive setting')
    parser.add_argument('--num_negatives', type=int, default=1, help='Number of negative relations per positive')
    parser.add_argument('--fold_index', type=int, default=None,
                        help='Single fold index (1-based), e.g. 1 = iFold_1')
    parser.add_argument('--inductive_setting', type=str, default=None,
                        choices=['S1', 'S2'],
                        help='Inductive setting type (requires --inductive)')

    # ===== Message passing =====
    parser.add_argument('--use_pair_attention', action='store_true',
                        help='Enable pair attention (local neighbor aggregation with cross info)')
    parser.add_argument('--max_neighbors', type=int, default=64,
                        help='Max neighbors per node in pair attention')
    parser.add_argument('--use_pair_context', action='store_true',
                        help='Enable drug-pair context injection into relation features')

    # ===== Physicochemical descriptors =====
    parser.add_argument('--use_desc', action='store_true',
                        help='Enable descriptor modality (187-dim = 20 physicochem + 167 MACCS)')
    parser.add_argument('--desc_dim', type=int, default=187,
                        help='Descriptor dimension')

    # ===== Inductive training =====
    parser.add_argument('--prop_drop_prob', type=float, default=0.0)
    parser.add_argument('--ddi_drop_prob', type=float, default=0.0,
                        help='Drop DDI edges with this probability during inductive training')

    params = parser.parse_args()
    params.rank = getattr(params, 'rank', 0)
    params.local_rank = getattr(params, 'local_rank', 0)
    params.world_size = getattr(params, 'world_size', 1)
    main(params)
