import numpy as np
import torch
import networkx as nx
import dgl
from torch_scatter import scatter_add
from functools import reduce
from torch_geometric.data import Data
from tqdm import tqdm


class AdaptData(Data):
    """
    Data class adapted for NBFNet, inheriting from torch_geometric.data.Data.
    Converts DGL graphs to PyG format for GeneralizedRelationalConv.
    """

    def __init__(self, edge_index, edge_type, num_nodes, num_relations, **kwargs):
        super(AdaptData, self).__init__(**kwargs)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_nodes = num_nodes
        self.num_relations = num_relations

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.num_nodes
        elif key == 'edge_type':
            return self.num_relations
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        return 1 if key in ['edge_index', 'edge_type'] else 0


def ssp_multigraph_to_dgl_v1(graph):
    """
    Convert ssp multigraph (list of adjs) to DGL multigraph.
    Uses NetworkX as intermediate (for transductive mode, prevents OOM on large datasets).
    """
    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    for rel, adj in enumerate(graph):
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    g_dgl = dgl.from_networkx(g_nx, edge_attrs=['type'])
    g_dgl.ndata['idx'] = torch.LongTensor(np.arange(g_dgl.num_nodes()))
    return g_dgl


def ssp_multigraph_to_dgl(graph):
    """Accelerated version: bypass NetworkX, build DGL graph directly from tensors (for inductive mode)."""
    src_list, dst_list, rel_list = [], [], []
    num_nodes = graph[0].shape[0] if graph and len(graph) > 0 else 0

    for rel, adj in enumerate(graph):
        if adj.nnz == 0:
            continue
        coo = adj.tocoo()
        src_list.append(torch.tensor(coo.row, dtype=torch.long))
        dst_list.append(torch.tensor(coo.col, dtype=torch.long))
        rel_list.append(torch.full((coo.nnz,), rel, dtype=torch.long))

    if not src_list:
        g_dgl = dgl.graph(([], []), num_nodes=num_nodes)
        g_dgl.edata['type'] = torch.empty((0,), dtype=torch.long)
    else:
        g_dgl = dgl.graph((torch.cat(src_list), torch.cat(dst_list)), num_nodes=num_nodes)
        g_dgl.edata['type'] = torch.cat(rel_list)

    g_dgl.ndata['idx'] = torch.arange(g_dgl.num_nodes(), dtype=torch.long)
    return g_dgl


def move_to_device_dgl(molecular_graphs, device):
    """Move molecular graphs to device with self-loops added."""
    graphs = []
    for g in dict(sorted(molecular_graphs.items())).values():
        src, dst = g.edges()
        n = int(g.num_nodes())

        sl_idx = torch.arange(n, device=src.device)
        new_src = torch.cat([src, sl_idx])
        new_dst = torch.cat([dst, sl_idx])

        new_g = dgl.graph((new_src, new_dst), num_nodes=n)
        new_g.ndata['feat'] = g.ndata['feat']
        graphs.append(new_g)

    return dgl.batch(graphs).to(device=device)


def collate_dgl(samples):
    heads = torch.tensor([s[0][0] for s in samples], dtype=torch.long)
    rels = torch.tensor([s[0][1] for s in samples], dtype=torch.long)
    tails = torch.tensor([s[0][2] for s in samples], dtype=torch.long)
    h_neg_list = [s[0][3] for s in samples]
    t_neg_list = [s[0][4] for s in samples]
    K = len(h_neg_list[0])
    heads_neg = torch.tensor(h_neg_list, dtype=torch.long)
    tails_neg = torch.tensor(t_neg_list, dtype=torch.long)
    drug_pairs = torch.stack([heads, tails, rels], dim=1)
    frag_graphs = dgl.batch([s[1] for s in samples])
    r_labels = torch.tensor([s[3] for s in samples], dtype=torch.long)
    g_labels = torch.tensor([s[4] for s in samples], dtype=torch.float32)
    return (heads, tails), frag_graphs, drug_pairs, r_labels, g_labels, heads_neg, tails_neg


def move_batch_to_device_dgl(batch, device):
    (heads, tails), frag_graphs, drug_pairs, r_labels, g_labels, heads_neg, tails_neg = batch
    return (
        (heads.to(device), tails.to(device)),
        frag_graphs.to(device),
        drug_pairs.to(device),
        r_labels.to(device),
        g_labels.to(device),
        heads_neg.to(device),
        tails_neg.to(device)
    )


def build_weighted_degree_relation_graph(h, r, t, num_rels, min_weight=0.01, prune_fullconn=False, fullconn_thresh=5):
    """Build weighted degree-biased relation meta-graph with 8 edge type patterns."""
    max_node = int(torch.cat([h, t]).max().item()) + 1
    deg = torch.bincount(torch.cat([h, t]), minlength=max_node).numpy()

    h_np, r_np, t_np = h.numpy(), r.numpy(), t.numpy()
    order = np.argsort(h_np, kind='stable')
    h_sorted, r_sorted, t_sorted = h_np[order], r_np[order], t_np[order]

    unique_pivots, split_indices = np.unique(h_sorted, return_index=True)
    split_indices = np.append(split_indices, len(h_np))

    num_rels_sq = num_rels * num_rels
    counts_flat = np.zeros(8 * num_rels_sq, dtype=np.uint32)
    chunk_size = 4096

    with tqdm(total=len(unique_pivots), desc="Building relation meta-graph", unit="entity", ncols=80) as pbar:
        for idx in range(len(unique_pivots)):
            start, end = split_indices[idx], split_indices[idx + 1]
            count = end - start
            pbar.update(1)

            if count < 2:
                continue

            pivot = unique_pivots[idx]
            deg_p = deg[pivot]
            rels = r_sorted[start:end]
            d_neigh = deg[t_sorted[start:end]]

            f1 = (deg_p > d_neigh).astype(np.int8)

            for c in range(0, count, chunk_size):
                c_end = min(c + chunk_size, count)
                k = c_end - c
                f1_chunk = f1[c:c_end]
                d_chunk = d_neigh[c:c_end]

                f3 = (d_chunk[:, None] > d_neigh[None, :]).astype(np.int8)
                types = (f1_chunk[:, None] << 2) | (f1[None, :] << 1) | f3

                diag_start = max(0, c)
                diag_end = min(count, c + k)
                if diag_start < diag_end:
                    row_idx = np.arange(diag_end - diag_start)
                    col_idx = np.arange(diag_start, diag_end)
                    types[row_idx, col_idx] = -1

                types_flat = types.ravel()
                valid_mask = types_flat >= 0
                if not np.any(valid_mask):
                    continue

                valid_types = types_flat[valid_mask].astype(np.int32)
                flat_indices = np.nonzero(valid_mask)[0]
                r1_indices = flat_indices // count
                r2_indices = flat_indices % count

                r1_v = rels[c + r1_indices]
                r2_v = rels[r2_indices]

                flat_idx = valid_types.astype(np.int64) * num_rels_sq + r1_v.astype(np.int64) * num_rels + r2_v.astype(
                    np.int64)
                counts_flat += np.bincount(flat_idx, minlength=8 * num_rels_sq).astype(np.uint32)

    counts = counts_flat.reshape(8, num_rels, num_rels)

    # Remove self-loops
    counts[:, np.arange(num_rels), np.arange(num_rels)] = 0
    types_arr, r1_arr, r2_arr = np.where(counts > 0)

    if len(types_arr) == 0:
        return Data(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_type=torch.empty((0,), dtype=torch.long),
            edge_weight=torch.empty((0,), dtype=torch.float32),
            num_nodes=num_rels, num_rel_types=8
        )

    edge_index = torch.tensor([r1_arr, r2_arr], dtype=torch.long)
    edge_type = torch.tensor(types_arr, dtype=torch.long)

    src_nodes = r1_arr
    out_degree = np.bincount(src_nodes, minlength=num_rels)
    out_degree_safe = np.where(out_degree > 0, out_degree, 1)
    weight_vals = 1.0 / out_degree_safe[src_nodes].astype(np.float32)
    edge_weight = torch.tensor(weight_vals, dtype=torch.float32)

    return Data(
        edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight,
        num_nodes=num_rels, num_rel_types=8
    )


def edge_match(edge_index, query_index):
    """
        Match query edges against a large edge index.
        O((n + q)logn) time, O(n) memory.
    """
    if edge_index.shape[1] == 0:
        return (torch.empty(0, dtype=torch.long, device=edge_index.device),
                torch.zeros(query_index.shape[1], dtype=torch.long, device=edge_index.device))

    base = edge_index.max(dim=1)[0] + 1
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    num_match = end - start

    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match
