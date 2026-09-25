import numpy as np
from scipy.sparse import csc_matrix
from collections import defaultdict


def process_files_ddi(files, BKG_file, keeptrainone=False):
    entity2id = {}
    relation2id = {}

    triplets = {}
    kg_triple = []
    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = []
        file_data = np.loadtxt(file_path)
        for triplet in file_data:
            triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = triplet[0]
            if triplet[1] not in entity2id:
                entity2id[triplet[1]] = triplet[1]
            if triplet[2] not in relation2id:
                if keeptrainone:
                    triplet[2] = 0
                    relation2id[triplet[2]] = 0
                    rel = 1
                else:
                    relation2id[triplet[2]] = triplet[2]
                    rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[2] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
        triplets[file_type] = np.array(data)

    num_drugs = len(entity2id)

    triplet_kg = np.loadtxt(BKG_file)
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(t), int(r)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t
        if rel + r not in relation2id:
            relation2id[rel + r] = rel + r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    max_entity_id = max(np.max(kg_triple[:, 0]), np.max(kg_triple[:, 1])) + 1
    # Construct the list of adjacency matrix each corresponding to each relation.
    adj_list = []
    for i in range(rel):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                   shape=(max_entity_id, max_entity_id)))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i - rel)
        adj_list.append(csc_matrix(
            (np.ones(len(idx), dtype=np.uint8), (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))),
            shape=(max_entity_id, max_entity_id)))
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel, num_drugs


def build_entity_relation_adj_v1(adj_list, num_entities, num_rels, include_transpose=True, exclude_self_loop=True):
    """
    Build entity out/in-edge relation adjacency table (CSR format).
    - include_transpose=True: includes transpose relations (index num_rels ~ 2*num_rels-1)
    - exclude_self_loop=True: excludes trailing self-loop matrix
    """
    out_rels = [[] for _ in range(num_entities)]
    in_rels = [[] for _ in range(num_entities)]

    total_adjs = len(adj_list)
    limit = total_adjs
    if exclude_self_loop:
        limit -= 1

    for rel_id in range(limit):
        adj = adj_list[rel_id]
        if adj.nnz == 0:
            continue
        coo = adj.tocoo()
        for h, t in zip(coo.row, coo.col):
            out_rels[h].append(rel_id)
            in_rels[t].append(rel_id)

    def to_csr(rel_list):
        indices, indptr = [], [0]
        for rels in rel_list:
            indices.extend(sorted(rels))
            indptr.append(len(indices))
        return {'indices': indices, 'indptr': indptr}

    return {'out': to_csr(out_rels), 'in': to_csr(in_rels)}


def build_entity_relation_adj(adj_list, num_entities, num_rels,
                              include_transpose=True, exclude_self_loop=True):
    """Vectorized CSR adjacency table construction (lossless accelerated version)."""
    total_adjs = len(adj_list)
    limit = total_adjs
    if exclude_self_loop and total_adjs > 0:
        limit -= 1

    all_h, all_t, all_rel = [], [], []
    for rel_id in range(limit):
        adj = adj_list[rel_id]
        if adj.nnz == 0:
            continue
        coo = adj.tocoo()
        all_h.append(coo.row)
        all_t.append(coo.col)
        all_rel.append(np.full(coo.nnz, rel_id, dtype=np.int32))

    if not all_h:
        return {'out': {'indices': [], 'indptr': [0] * (num_entities + 1)},
                'in': {'indices': [], 'indptr': [0] * (num_entities + 1)}}

    all_h = np.concatenate(all_h)
    all_t = np.concatenate(all_t)
    all_rel = np.concatenate(all_rel)

    # Build out_rels (CSR format)
    order_out = np.argsort(all_h, kind='stable')
    sorted_h = all_h[order_out]
    sorted_rel_out = all_rel[order_out]
    out_indptr = np.zeros(num_entities + 1, dtype=np.int64)
    np.add.at(out_indptr, sorted_h + 1, 1)
    np.cumsum(out_indptr, out=out_indptr)

    # Build in_rels (CSR format)
    order_in = np.argsort(all_t, kind='stable')
    sorted_t = all_t[order_in]
    sorted_rel_in = all_rel[order_in]
    in_indptr = np.zeros(num_entities + 1, dtype=np.int64)
    np.add.at(in_indptr, sorted_t + 1, 1)
    np.cumsum(in_indptr, out=in_indptr)

    return {
        'out': {'indices': sorted_rel_out.tolist(), 'indptr': out_indptr.tolist()},
        'in': {'indices': sorted_rel_in.tolist(), 'indptr': in_indptr.tolist()}
    }


def compute_relation_stats(train_data, num_rels):
    """
    Compute per-relation inductive statistics. Returns [num_rels, stat_dim] numpy array.
    Features: global frequency quantile, avg head/tail entity degree,
    relation graph in/out degree distribution (6 valid patterns).
    """

    # 1. Training triplet frequency
    triplets = train_data.triplets
    count = np.bincount(triplets[:, 2], minlength=num_rels)
    sorted_counts = np.sort(count[count > 0])
    quantile = np.zeros(num_rels, dtype=np.float32)
    if len(sorted_counts) > 0:
        quantile[count > 0] = np.searchsorted(sorted_counts, count[count > 0]) / len(sorted_counts)
    else:
        quantile[:] = 0.0

    # 2. Average head/tail entity degree
    adj_list = train_data.ssp_graph
    entity_degree = defaultdict(int)
    for adj in adj_list:
        coo = adj.tocoo()
        for u, v in zip(coo.row, coo.col):
            entity_degree[u] += 1
            entity_degree[v] += 1
    max_entity = max(entity_degree.keys()) + 1
    degs = np.zeros(max_entity, dtype=np.float32)
    for e, d in entity_degree.items():
        degs[e] = d

    avg_head_deg = np.zeros(num_rels, dtype=np.float32)
    avg_tail_deg = np.zeros(num_rels, dtype=np.float32)
    for r, adj in enumerate(adj_list):
        if adj.nnz == 0:
            continue
        coo = adj.tocoo()
        heads = coo.row
        tails = coo.col
        if len(heads) > 0:
            avg_head_deg[r] = np.mean(degs[heads])
            avg_tail_deg[r] = np.mean(degs[tails])

    # 3. Degree-biased relation graph in/out degree distribution (6 valid patterns: 0,1,3,4,6,7)
    rel_graph = train_data.train_relation_graph
    edge_index = rel_graph.edge_index.numpy()
    edge_type = rel_graph.edge_type.numpy()
    valid_types = [0, 1, 3, 4, 6, 7]
    out_deg_mat = np.zeros((num_rels, 6), dtype=np.float32)
    in_deg_mat = np.zeros((num_rels, 6), dtype=np.float32)
    type_to_col = {t: i for i, t in enumerate(valid_types)}
    for i in range(len(edge_type)):
        t = edge_type[i]
        if t not in type_to_col:
            continue
        col = type_to_col[t]
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < num_rels:
            out_deg_mat[src, col] += 1
        if dst < num_rels:
            in_deg_mat[dst, col] += 1

    # 4. Concatenate features [num_rels, 1+1+1+6+6] = 15
    stats = np.stack([
        quantile,
        avg_head_deg,
        avg_tail_deg,
        out_deg_mat[:, 0], out_deg_mat[:, 1], out_deg_mat[:, 2],
        out_deg_mat[:, 3], out_deg_mat[:, 4], out_deg_mat[:, 5],
        in_deg_mat[:, 0], in_deg_mat[:, 1], in_deg_mat[:, 2],
        in_deg_mat[:, 3], in_deg_mat[:, 4], in_deg_mat[:, 5]
    ], axis=1)

    # Z-score normalization
    mean = stats.mean(axis=0, keepdims=True) + 1e-8
    std = stats.std(axis=0, keepdims=True) + 1e-8
    stats_norm = (stats - mean) / std
    return stats_norm


def build_adj_from_components(ddi_tri, extra_ddi, bkg_all, ent_map, rel_set,
                              add_transpose, allowed_drugs):
    n_ent = len(ent_map)
    sorted_rels = sorted(rel_set)
    rel_map = {orig: i for i, orig in enumerate(sorted_rels)}

    adj_rows = {i: [] for i in range(len(sorted_rels))}
    adj_cols = {i: [] for i in range(len(sorted_rels))}

    # 1. Process DDI edges
    for tri in [ddi_tri, extra_ddi]:
        if tri is None or tri.size == 0:
            continue
        for h, t, r in tri:
            if h not in allowed_drugs or t not in allowed_drugs:
                continue
            if r not in rel_set:
                continue
            hc, tc, rc = ent_map[h], ent_map[t], rel_map[r]
            adj_rows[rc].append(hc)
            adj_cols[rc].append(tc)

    # 2. Process BKG edges
    bkg_rel_ids = set()
    bkg_rel_sorted = []
    bkg_offset = len(sorted_rels)
    bkg_rel_map = {}
    total_rels = bkg_offset

    if bkg_all is not None and bkg_all.size > 0:
        ent_keys = set(ent_map.keys())

        valid_mask = np.array([(h in ent_keys and t in ent_keys)
                               for h, t in zip(bkg_all[:, 0], bkg_all[:, 1])])
        valid_bkg = bkg_all[valid_mask]

        if valid_bkg.size > 0:
            hc_arr = np.array([ent_map[h] for h in valid_bkg[:, 0]])
            tc_arr = np.array([ent_map[t] for t in valid_bkg[:, 1]])
            r_arr = valid_bkg[:, 2]

            bkg_rel_ids.update(r_arr)
            bkg_rel_sorted = sorted(bkg_rel_ids)
            bkg_offset = len(sorted_rels)
            bkg_rel_map = {orig: i + bkg_offset for i, orig in enumerate(bkg_rel_sorted)}
            total_rels = bkg_offset + len(bkg_rel_sorted)

            for i in range(len(sorted_rels), total_rels):
                adj_rows[i] = []
                adj_cols[i] = []

            for hc, tc, r_orig in zip(hc_arr, tc_arr, r_arr):
                rc = bkg_rel_map[r_orig]
                adj_rows[rc].append(hc)
                adj_cols[rc].append(tc)

    # 3. Build CSC matrices
    adj = []
    for i in range(total_rels):
        if len(adj_rows[i]) > 0:
            rows = np.array(adj_rows[i])
            cols = np.array(adj_cols[i])

            unique_edges = np.unique(np.stack([rows, cols]), axis=1)
            rows = unique_edges[0]
            cols = unique_edges[1]

            data = np.ones(len(rows), dtype=np.uint8)
            adj.append(csc_matrix((data, (rows, cols)), shape=(n_ent, n_ent)))
        else:
            adj.append(csc_matrix((n_ent, n_ent), dtype=np.uint8))

    # 4. Handle transpose
    if add_transpose:
        num_ddi = len(sorted_rels)
        for i in range(num_ddi):
            adj[i] = adj[i] + adj[i].T
        bkg_part = adj[num_ddi:]
        bkg_trans = [a.T for a in bkg_part]
        adj = adj[:num_ddi] + bkg_part + bkg_trans

    full_rel_map = {}
    full_rel_map.update(rel_map)
    for orig, cont in bkg_rel_map.items():
        full_rel_map[orig] = cont

    return full_rel_map, adj, len(bkg_rel_sorted)


def load_and_map_triplets(filepath, entity_map=None, relation_map=None):
    """Load triplet file and apply entity/relation ID mapping."""
    tri = np.loadtxt(filepath, dtype=np.int64)
    if tri.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if entity_map is not None:
        tri[:, 0] = np.vectorize(entity_map.get)(tri[:, 0])
        tri[:, 1] = np.vectorize(entity_map.get)(tri[:, 1])
    if relation_map is not None:
        tri[:, 2] = np.vectorize(relation_map.get)(tri[:, 2])
    return tri


def generate_strict_negatives(head, tail, rel, num_negs, true_pairs_set, num_entities, max_retries=10):
    """
    Generate strictly filtered negative samples (exclude self-loops + exclude false negatives).
    """
    neg_heads = []
    neg_tails = []

    for _ in range(num_negs):
        for _ in range(max_retries):
            corr_h = np.random.randint(0, num_entities)
            if corr_h != tail and (corr_h, tail) not in true_pairs_set:
                break
        neg_heads.append(corr_h)

    for _ in range(num_negs):
        for _ in range(max_retries):
            corr_t = np.random.randint(0, num_entities)
            if head != corr_t and (head, corr_t) not in true_pairs_set:
                break
        neg_tails.append(corr_t)

    return np.array(neg_heads), np.array(neg_tails)
