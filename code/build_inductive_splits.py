#!/usr/bin/env python3
"""
build_inductive_splits.py
Generate inductive settings S1~S5 with 5-fold splits from transductive data.
Uses rejection sampling to ensure training set covers all relations with no data leakage.
"""
import os
import argparse
import random
import numpy as np
import networkx as nx
from collections import defaultdict


os.environ['PYTHONHASHSEED'] = '42'


def load_ddi_all(data_dir, dataset):
    triplets = []
    for fold in range(1, 6):
        fold_dir = os.path.join(data_dir, dataset, f"iFold_{fold}")
        for split in ['train', 'valid', 'test']:
            fpath = os.path.join(fold_dir, f"{split}.txt")
            if not os.path.exists(fpath): continue
            data = np.loadtxt(fpath, dtype=np.int64)
            if data.size == 0: continue
            if data.ndim == 1: data = data.reshape(-1, 3)
            triplets.extend(data.tolist())
    unique = list(set((int(h), int(t), int(r)) for h, t, r in triplets))
    print(f"[Info] Total DDI triplets (deduplicated): {len(unique)}")
    return unique


def load_bkg_all(data_dir, dataset):
    bkg_path = os.path.join(data_dir, dataset, "BKG_file.txt")
    if not os.path.exists(bkg_path): return []
    data = np.loadtxt(bkg_path, dtype=np.int64)
    if data.size == 0: return []
    if data.ndim == 1: data = data.reshape(-1, 3)
    return data.tolist()


def save_triplets(triplets, filepath):
    with open(filepath, 'w') as f:
        for h, t, r in triplets:
            f.write(f"{h} {t} {r}\n")


def split_ddi_by_setting(all_ddi, setting, unseen_entities, unseen_relations):
    T_train, T_inf_cand, T_sup_only = [], [], []
    for h, t, r in all_ddi:
        in_new_e_h = (h in unseen_entities)
        in_new_e_t = (t in unseen_entities)
        in_new_r = (r in unseen_relations) if unseen_relations is not None else False

        if setting == 'S1':
            if in_new_e_h or in_new_e_t:
                T_inf_cand.append((h, t, r))
            else:
                T_train.append((h, t, r))
        elif setting == 'S2':
            if in_new_e_h and in_new_e_t:
                T_inf_cand.append((h, t, r))
            elif (not in_new_e_h) and (not in_new_e_t):
                T_train.append((h, t, r))
            else:
                T_sup_only.append((h, t, r))
        elif setting == 'S3':
            if in_new_r:
                T_inf_cand.append((h, t, r))
            else:
                T_train.append((h, t, r))
        elif setting == 'S4':
            if in_new_e_t and in_new_r:
                T_inf_cand.append((h, t, r))
            elif (not in_new_e_t) and (not in_new_r):
                T_train.append((h, t, r))
            else:
                T_sup_only.append((h, t, r))
        elif setting == 'S5':
            if in_new_e_h and in_new_e_t and in_new_r:
                T_inf_cand.append((h, t, r))
            elif (not in_new_e_h) and (not in_new_e_t) and (not in_new_r):
                T_train.append((h, t, r))
            else:
                T_sup_only.append((h, t, r))
    return T_train, T_inf_cand, T_sup_only


def sample_unseen_entities(entities, ratio, seed):
    random.seed(seed)
    ents = sorted(list(entities))
    random.shuffle(ents)
    return set(ents[:int(len(ents) * ratio)])


def sample_unseen_entities_with_retry(all_ddi, entities, initial_ratio, seed, max_retries=50, min_edges_per_rel=1):
    """Rejection sampling for unseen entities (ensures training set covers all relations)."""
    rel_to_edges = defaultdict(list)
    for h, t, r in all_ddi:
        rel_to_edges[r].append((h, t))

    ents = sorted(list(entities))
    current_ratio = initial_ratio

    random.seed(seed)

    while current_ratio > 0.01:
        for attempt in range(max_retries):
            random.shuffle(ents)
            n_unseen = int(len(ents) * current_ratio)
            unseen_e = set(ents[:n_unseen])
            seen_e = set(ents) - unseen_e

            is_valid = True
            for r, edges in rel_to_edges.items():
                seen_edges_count = sum(1 for h, t in edges if h in seen_e and t in seen_e)
                if seen_edges_count < min_edges_per_rel:
                    is_valid = False
                    break

            if is_valid:
                if attempt > 0 or current_ratio < initial_ratio:
                    print(
                        f"[Info] Succeeded after {attempt + 1} attempts (ratio {current_ratio:.2f}), "
                        f"covering all {len(rel_to_edges)} relations.")
                return unseen_e

        print(f"[Warning] Ratio {current_ratio:.2f} failed after {max_retries} retries, reducing by 0.05...")
        current_ratio -= 0.05

    print(f"[Warning] Cannot guarantee full relation coverage at reasonable ratio, "
          f"returning last result (ratio {current_ratio:.2f}).")
    return unseen_e


def sample_feasible_unseen_relations(all_ddi, setting, unseen_entities, ratio, seed, min_edges=1):
    random.seed(seed)
    rel_avail = defaultdict(int)
    for h, t, r in all_ddi:
        in_new_e_h, in_new_e_t = (h in unseen_entities), (t in unseen_entities)
        if setting == 'S3':
            rel_avail[r] += 1
        elif setting == 'S4' and in_new_e_t:
            rel_avail[r] += 1
        elif setting == 'S5' and in_new_e_h and in_new_e_t:
            rel_avail[r] += 1

    eligible = [r for r, cnt in rel_avail.items() if cnt >= min_edges]
    if not eligible: raise RuntimeError("No relation satisfies the constraint. Please reduce the ratio.")
    random.shuffle(eligible)
    return set(eligible[:int(len(eligible) * ratio)])


def ensure_undirected_connectivity(all_ddi, all_bkg, unseen_e, unseen_r, setting):
    if setting not in ['S2', 'S5']: return unseen_e, unseen_r
    G = nx.Graph()
    for h, t, r in all_bkg: G.add_edge(h, t)
    for h, t, r in all_ddi:
        in_h, in_t = (h in unseen_e), (t in unseen_e)
        if setting == 'S2' and in_h and in_t:
            G.add_edge(h, t)
        elif setting == 'S5' and in_h and in_t and (r in unseen_r):
            G.add_edge(h, t)

    components = list(nx.connected_components(G))
    if not components: return unseen_e, unseen_r
    best_comp = max(components, key=lambda c: len(c.intersection(unseen_e)))
    isolated = unseen_e - best_comp
    if isolated:
        print(f"[Warning] Removing {len(isolated)} isolated unseen entities to ensure connectivity.")
        unseen_e -= isolated
    return unseen_e, unseen_r


def build_support_and_eval(T_inf_cand, setting, unseen_entities, unseen_relations):
    by_rel = defaultdict(list)
    for h, t, r in T_inf_cand: by_rel[r].append((h, t))
    for r in by_rel:
        freq = defaultdict(int)
        for h, t in by_rel[r]: freq[(h, t)] += 1
        by_rel[r] = sorted(by_rel[r], key=lambda x: freq[x])

    T_sup, L_dict = [], defaultdict(list)
    covered_e, covered_r = set(), set()
    need_e = setting in ['S1', 'S2', 'S4', 'S5']
    need_r = setting in ['S3', 'S4', 'S5']

    for r, edges in by_rel.items():
        for h, t in edges:
            if need_e:
                if (h in unseen_entities and h not in covered_e) or (t in unseen_entities and t not in covered_e):
                    T_sup.append((h, t, r))
                    covered_e.update([h, t])
                    if need_r and r in unseen_relations: covered_r.add(r)
                else:
                    L_dict[r].append((h, t, r))
            else:
                if need_r and r in unseen_relations and r not in covered_r:
                    T_sup.append((h, t, r))
                    covered_r.add(r)
                else:
                    L_dict[r].append((h, t, r))
        if need_r and r in unseen_relations and r not in covered_r and L_dict[r]:
            T_sup.append(L_dict[r].pop(0))
            covered_r.add(r)

    L = [e for sub in L_dict.values() for e in sub]
    return T_sup, L


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--setting', type=str, required=True, choices=['S1', 'S2', 'S3', 'S4', 'S5'])
    parser.add_argument('--unseen_entity_ratio', type=float, default=0.2)
    parser.add_argument('--unseen_rel_ratio', type=float, default=0.2)
    parser.add_argument('--min_support_edges', type=int, default=1)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
    args = parser.parse_args()

    all_ddi = load_ddi_all(args.data_root, args.dataset)
    all_bkg = load_bkg_all(args.data_root, args.dataset)

    entities, relations = set(), set()
    for h, t, r in all_ddi: entities.update([h, t]); relations.add(r)

    out_root = os.path.join(args.data_root, args.dataset, f"inductive_{args.setting}")
    os.makedirs(out_root, exist_ok=True)
    need_rel = args.setting in ['S3', 'S4', 'S5']

    for fold_idx in range(1, args.folds + 1):
        seed = args.seed + fold_idx
        print(f"\n===== Fold {fold_idx} (seed {seed}) =====")

        # 1. Sample unseen entities
        if args.setting in ('S1', 'S2'):
            unseen_e = sample_unseen_entities_with_retry(all_ddi, entities, args.unseen_entity_ratio, seed)
        else:
            unseen_e = sample_unseen_entities(entities, args.unseen_entity_ratio, seed)

        # 2. Sample unseen relations
        unseen_r = sample_feasible_unseen_relations(all_ddi, args.setting, unseen_e, args.unseen_rel_ratio, seed,
                                                    args.min_support_edges) if need_rel else set()

        # 3. Connectivity check
        unseen_e, unseen_r = ensure_undirected_connectivity(all_ddi, all_bkg, unseen_e, unseen_r, args.setting)

        # 4. DDI assignment
        T_train, T_inf_cand, T_sup_only = split_ddi_by_setting(all_ddi, args.setting, unseen_e, unseen_r)
        if not T_inf_cand and not T_sup_only: continue

        # 5. Support and evaluation edge extraction
        T_sup, L = build_support_and_eval(T_inf_cand, args.setting, unseen_e, unseen_r)
        T_sup.extend(T_sup_only)
        if not L: continue

        random.shuffle(L)
        mid = len(L) // 2

        # 6. Save files
        fold_dir = os.path.join(out_root, f"iFold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        save_triplets(T_train, os.path.join(fold_dir, "train.txt"))
        save_triplets(L[:mid], os.path.join(fold_dir, "valid.txt"))
        save_triplets(L[mid:], os.path.join(fold_dir, "test.txt"))
        save_triplets(T_sup, os.path.join(fold_dir, "graph_B_support.txt"))
        save_triplets(all_bkg, os.path.join(fold_dir, "BKG_full.txt"))

        with open(os.path.join(fold_dir, "unseen_entities.txt"), 'w') as f:
            for e in sorted(list(unseen_e)): f.write(f"{e}\n")
        if need_rel:
            with open(os.path.join(fold_dir, "unseen_relations.txt"), 'w') as f:
                for r in sorted(list(unseen_r)): f.write(f"{r}\n")

        print(
            f"Fold {fold_idx} done: Train({len(T_train)}), Sup({len(T_sup)}), Val({len(L[:mid])}), Test({len(L[mid:])})")


if __name__ == "__main__":
    main()
