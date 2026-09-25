from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch
import logging
from utils.data_utils import process_files_ddi, build_entity_relation_adj, build_adj_from_components, load_and_map_triplets, generate_strict_negatives, build_entity_relation_adj_v1
from utils.graph_utils import ssp_multigraph_to_dgl, build_weighted_degree_relation_graph, ssp_multigraph_to_dgl_v1
from data_processor.mol_graph import pre_dgl_graphs, merge_graphs


class SubgraphDataset(Dataset):
    """
    Joint dataset adapted for ULTRA:
    - Removed SEAL/LMDB dependency, loads triplets directly
    - Molecular graph / fragment graph construction logic preserved
    - __getitem__ returns (h, t, r) for ULTRA full-graph gathering
    """

    def __init__(self, split='train', raw_data_paths=None, add_transpose_rels=True,
                 dataset='', ssp_graph=None, molecular_graphs=None, frag_graphs=None,
                 id2entity=None, id2relation=None, rel=None, global_graph=None,
                 BKG_file_name='', mode='transductive', data_root='data',
                 rel_graph_min_weight=0.01, train_relation_graph=None, inference_relation_graph=None,
                 num_negatives=2, num_drugs=None,
                 inductive_setting=None, bkg_full_file=None,
                 unseen_entities_file=None, unseen_relations_file=None,
                 support_file=None, entity_map=None, relation_map=None):

        self.mode = mode
        self.num_negatives = num_negatives

        # Unified path handling
        main_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        if mode == 'inductive':
            base_dataset = dataset.split('/')[0]
            data_dir = os.path.join(main_dir, '..', data_root, base_dataset)
        else:
            data_dir = os.path.join(main_dir, '..', data_root, dataset)

        BKG_file = os.path.join(data_dir, f"{BKG_file_name}.txt") if BKG_file_name else None
        self.entity_type = np.loadtxt(os.path.join(data_dir, 'entity.txt'))

        # ===== KG graph construction =====
        if not ssp_graph:
            if inductive_setting is not None:
                # Inductive training set
                self._build_inductive_kg(
                    inductive_setting, bkg_full_file,
                    unseen_entities_file, unseen_relations_file,
                    support_file, raw_data_paths, add_transpose_rels,
                    rel_graph_min_weight, data_dir
                )
                return
            else:
                # Transductive training set
                BKG_file = os.path.join(data_dir, f"{BKG_file_name}.txt")
                ssp_graph, triplets, entity2id, relation2id, id2entity, id2relation, rel, num_drugs = \
                    process_files_ddi(raw_data_paths, BKG_file)

                self.num_rels = rel
                self.num_drugs = num_drugs
                self.id2entity = id2entity
                self.id2relation = id2relation

                if add_transpose_rels:
                    num_ddi = self.num_rels
                    for i in range(num_ddi):
                        ssp_graph[i] = ssp_graph[i] + ssp_graph[i].T
                    bkg_part = ssp_graph[num_ddi:]
                    bkg_trans = [adj.T for adj in bkg_part]
                    ssp_graph = ssp_graph[:num_ddi] + bkg_part + bkg_trans

                self.ssp_graph = ssp_graph
                self.aug_num_rels = len(ssp_graph)
                self.num_base_relations = self.num_rels + (self.aug_num_rels - self.num_rels) // 2
                self.global_graph = self._to_dgl(ssp_graph)
                self.num_entity = ssp_graph[0].shape[0]

                h, t = self.global_graph.edges()
                r = self.global_graph.edata['type']
                self.train_relation_graph = build_weighted_degree_relation_graph(
                    h, r, t, self.aug_num_rels, min_weight=rel_graph_min_weight)
                self.inference_relation_graph = self.train_relation_graph

                self.entity_relation_adj = self._build_er_adj(
                    ssp_graph, self.num_entity, self.num_rels,
                    include_transpose=True, exclude_self_loop=True)
                self.num_meta_relations = 8
        else:
            # External ssp_graph provided
            self.ssp_graph = ssp_graph
            self.global_graph = global_graph
            self.num_entity = ssp_graph[0].shape[0]
            self.num_rels = rel
            self.aug_num_rels = len(ssp_graph)
            self.train_relation_graph = train_relation_graph
            self.inference_relation_graph = inference_relation_graph
            self.entity_relation_adj = self._build_er_adj(
                ssp_graph, self.num_entity, self.num_rels,
                include_transpose=True, exclude_self_loop=True)
            self.num_base_relations = self.num_rels + (self.aug_num_rels - self.num_rels) // 2
            self.num_meta_relations = 8

        # ===== Molecular graph construction =====
        if not molecular_graphs:
            smiles_path = os.path.join(data_dir, 'Drug_Information.txt')
            smiles_df = pd.read_csv(smiles_path, sep='\t', header=None, names=['entity', 'smiles'])
            smiles_df = smiles_df.dropna(subset=['entity', 'smiles'])

            smiles_df['entity'] = smiles_df['entity'].astype(str).str.replace(r'\.0+$', '', regex=True).str.strip()
            entity2smiles = dict(zip(smiles_df['entity'], smiles_df['smiles']))

            def normalize_entity_id(val):
                try:
                    return str(int(float(val)))
                except (ValueError, TypeError):
                    return str(val).strip().replace('.0', '')

            id2smiles = {}
            missing_count = 0
            for eid, entity_val in self.id2entity.items():
                clean_name = normalize_entity_id(entity_val)
                if clean_name in entity2smiles:
                    id2smiles[eid] = entity2smiles[clean_name]
                else:
                    missing_count += 1

            print(f"id2entity total: {len(self.id2entity)} | mapped: {len(id2smiles)} | missing: {missing_count}")
            if len(id2smiles) == 0:
                print("id2smiles is empty! Entity IDs may contain non-numeric prefixes.")
                raise RuntimeError("SMILES mapping failed. Please check entity ID format consistency.")

            molecular_graphs, frag_graphs, atom_feats_size, frag_feats_size = pre_dgl_graphs(id2smiles)
            self.molecular_graphs = molecular_graphs
            self.frag_graphs = frag_graphs
            self.atom_feats_size = atom_feats_size
            self.frag_feats_size = frag_feats_size
        else:
            self.molecular_graphs = molecular_graphs
            self.frag_graphs = frag_graphs

        # ===== Load triplets for current split =====
        if raw_data_paths and split in raw_data_paths:
            triplet_file = raw_data_paths[split]
        else:
            triplet_file = os.path.join(data_dir, f"{split}.txt")

        if entity_map is not None or relation_map is not None:
            self.triplets = load_and_map_triplets(triplet_file, entity_map, relation_map)
        else:
            self.triplets = np.loadtxt(triplet_file, dtype=np.int64)

        self.num_graphs = len(self.triplets)

        self.max_n_label = [0, 0]
        self.subgraph_feature_num = None
        self.num_node = self.global_graph.num_nodes()
        if not hasattr(self, 'true_pairs'):
            self.true_pairs = None

    def __getitem__(self, index):
        """
        Returns: (h, t, r), frag_comb_graph, drug_pair, r_label, g_label
        """
        h, t, r = self.triplets[index]
        r_label = r
        g_label = 1.0

        # Original ID unification: training compact ID -> original ID (inductive train only)
        if getattr(self, 'compact2orig', None) is not None:
            h = int(self.compact2orig[int(h)])
            t = int(self.compact2orig[int(t)])
        # Original ID unification: inference compact ID -> original ID (inductive eval only)
        elif getattr(self, 'inv_map_active', None) is not None:
            h = int(self.inv_map_active[int(h)])
            t = int(self.inv_map_active[int(t)])

        # Negative sampling
        if self.true_pairs is not None:
            if getattr(self, 'neg_pool', None) is not None:
                h_pool = self.orig2pool[int(h)]
                t_pool = self.orig2pool[int(t)]
                h_neg, t_neg = generate_strict_negatives(
                    h_pool, t_pool, r, self.num_negatives,
                    self.true_pairs, len(self.neg_pool))
                h_neg_list = self.neg_pool[h_neg].tolist()
                t_neg_list = self.neg_pool[t_neg].tolist()
            else:
                h_neg_list, t_neg_list = generate_strict_negatives(
                    h, t, r, self.num_negatives, self.true_pairs, self.num_entity)
                h_neg_list = h_neg_list.tolist()
                t_neg_list = t_neg_list.tolist()
        else:
            # Validation/test mode: simple sampling
            h_neg_list = []
            for _ in range(self.num_negatives):
                hn = h
                while hn == h:
                    hn = np.random.randint(0, self.num_entity)
                h_neg_list.append(hn)
            t_neg_list = []
            for _ in range(self.num_negatives):
                tn = t
                while tn == t:
                    tn = np.random.randint(0, self.num_entity)
                t_neg_list.append(tn)

        drug_pair = (int(h), int(r), int(t))

        frag_graph1 = self.frag_graphs[int(h)]
        frag_graph2 = self.frag_graphs[int(t)]
        frag_comb_graph = merge_graphs(frag_graph1, frag_graph2)

        r_label = torch.tensor(r_label, dtype=torch.long)

        return (h, r, t, h_neg_list, t_neg_list), frag_comb_graph, drug_pair, r_label, g_label

    def __len__(self):
        return self.num_graphs

    def _to_dgl(self, adj_list):
        """Select graph conversion function based on mode."""
        if self.mode == 'transductive':
            return ssp_multigraph_to_dgl_v1(adj_list)
        else:
            return ssp_multigraph_to_dgl(adj_list)

    def _build_er_adj(self, adj_list, num_entities, num_rels,
                      include_transpose=True, exclude_self_loop=True):
        """Select adjacency table construction function based on mode."""
        if self.mode == 'transductive':
            return build_entity_relation_adj_v1(adj_list, num_entities, num_rels,
                                                include_transpose, exclude_self_loop)
        else:
            return build_entity_relation_adj(adj_list, num_entities, num_rels,
                                             include_transpose, exclude_self_loop)

    def _build_inductive_kg(self, setting, bkg_full_file, unseen_entities_file,
                            unseen_relations_file, support_file, raw_data_paths,
                            add_transpose_rels, rel_graph_min_weight, data_dir):
        # 1. Read files
        train_ddi = np.loadtxt(raw_data_paths['train'], dtype=np.int64).reshape(-1, 3)
        support_ddi = np.loadtxt(support_file, dtype=np.int64).reshape(-1, 3) if support_file else np.zeros((0, 3), dtype=np.int64)
        bkg_all = np.loadtxt(bkg_full_file, dtype=np.int64).reshape(-1, 3)
        unseen_e = set(np.loadtxt(unseen_entities_file, dtype=np.int64)) if unseen_entities_file else set()
        unseen_r = set(np.loadtxt(unseen_relations_file, dtype=np.int64)) if unseen_relations_file else set()

        # 2. Collect drug entities, all relations, and non-drug entities from BKG
        all_ddi = [train_ddi,
                   np.loadtxt(raw_data_paths['valid'], dtype=np.int64).reshape(-1, 3),
                   np.loadtxt(raw_data_paths['test'], dtype=np.int64).reshape(-1, 3),
                   support_ddi]
        drug_entities = set()
        all_relations = set()
        for arr in all_ddi:
            if arr.size:
                drug_entities.update(arr[:, 0], arr[:, 1])
                all_relations.update(arr[:, 2])

        bkg_entities = set()
        if bkg_all.size:
            bkg_entities.update(bkg_all[:, 0], bkg_all[:, 1])
        non_drug_entities = bkg_entities - drug_entities

        E_old_drugs = drug_entities - unseen_e
        need_rel = setting in ('S3', 'S4', 'S5')
        R_old = all_relations - unseen_r if need_rel else all_relations

        # ================= Training graph A (offset scheme) =================
        visible_drugs = sorted(E_old_drugs)
        invisible_drugs = sorted(unseen_e)
        train_ent_list = visible_drugs + invisible_drugs + sorted(non_drug_entities)
        train_ent_map = {orig: i for i, orig in enumerate(train_ent_list)}
        allowed_drugs_A = E_old_drugs

        train_rel_map_full, train_adj, num_bkg_train = build_adj_from_components(
            train_ddi, None, bkg_all, train_ent_map, R_old,
            add_transpose_rels, allowed_drugs_A)

        train_g = self._to_dgl(train_adj)
        train_rg = build_weighted_degree_relation_graph(
            train_g.edges()[0], train_g.edata['type'], train_g.edges()[1],
            num_rels=len(train_adj), min_weight=rel_graph_min_weight)
        train_er_adj = self._build_er_adj(train_adj, len(train_ent_map), len(R_old),
                                          include_transpose=True, exclude_self_loop=True)
        train_num_base = len(R_old) + num_bkg_train

        train_ddi_rel_map = {orig: i for i, orig in enumerate(sorted(R_old))}
        train_triplet_ent_map = {orig: i for i, orig in enumerate(visible_drugs)}

        # ================= Inference graph B (compact mapping, no offset) =================
        if setting in ('S2', 'S5'):
            allowed_drugs_B = drug_entities
            inf_drugs = sorted(drug_entities)
        else:
            allowed_drugs_B = drug_entities
            inf_drugs = sorted(drug_entities)
        inf_ent_list = inf_drugs + sorted(non_drug_entities)
        inf_ent_map = {orig: i for i, orig in enumerate(inf_ent_list)}

        combined_ddi = np.vstack([train_ddi, support_ddi]) if train_ddi.size > 0 else support_ddi
        inf_rel_map, inf_adj, num_bkg_inf = build_adj_from_components(
            combined_ddi, None, bkg_all, inf_ent_map, all_relations,
            add_transpose_rels, allowed_drugs_B)

        inf_g = self._to_dgl(inf_adj)
        inf_rg = build_weighted_degree_relation_graph(
            inf_g.edges()[0], inf_g.edata['type'], inf_g.edges()[1],
            num_rels=len(inf_adj), min_weight=rel_graph_min_weight)
        inf_num_base = len(all_relations) + num_bkg_inf

        inf_ddi_rel_map = {orig: i for i, orig in enumerate(sorted(all_relations))}

        # ================= Molecular graph loading (with cache) =================
        orig_id2entity = {orig: orig for orig in drug_entities}
        cache_path = os.path.join(data_dir, f'inductive_{setting}_mol_cache.pt')

        if os.path.exists(cache_path):
            logging.info(f'Loading molecular graphs from cache: {cache_path}')
            cache_data = torch.load(cache_path)
            mol_orig, frag_orig = cache_data['mol'], cache_data['frag']
            self.atom_feats_size, self.frag_feats_size = cache_data['atom_dim'], cache_data['frag_dim']
        else:
            smiles_path = os.path.join(data_dir, 'Drug_Information.txt')
            smiles_df = pd.read_csv(smiles_path, sep='\t', header=None, names=['entity', 'smiles'])
            smiles_df = smiles_df.dropna(subset=['entity', 'smiles'])
            smiles_df['entity'] = smiles_df['entity'].astype(str).str.replace(r'\.0+$', '', regex=True).str.strip()
            entity2smiles = dict(zip(smiles_df['entity'], smiles_df['smiles']))

            def normalize_entity_id(val):
                try:
                    return str(int(float(val)))
                except:
                    return str(val).strip().replace('.0', '')

            id2smiles = {}
            for eid, orig_val in orig_id2entity.items():
                clean = normalize_entity_id(orig_val)
                if clean in entity2smiles:
                    id2smiles[eid] = entity2smiles[clean]
            mol_orig, frag_orig, self.atom_feats_size, self.frag_feats_size = pre_dgl_graphs(id2smiles)
            torch.save({'mol': mol_orig, 'frag': frag_orig,
                        'atom_dim': self.atom_feats_size, 'frag_dim': self.frag_feats_size}, cache_path)

        # Convert molecular graph keys
        self.molecular_graphs = {e: g for e, g in mol_orig.items() if e in visible_drugs}
        self.frag_graphs = {e: g for e, g in frag_orig.items() if e in visible_drugs}
        self.inference_molecular_graphs = {e: g for e, g in mol_orig.items() if e in inf_drugs}
        self.inference_frag_graphs = {e: g for e, g in frag_orig.items() if e in inf_drugs}

        # ================= Attribute assignment =================
        self.ssp_graph = train_adj
        self.global_graph = train_g
        self.train_relation_graph = train_rg
        self.inference_relation_graph = inf_rg
        self.num_rels = len(all_relations)
        self.aug_num_rels = len(train_adj)
        self.num_drugs = len(drug_entities)
        self.id2entity = {c: o for o, c in train_ent_map.items()}
        self.id2relation = {c: o for o, c in train_rel_map_full.items()}
        self.entity_relation_adj = train_er_adj
        self.num_meta_relations = 8
        self.num_base_relations = train_num_base
        self.num_node = train_g.num_nodes()

        self.inference_ssp_graph = inf_adj
        self.inference_global_graph = inf_g
        self.inference_num_entity = len(inf_drugs)
        self.inference_entity_map = inf_ent_map
        self.inference_relation_map = inf_ddi_rel_map
        self.inference_num_base_relations = inf_num_base

        self.triplets = load_and_map_triplets(raw_data_paths['train'], train_triplet_ent_map, train_ddi_rel_map)
        self.num_graphs = len(self.triplets)

        self.max_n_label = [0, 0]
        self.subgraph_feature_num = None

        # Strict negative sampling: training positive pairs (undirected)
        self.true_pairs = set()
        for hh, tt, rr in self.triplets:
            self.true_pairs.add((int(hh), int(tt)))
            self.true_pairs.add((int(tt), int(hh)))
        self.orig2pool = {int(o): i for i, o in enumerate(visible_drugs)}
        self.unseen_ids_inf = {inf_ent_map[e] for e in unseen_e if e in inf_ent_map}

        # Original ID unification + strict inductive constraint
        self.compact2orig = torch.tensor(visible_drugs, dtype=torch.long)
        self.inv_inference_entity_map = {v: k for k, v in inf_ent_map.items()}
        self.unseen_orig_ids = sorted(unseen_e)

        # Negative sampling space: visible drugs only (original IDs), no unseen entities during training
        self.neg_pool = torch.tensor(visible_drugs, dtype=torch.long)
        self.num_entity = len(visible_drugs)

        # inductive safety: original ID -> graph-space ID mapping (for propagation)
        max_orig = max(drug_entities) + 1
        self.orig2trainA = torch.full((max_orig,), -1, dtype=torch.long)
        self.orig2infB = torch.full((max_orig,), -1, dtype=torch.long)
        for orig in drug_entities:
            self.orig2trainA[orig] = train_ent_map[orig]
            self.orig2infB[orig] = inf_ent_map[orig]
