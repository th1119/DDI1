import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.graph_utils import AdaptData
from .encoder import UltraKGEncoder_Bi, MG_encoder
from .mol_model import MolecularUpdateModule


class ProjectionHead(nn.Module):
    """Shallow projection head: LayerNorm + ReLU + Dropout."""

    def __init__(self, in_dim, out_dim, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            curr_dim = out_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class Classifier_model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device = params.device
        self.emb_dim = params.emb_dim
        self.num_drugs = params.num_drugs
        self.n_rel = params.num_rels

        self.atom_dim = params.atom_feats_size
        self.frag_dim = params.frag_feats_size
        self.kg_emb_dim = getattr(params, 'kg_emb_dim', max(self.emb_dim // 2, 64))

        self.num_base_rels = getattr(params, 'num_base_relations', params.num_rels)

        # 1. KG dimension alignment projection
        self.kg_proj = nn.Linear(self.kg_emb_dim * 4, self.emb_dim)
        nn.init.kaiming_uniform_(self.kg_proj.weight)

        # 2. Encoder branches
        self.KG_encoder = UltraKGEncoder_Bi(
            kg_emb_dim=self.kg_emb_dim,
            num_layers=getattr(params, 'kg_num_layers', 2),
            num_relations=params.aug_num_rels,
            num_drugs=self.num_drugs,
            device=self.device,
            train_relation_graph=params.train_relation_graph,
            inference_relation_graph=params.inference_relation_graph,
            entity_relation_adj=params.entity_relation_adj,
            emb_dim=self.emb_dim,
            num_entities=getattr(params, 'num_entities', params.num_nodes),
            num_base_relations=self.num_base_rels,
            relation_stats=getattr(params, 'relation_stats', None),
            avg_opposite_rels=getattr(params, 'avg_opposite_rels', False),
            num_ddi_rels=self.n_rel
        )
        self.MolCov = MolecularUpdateModule(self.atom_dim, 512, self.emb_dim)
        self.MG_encoder = MG_encoder(self.frag_dim, 128, 128, self.emb_dim, num_heads=5)

        # 3. Modality interaction projection
        fusion_in_dim = self.emb_dim * 2
        self.W_d = ProjectionHead(fusion_in_dim, self.emb_dim, num_layers=2, dropout=0.1)

        # 4. Auxiliary classification heads
        self.cls_kg = nn.Linear(self.emb_dim, self.n_rel)
        self.cls_mol = nn.Linear(self.emb_dim, self.n_rel)

        for m in [self.kg_proj, self.cls_kg, self.cls_mol]:
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        self._inductive = getattr(params, 'inductive', False)
        self.entity2mol_idx = getattr(params, 'entity2mol_idx', None)

        # Transductive unconditioned propagation
        self.uncond_proj = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU()
        )
        self.pair_proj = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU()
        )
        self.rel_classifier = nn.Linear(self.emb_dim, self.n_rel)

        # Cache
        self._rel_feat_cache = None
        self._cache_epoch = -1
        self._current_epoch = 0

        for m in [self.uncond_proj, self.pair_proj]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(self.rel_classifier.weight)
        nn.init.zeros_(self.rel_classifier.bias)

        # Entity embedding + bilinear interaction
        self.drug_embeddings = nn.Embedding(self.num_drugs, 64)
        nn.init.xavier_uniform_(self.drug_embeddings.weight)
        self.emb_dropout = nn.Dropout(0.15)

        self.bilinear = nn.Bilinear(64, 64, self.emb_dim)

        # Transductive fusion layer (fixed 576-dim: mol 256 + uncond 256 + emb 64)
        self.W_d_trans = nn.Sequential(
            nn.Linear(self.emb_dim * 2 + 64, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU()
        )

        # Joint classifier
        if getattr(params, 'use_desc', False):
            joint_feat_dim = self.emb_dim * 5 + 64 * 2  # 1408
        else:
            joint_feat_dim = self.emb_dim * 5  # 1280
        self.joint_classifier = ProjectionHead(joint_feat_dim, self.n_rel, num_layers=2, dropout=0.2)

        # Class weights (for auxiliary relation prediction loss)
        if hasattr(params, 'class_weights') and params.class_weights is not None:
            self.register_buffer('class_weights_buf', params.class_weights)
        else:
            self.class_weights_buf = None

        # Pair attention
        self.use_pair_attention = getattr(params, 'use_pair_attention', False)
        if self.use_pair_attention:
            self.pair_query_proj = nn.Sequential(
                nn.Linear(64 * 2, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
            self.pair_key_proj = nn.Linear(64, 64)
            self.cn_proj = nn.Linear(64, self.emb_dim)
            nn.init.kaiming_uniform_(self.cn_proj.weight)
            nn.init.zeros_(self.cn_proj.bias)
            self.cn_alpha = nn.Parameter(torch.tensor(0.1))
            self.max_neighbors = getattr(params, 'max_neighbors', 64)

        # Drug-pair context injection
        self.use_pair_context = getattr(params, 'use_pair_context', False)
        if self.use_pair_context:
            self.pair_context_proj = nn.Sequential(
                nn.Linear(64 * 2, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, self.kg_emb_dim)
            )

        # Physicochemical descriptors
        self.use_desc = getattr(params, 'use_desc', False)
        if self.use_desc:
            desc_dim = getattr(params, 'desc_dim', 187)
            self.desc_proj = nn.Sequential(
                nn.Linear(desc_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU()
            )
            nn.init.kaiming_uniform_(self.desc_proj[0].weight)
            nn.init.zeros_(self.desc_proj[0].bias)
        self.desc_cache = None

        # Inductive safety: unseen entity embedding mask
        self.register_buffer('_unseen_emb_mask', torch.zeros(self.num_drugs, dtype=torch.bool))

        self.prop_drop_prob = getattr(params, 'prop_drop_prob', 0.0)
        self.inductive_setting = getattr(params, 'inductive_setting', None)
        self.ddi_drop_prob = getattr(params, 'ddi_drop_prob', 0.0)

    def mark_unseen_entities(self, unseen_ids):
        """Called by Trainer in inductive mode: mark unseen entities."""
        ids = torch.as_tensor(unseen_ids, dtype=torch.long).clamp(max=self.num_drugs - 1)
        self._unseen_emb_mask[ids] = True

    def _safe_emb(self, ids):
        """unseen -> fill with descriptor projection; seen -> normal embedding."""
        if self._inductive:
            if self.use_desc and self.desc_cache is not None:
                return self.desc_proj(self.desc_cache[ids])
            else:
                return torch.zeros(ids.size(0), self.kg_emb_dim, device=ids.device)
        emb = self.drug_embeddings(ids)
        if self._unseen_emb_mask.any():
            m = self._unseen_emb_mask[ids]
            if m.any():
                if self.use_desc and self.desc_cache is not None:
                    fill = self.desc_proj(self.desc_cache[ids[m]])
                else:
                    fill = torch.zeros(m.sum(), 64, device=emb.device)
                emb = emb.clone()
                emb[m] = fill
        return emb

    def forward(self, kg_graph, molecular_graphs, frag_graphs, drug_pairs, labels=None, mode='train',
                heads_neg=None, tails_neg=None, batch_mol_indices=None):
        # heads_neg/tails_neg/batch_mol_indices: [DEPRECATED] kept for data pipeline compatibility
        return self.forward_transductive(
            kg_graph, molecular_graphs, frag_graphs, drug_pairs,
            labels=labels, mode=mode)

    def _get_cached_rel_features(self, mode='train'):
        if self._rel_feat_cache is None or self._cache_epoch != self._current_epoch:
            self._rel_feat_cache = self.KG_encoder.get_unconditioned_rel_features(mode).detach()
            self._cache_epoch = self._current_epoch
        return self._rel_feat_cache

    def _uncond_propagate(self, kg_graph, heads, tails, rel_feat_global, emb_heads=None, emb_tails=None, ddi_drop=False):
        B = heads.size(0)
        device = heads.device
        if emb_heads is None: emb_heads = heads
        if emb_tails is None: emb_tails = tails

        # Drug-pair context injection
        if self.use_pair_context:
            h_emb = self.emb_dropout(self._safe_emb(emb_heads))
            t_emb = self.emb_dropout(self._safe_emb(emb_tails))
            pair_context = self.pair_context_proj(
                torch.cat([h_emb, t_emb], dim=-1))
            rel_feat_global = rel_feat_global.unsqueeze(0) + pair_context.unsqueeze(1)
        else:
            rel_feat_global = rel_feat_global.unsqueeze(0).expand(B, -1, -1)

        fwd_mask = kg_graph.fwd_mask.to(device)
        bwd_mask = kg_graph.bwd_mask.to(device)
        edge_index = kg_graph.edge_index.to(device)
        edge_type = kg_graph.edge_type.to(device)

        # DDI Dropout: filter edges where both endpoints are drugs
        if ddi_drop:
            src, dst = edge_index[0], edge_index[1]
            is_ddi = (src < self.num_drugs) & (dst < self.num_drugs)
            fwd_mask = fwd_mask & (~is_ddi)
            bwd_mask = bwd_mask & (~is_ddi)

        num_base = getattr(kg_graph, 'num_base_relations', self.num_base_rels)
        num_nodes = getattr(kg_graph, 'num_nodes', self.KG_encoder.num_entities)

        # Forward graph
        fwd_ei, fwd_et = edge_index[:, fwd_mask], edge_type[fwd_mask]
        if fwd_ei.size(1) > 0:
            _, order = torch.sort(fwd_ei[1])
            fwd_ei, fwd_et = fwd_ei[:, order], fwd_et[order]
        kg_data_fwd = AdaptData(edge_index=fwd_ei, edge_type=fwd_et,
                                num_nodes=num_nodes, num_relations=2 * num_base)

        # Backward graph
        shift = num_base
        bwd_ei = torch.stack([edge_index[1, bwd_mask], edge_index[0, bwd_mask]])
        bwd_et = edge_type[bwd_mask] - shift
        if bwd_ei.size(1) > 0:
            _, order = torch.sort(bwd_ei[1])
            bwd_ei, bwd_et = bwd_ei[:, order], bwd_et[order]
        kg_data_bwd = AdaptData(edge_index=bwd_ei, edge_type=bwd_et,
                                num_nodes=num_nodes, num_relations=2 * num_base)

        # Relation matrices
        rel_mat_fwd = rel_feat_global[:num_base]
        rel_mat_bwd = torch.cat([
            rel_feat_global[:self.n_rel],
            rel_feat_global[num_base:]
        ], dim=0)

        # Shared source/sink signals
        src_signal = self.KG_encoder.entity_encoder.shared_src.expand(B, -1)
        snk_signal = self.KG_encoder.entity_encoder.shared_sink.expand(B, -1)

        node_feat_fwd = self.KG_encoder.entity_encoder.bellmanford(
            kg_data_fwd, heads, src_signal, rel_mat_fwd)

        node_feat_bwd = self.KG_encoder.entity_encoder.bellmanford(
            kg_data_bwd, tails, snk_signal, rel_mat_bwd)

        b_idx = torch.arange(B, device=device)

        head_uncond = torch.cat([
            node_feat_fwd[b_idx, heads],
            node_feat_bwd[b_idx, heads]
        ], dim=-1)

        tail_uncond = torch.cat([
            node_feat_fwd[b_idx, tails],
            node_feat_bwd[b_idx, tails]
        ], dim=-1)

        return head_uncond, tail_uncond

    def forward_transductive(self, kg_graph, molecular_graphs, frag_graphs,
                             drug_pairs, labels=None, mode='train',
                             heads_neg=None, tails_neg=None):
        heads, tails, rels = drug_pairs[:, 0], drug_pairs[:, 1], drug_pairs[:, 2]
        B = heads.size(0)
        device = heads.device

        # Inductive safety: use graph-space IDs for propagation; original IDs for other components
        if self._inductive and getattr(self, 'orig2trainA', None) is not None:
            _m = self.orig2trainA if self.training else self.orig2infB
            heads_g = _m[heads]
            tails_g = _m[tails]
        else:
            heads_g, tails_g = heads, tails

        # Shared: molecular features + physicochemical descriptors
        mol_emb = self.MolCov(molecular_graphs)
        frag_emb, _ = self.MG_encoder(frag_graphs)

        if self.entity2mol_idx is not None:
            idx_h = self.entity2mol_idx[heads]
            idx_t = self.entity2mol_idx[tails]
            has_mol_h = (idx_h != -1)
            has_mol_t = (idx_t != -1)
            mol_h = mol_emb[idx_h.clamp(min=0)]
            mol_t = mol_emb[idx_t.clamp(min=0)]
            mol_h = torch.where(has_mol_h.unsqueeze(1), mol_h, torch.zeros_like(mol_h))
            mol_t = torch.where(has_mol_t.unsqueeze(1), mol_t, torch.zeros_like(mol_t))
        else:
            mol_h = mol_emb[heads]
            mol_t = mol_emb[tails]

        # Physicochemical descriptors
        if self.use_desc and self.desc_cache is not None:
            desc_h = self.desc_proj(self.desc_cache[heads])
            desc_t = self.desc_proj(self.desc_cache[tails])
        else:
            desc_h = torch.zeros(B, 64, device=device)
            desc_t = torch.zeros(B, 64, device=device)

        # Unconditioned propagation
        _use_ddi_drop = False
        if (self._inductive and self.training
                and getattr(self, 'ddi_drop_prob', 0.0) > 0
                and getattr(self, 'inductive_setting', None) == 'S1'):
            _use_ddi_drop = (torch.rand(1).item() < self.ddi_drop_prob)

        rel_feat_global = self._get_cached_rel_features(mode)
        head_uncond, tail_uncond = self._uncond_propagate(
            kg_graph, heads_g, tails_g, rel_feat_global,
            emb_heads=heads, emb_tails=tails,
            ddi_drop=_use_ddi_drop)

        # Pair attention (inject cross information)
        if self.use_pair_attention:
            head_uncond, tail_uncond = self._pair_attention_enhance(
                heads, tails, head_uncond, tail_uncond)

        # Propagation-level dropout (training only)
        if self.training and self.prop_drop_prob > 0:
            drop_h = (torch.rand(B, 1, device=device) > self.prop_drop_prob).float()
            drop_t = (torch.rand(B, 1, device=device) > self.prop_drop_prob).float()
            head_uncond = head_uncond * drop_h
            tail_uncond = tail_uncond * drop_t

        # Entity embedding + bilinear interaction
        h_emb = self.emb_dropout(self._safe_emb(heads))
        t_emb = self.emb_dropout(self._safe_emb(tails))
        interaction = self.bilinear(h_emb, t_emb)

        # Fusion layer (W_d_trans: fixed 576-dim input: mol 256 + uncond 256 + emb 64)
        drugA = self.W_d_trans(torch.cat((mol_h, head_uncond, h_emb), dim=-1))
        drugB = self.W_d_trans(torch.cat((mol_t, tail_uncond, t_emb), dim=-1))

        # Relation prediction head
        pair_feat = self.pair_proj(torch.cat((head_uncond, tail_uncond), dim=-1))
        logits_rel = self.rel_classifier(pair_feat)

        # Joint features
        uncond_pair = self.uncond_proj(torch.cat((head_uncond, tail_uncond), dim=-1))
        if self.use_desc:
            joint_feat = torch.cat((frag_emb, uncond_pair, drugA, drugB, interaction, desc_h, desc_t), dim=-1)
        else:
            joint_feat = torch.cat((frag_emb, uncond_pair, drugA, drugB, interaction), dim=-1)
        logits_main = self.joint_classifier(joint_feat)

        # Loss computation
        if self.training and labels is not None:
            loss_main = self._focal_loss_impl(logits_main, labels)

            if self.class_weights_buf is not None:
                loss_rel_cls = F.cross_entropy(logits_rel, labels, weight=self.class_weights_buf)
            else:
                loss_rel_cls = F.cross_entropy(logits_rel, labels)

            total_loss = loss_main + 0.1 * loss_rel_cls

            loss_dict = {
                "main": loss_main.item(),
                "kl": loss_rel_cls.item(),
                "kg": 0.0
            }
            return logits_main, total_loss, loss_dict
        else:
            return logits_main, torch.tensor(0.0, device=device), {}

    def _pair_attention_enhance(self, heads, tails, head_uncond, tail_uncond):
        """Pair attention: use drug pair as query to selectively aggregate 1-hop neighbor info."""
        h_emb = self._safe_emb(heads)
        t_emb = self._safe_emb(tails)
        pair_query = self.pair_query_proj(torch.cat([h_emb, t_emb], dim=-1))

        h_agg = self._attend_neighbors(heads, pair_query)
        t_agg = self._attend_neighbors(tails, pair_query)

        h_agg_proj = self.cn_proj(h_agg)
        t_agg_proj = self.cn_proj(t_agg)

        alpha = torch.sigmoid(self.cn_alpha)
        head_uncond = head_uncond + alpha * h_agg_proj
        tail_uncond = tail_uncond + alpha * t_agg_proj
        return head_uncond, tail_uncond

    def _attend_neighbors(self, entity_ids, pair_query):
        """Attention-based aggregation over 1-hop neighbors of given entities."""
        B = entity_ids.size(0)
        device = entity_ids.device

        neighbor_idx = self.neighbor_indices[entity_ids]
        safe_idx = neighbor_idx.clamp(min=0)
        neighbor_emb = self._safe_emb(safe_idx)

        valid_mask = (neighbor_idx != -1)

        key = self.pair_key_proj(neighbor_emb)
        attn_scores = torch.bmm(
            pair_query.unsqueeze(1),
            key.transpose(1, 2)
        ).squeeze(1) / 8.0

        attn_scores = attn_scores.masked_fill(~valid_mask, float('-inf'))

        all_pad = ~valid_mask.any(dim=1)
        if all_pad.any():
            attn_scores[all_pad] = 0.0

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~valid_mask, 0.0)

        agg_feat = torch.bmm(
            attn_weights.unsqueeze(1),
            neighbor_emb
        ).squeeze(1)
        agg_feat = agg_feat * valid_mask.any(dim=1, keepdim=True)

        return agg_feat

    def _precompute_neighbor_indices(self, global_graph, num_drugs, max_neighbors=64):
        """Precompute 1-hop neighbor indices for each drug node."""
        device = self.device
        edge_index = global_graph.edge_index.to(device)
        src, dst = edge_index[0], edge_index[1]

        valid_mask = (src < num_drugs) & (dst < num_drugs) & (src != dst)
        src_valid, dst_valid = src[valid_mask], dst[valid_mask]

        edge_pairs = torch.unique(torch.stack([src_valid, dst_valid], dim=1), dim=0)
        src_unique, dst_unique = edge_pairs[:, 0], edge_pairs[:, 1]

        neighbor_indices = torch.full((num_drugs, max_neighbors), -1,
                                      dtype=torch.long, device=device)
        neighbor_counts = torch.zeros(num_drugs, dtype=torch.long, device=device)

        sort_order = torch.argsort(src_unique)
        sorted_src = src_unique[sort_order]
        sorted_dst = dst_unique[sort_order]

        node_starts = torch.searchsorted(sorted_src, torch.arange(num_drugs, device=device))
        node_ends = torch.searchsorted(sorted_src, torch.arange(num_drugs, device=device), right=True)

        for node in range(num_drugs):
            s, e = node_starts[node].item(), node_ends[node].item()
            if s == e:
                continue
            neighbors = sorted_dst[s:e]
            n = min(len(neighbors), max_neighbors)
            neighbor_indices[node, :n] = neighbors[:n]
            neighbor_counts[node] = n

        return neighbor_indices, neighbor_counts

    def _focal_loss_impl(self, logits, labels):
        num_classes = logits.size(-1)
        gamma = 2.5
        label_smoothing = 0.03

        if label_smoothing > 0:
            smooth_targets = torch.full_like(logits, label_smoothing / (num_classes - 1))
            smooth_targets[torch.arange(labels.size(0), device=labels.device), labels] = 1.0 - label_smoothing
            ce = torch.sum(-smooth_targets * F.log_softmax(logits, dim=-1), dim=-1)
        else:
            ce = F.cross_entropy(logits, labels, reduction='none')

        pt = torch.exp(-ce).clamp(1e-7, 1 - 1e-7)
        return ((1 - pt) ** gamma * ce).mean()
