import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import dgl
import dgl.nn as dglnn
from .layers import GeneralizedRelationalConv, BaseNBFNet
from torch_geometric.data import Data
from utils.graph_utils import AdaptData
import numpy as np


class MG_encoder(nn.Module):
    def __init__(self, frag_dim, hidden_dim1, hidden_dim2, emb_dim, num_heads):
        super(MG_encoder, self).__init__()

        self.frag_dim = frag_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.emb_dim = emb_dim

        self.num_heads = num_heads

        self.layer1 = dglnn.HeteroGraphConv({
            'inter': dglnn.GraphConv(self.frag_dim, self.hidden_dim1 * self.num_heads),
            'cross': dglnn.GATConv(self.frag_dim, self.hidden_dim1, num_heads=self.num_heads, activation=None)},
            aggregate=self._my_agg_func)

        self.layer2 = dglnn.HeteroGraphConv({
            'inter': dglnn.GraphConv(self.hidden_dim1 * self.num_heads, self.hidden_dim2 * self.num_heads),
            'cross': dglnn.GATConv(self.hidden_dim1 * self.num_heads, self.hidden_dim2, num_heads=self.num_heads,
                                   activation=None)},
            aggregate=self._my_agg_func)

        self.layer3_inter = dglnn.GraphConv(self.hidden_dim2 * self.num_heads, self.emb_dim, allow_zero_in_degree=True)
        self.layer3_cross = dglnn.GATConv(self.hidden_dim2 * self.num_heads, self.emb_dim, num_heads=1, activation=None)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim1 * self.num_heads)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim2 * self.num_heads)
        self.bn3 = nn.BatchNorm1d(self.emb_dim)

    def _my_agg_func(self, outputs, dsttype):
        tensor = []
        for data in outputs:
            tensor.append(data.view(data.shape[0], -1))
        stacked = torch.stack(tensor, dim=0)
        return torch.sum(stacked, dim=0)

    def forward(self, frag_graphs):
        h = frag_graphs.ndata['feat']

        h = self.layer1(frag_graphs, {'drug': h})
        h = F.leaky_relu(h['drug'], negative_slope=0.2)
        h = self.bn1(h.view(h.shape[0], -1))

        h = self.layer2(frag_graphs, {'drug': h})
        h = F.leaky_relu(h['drug'], negative_slope=0.2)
        h = self.bn2(h.view(h.shape[0], -1))

        h1 = self.layer3_inter(frag_graphs['inter'], h)
        h2, attention_weights = self.layer3_cross(frag_graphs['cross'], (h, h), get_attention=True)
        h = (h1 + h2.squeeze()).squeeze()
        h = self.bn3(h)

        frag_graphs.ndata['h'] = h

        src_nodes, _ = frag_graphs['cross'].edges()

        node_out_attention_sum = scatter_add(attention_weights.squeeze(), src_nodes, dim=0)
        frag_graphs.ndata['weighted_h'] = h * node_out_attention_sum.unsqueeze(1)

        # Fragment graph readout (compatible with multiple DGL versions)
        if hasattr(frag_graphs, 'batch_num_nodes'):
            graph_list = dgl.unbatch(frag_graphs)
            embs = []
            for g in graph_list:
                emb = dgl.readout_nodes(g, 'weighted_h', op='mean', ntype='drug')
                if emb.dim() == 2 and emb.size(0) == 1:
                    emb = emb.squeeze(0)
                elif emb.dim() > 1:
                    emb = emb.view(-1)
                embs.append(emb)
            frag_embeddings = torch.stack(embs)
        else:
            frag_embeddings = dgl.readout_nodes(frag_graphs, 'weighted_h', op='mean', ntype='drug')
            if frag_embeddings.dim() == 2 and frag_embeddings.size(0) == 1:
                frag_embeddings = frag_embeddings.squeeze(0)
            elif frag_embeddings.dim() > 1:
                frag_embeddings = frag_embeddings.view(-1)
            frag_embeddings = frag_embeddings.unsqueeze(0)

        return frag_embeddings, node_out_attention_sum


class BidirectionalRelNBFNet(BaseNBFNet):
    """
    Bidirectional relation graph NBFNet - processes relation graph only, not entity graph.
    """

    def __init__(self, input_dim, hidden_dims, num_relation=4,
                 learnable_signal=True, signal_dim=None, relation_stats=None,
                 avg_opposite_rels=False, num_ddi=None, num_base=None, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        # Dual-channel dimension: source signal + sink signal
        self.signal_dim = signal_dim if signal_dim else input_dim // 2
        self.input_dim = self.signal_dim * 2
        self.dims = [self.input_dim] + list(hidden_dims)

        # Rebuild layers
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=False
                )
            )

        # Source/sink signals
        self.learnable_signal = learnable_signal
        if self.learnable_signal:
            self.src_embed = nn.Embedding(1, self.signal_dim)
            self.sink_embed = nn.Embedding(1, self.signal_dim)
            nn.init.xavier_uniform_(self.src_embed.weight)
            nn.init.xavier_uniform_(self.sink_embed.weight)
        else:
            self.register_buffer('src_const', torch.ones(1, self.signal_dim))
            self.register_buffer('sink_const', torch.ones(1, self.signal_dim) * -1)

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + self.input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, self.input_dim)
            )

        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU()
        )
        nn.init.xavier_uniform_(self.output_projection[0].weight)
        if self.output_projection[0].bias is not None:
            nn.init.zeros_(self.output_projection[0].bias)

        # Relation statistics projection
        self.relation_stats = relation_stats
        if relation_stats is not None:
            if isinstance(relation_stats, np.ndarray):
                relation_stats = torch.tensor(relation_stats, dtype=torch.float32)
            self.register_buffer('rel_stats', relation_stats)
            stat_dim = relation_stats.shape[1]
            self.rel_init_proj = nn.Sequential(
                nn.Linear(stat_dim, self.signal_dim),
                nn.LayerNorm(self.signal_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.signal_dim, self.signal_dim),
                nn.LayerNorm(self.signal_dim),
                nn.ReLU(),
                nn.Linear(self.signal_dim, self.input_dim)
            )
        else:
            self.rel_init_proj = None

        self.avg_opposite_rels = avg_opposite_rels
        self.num_ddi = num_ddi
        self.num_base = num_base

    def bellmanford(self, data, head_rel_indices, tail_rel_indices):
        batch_size = len(head_rel_indices)
        device = head_rel_indices[0].device

        if self.learnable_signal:
            src_signal = self.src_embed.weight.expand(batch_size, -1)
            sink_signal = self.sink_embed.weight.expand(batch_size, -1)
        else:
            src_signal = self.src_const.expand(batch_size, -1)
            sink_signal = self.sink_const.expand(batch_size, -1)

        # Build boundary condition [batch, num_relations, input_dim]
        boundary = torch.zeros(batch_size, data.num_nodes, self.input_dim, device=device)

        # Fill head relations (source signal -> first half dimensions)
        h_lens = torch.tensor([len(r) for r in head_rel_indices], device=device)
        if h_lens.sum() > 0:
            h_idx = torch.cat(head_rel_indices)
            h_batch = torch.repeat_interleave(torch.arange(batch_size, device=device), h_lens)
            boundary[h_batch, h_idx, :self.signal_dim] = src_signal[h_batch]

        # Fill tail relations (sink signal -> second half dimensions)
        t_lens = torch.tensor([len(r) for r in tail_rel_indices], device=device)
        if t_lens.sum() > 0:
            t_idx = torch.cat(tail_rel_indices)
            t_batch = torch.repeat_interleave(torch.arange(batch_size, device=device), t_lens)
            boundary[t_batch, t_idx, self.signal_dim:] = sink_signal[t_batch]

        # Inject structural initialization bias for all relation nodes
        if self.rel_init_proj is not None:
            rel_init = self.rel_init_proj(self.rel_stats)
            boundary = boundary + rel_init.unsqueeze(0)

        query = torch.zeros(batch_size, self.dims[0], device=device)
        size = (data.num_nodes, data.num_nodes)
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            edge_weight = data.edge_weight.to(device)
        else:
            edge_weight = torch.ones(data.num_edges, device=device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights
        }

    def forward(self, rel_graph, head_rel_indices, tail_rel_indices):
        node_feature = self.bellmanford(rel_graph, head_rel_indices, tail_rel_indices)["node_feature"]
        node_feature = self.output_projection(node_feature)  # [B, N, D]
        if self.avg_opposite_rels:
            num_ddi = self.num_ddi
            shift = self.num_base
            fwd = node_feature[:, :num_ddi, :]
            inv = node_feature[:, shift:shift + num_ddi, :]
            avg = (fwd + inv) / 2.0
            node_feature = torch.cat([
                avg,
                node_feature[:, num_ddi:shift, :],
                avg,
                node_feature[:, shift + num_ddi:, :]
            ], dim=1)
        return node_feature


class EntityNBFNet_Feature(BaseNBFNet):
    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation, self.dims[0],
                    self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True
                )
            )
        self.register_parameter('shared_src', nn.Parameter(torch.randn(input_dim) * 0.01))
        self.register_parameter('shared_sink', nn.Parameter(torch.randn(input_dim) * 0.01))

    def bellmanford(self, data, start_idx, init_signal, relation_embeddings, separate_grad=False):
        batch_size = init_signal.shape[0]
        device = init_signal.device
        dim = self.dims[0]
        num_nodes = data.num_nodes

        if data.edge_index.numel() == 0:
            out_dim = self.dims[-1] + dim
            return torch.zeros(batch_size, num_nodes, out_dim, device=device)

        boundary = torch.zeros(batch_size, num_nodes, dim, device=device)
        boundary.scatter_(1, start_idx.view(batch_size, 1, 1).expand(-1, 1, dim), init_signal.unsqueeze(1))

        size = (num_nodes, num_nodes)
        edge_weight = torch.ones(data.num_edges, device=device)
        layer_input = boundary
        hiddens = []

        for layer in self.layers:
            if separate_grad: edge_weight = edge_weight.clone().requires_grad_()
            hidden = layer(layer_input, init_signal, boundary, data.edge_index,
                           data.edge_type, size, edge_weight, relation_override=relation_embeddings)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        node_query = init_signal.unsqueeze(1).expand(-1, num_nodes, -1)
        if self.concat_hidden:
            return torch.cat(hiddens + [node_query], dim=-1)
        return torch.cat([hiddens[-1], node_query], dim=-1)

    def forward(self, data_fwd, data_bwd, h_idx, t_idx, rel_emb_fwd, rel_emb_bwd, src_sig, snk_sig):
        feat_src = self.bellmanford(data_fwd, h_idx, src_sig, rel_emb_fwd)
        feat_snk = self.bellmanford(data_bwd, t_idx, snk_sig, rel_emb_bwd)
        return torch.cat([feat_src, feat_snk], dim=-1)


class UltraKGEncoder_Bi(nn.Module):
    def __init__(self, kg_emb_dim, num_layers, num_relations, num_drugs, device,
                 train_relation_graph, inference_relation_graph=None,
                 entity_relation_adj=None, emb_dim=256,
                 num_entities=None, num_base_relations=None, num_meta_relations=8,
                 relation_stats=None, avg_opposite_rels=False,
                 num_ddi_rels=None):
        super().__init__()
        assert kg_emb_dim % 2 == 0
        self.kg_emb_dim = kg_emb_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.num_meta_relations = num_meta_relations
        self.num_drugs = num_drugs
        self.device = device

        assert num_entities is not None, "num_entities must be explicitly passed"
        assert num_base_relations is not None, "num_base_relations must be explicitly passed"
        self.num_entities = num_entities
        self.num_base_relations = num_base_relations

        self.num_rels = num_ddi_rels if num_ddi_rels is not None else num_relations

        # Relation encoder
        self.relation_encoder = BidirectionalRelNBFNet(
            input_dim=kg_emb_dim * 2, hidden_dims=[kg_emb_dim] * num_layers,
            num_relation=num_meta_relations,
            message_func="distmult", aggregate_func="mean",
            short_cut=True, layer_norm=True, learnable_signal=True, signal_dim=kg_emb_dim // 2,
            relation_stats=relation_stats,
            avg_opposite_rels=avg_opposite_rels,
            num_ddi=self.num_rels,
            num_base=self.num_base_relations
        )
        self.entity_encoder = EntityNBFNet_Feature(
            input_dim=kg_emb_dim, hidden_dims=[kg_emb_dim] * 1,
            num_relation=num_base_relations,
            message_func="distmult", aggregate_func="mean",
            short_cut=True, layer_norm=True, dependent=False, project_relations=True
        )

        if train_relation_graph is not None:
            self.register_buffer('train_rg_edge_index', train_relation_graph.edge_index.cpu())
            self.register_buffer('train_rg_edge_type', train_relation_graph.edge_type.cpu())
        if inference_relation_graph is not None:
            self.register_buffer('inf_rg_edge_index', inference_relation_graph.edge_index.cpu())
            self.register_buffer('inf_rg_edge_type', inference_relation_graph.edge_type.cpu())
        if entity_relation_adj is not None:
            def _safe_pad(indptr_np, max_nodes):
                t = torch.tensor(indptr_np, dtype=torch.long)
                target_len = max_nodes + 1
                if t.shape[0] < target_len:
                    pad_val = t[-1].item() if t.numel() > 0 else 0
                    return torch.cat([t, torch.full((target_len - t.shape[0],), pad_val, dtype=torch.long)])
                return t

            self.register_buffer('er_out_indptr', _safe_pad(entity_relation_adj['out']['indptr'], num_entities))
            self.register_buffer('er_out_indices',
                                 torch.tensor(entity_relation_adj['out']['indices'], dtype=torch.long))
            self.register_buffer('er_in_indptr', _safe_pad(entity_relation_adj['in']['indptr'], num_entities))
            self.register_buffer('er_in_indices', torch.tensor(entity_relation_adj['in']['indices'], dtype=torch.long))

        self.node_proj = nn.Linear(kg_emb_dim * 4, emb_dim)
        nn.init.kaiming_uniform_(self.node_proj.weight)
        if self.node_proj.bias is not None: nn.init.zeros_(self.node_proj.bias)

        self.register_buffer('relation_stats', torch.tensor(relation_stats, dtype=torch.float32))

        self.triplet_scorer = nn.Linear(kg_emb_dim * 4, 1)
        nn.init.xavier_uniform_(self.triplet_scorer.weight)

    def _build_fwd_graph(self, entity_graph):
        fwd_mask = entity_graph.fwd_mask
        edge_index = entity_graph.edge_index
        edge_type = entity_graph.edge_type
        fwd_ei, fwd_et = edge_index[:, fwd_mask], edge_type[fwd_mask]
        if fwd_ei.size(1) > 0:
            _, order = torch.sort(fwd_ei[1])
            fwd_ei = fwd_ei[:, order]
            fwd_et = fwd_et[order]
        num_nodes = entity_graph.num_nodes
        num_base_rels = entity_graph.num_base_relations
        return AdaptData(edge_index=fwd_ei, edge_type=fwd_et,
                         num_nodes=num_nodes,
                         num_relations=2 * num_base_rels)

    def _build_bwd_graph(self, entity_graph):
        fwd_mask = entity_graph.fwd_mask
        bwd_mask = entity_graph.bwd_mask
        edge_index = entity_graph.edge_index
        edge_type = entity_graph.edge_type
        shift = entity_graph.num_base_relations
        bwd_ei = torch.stack([edge_index[1, bwd_mask], edge_index[0, bwd_mask]])
        bwd_et = edge_type[bwd_mask] - shift
        if bwd_ei.size(1) > 0:
            _, order = torch.sort(bwd_ei[1])
            bwd_ei = bwd_ei[:, order]
            bwd_et = bwd_et[order]
        num_nodes = entity_graph.num_nodes
        num_base_rels = entity_graph.num_base_relations
        return AdaptData(edge_index=bwd_ei, edge_type=bwd_et,
                         num_nodes=num_nodes,
                         num_relations=2 * num_base_rels)

    def get_unconditioned_rel_features(self, mode='train'):
        """
        Get unconditioned global relation features.
        No specific (h,t) query signal is injected; relies solely on rel_init
        for multi-hop smoothing on the relation graph.
        """
        rg_edge_index = self.train_rg_edge_index if mode == 'train' else self.inf_rg_edge_index
        rg_edge_type = self.train_rg_edge_type if mode == 'train' else self.inf_rg_edge_type

        device = rg_edge_index.device
        num_rg_nodes = self.num_relations

        rel_graph = Data(edge_index=rg_edge_index.to(device).long(),
                         edge_type=rg_edge_type.to(device).long(),
                         num_nodes=num_rg_nodes)

        empty_h_list = [torch.tensor([], dtype=torch.long, device=device)]
        empty_t_list = [torch.tensor([], dtype=torch.long, device=device)]

        rel_feat_global = self.relation_encoder(rel_graph, empty_h_list, empty_t_list)

        return rel_feat_global.squeeze(0)

    def forward(self, entity_graph, head_idx, tail_idx, rel_idx, mode='train'):
        batch_size = head_idx.shape[0]
        device = head_idx.device

        # 1. Relation graph encoding: single-relation query for full relation representations
        rg_edge_index = self.train_rg_edge_index if mode == 'train' else self.inf_rg_edge_index
        rg_edge_type = self.train_rg_edge_type if mode == 'train' else self.inf_rg_edge_type
        num_rg_nodes = int(max(rg_edge_index.max().item() + 1, self.num_relations))
        rel_graph = Data(edge_index=rg_edge_index.to(device).long(),
                         edge_type=rg_edge_type.to(device).long(),
                         num_nodes=num_rg_nodes)
        h_list = [torch.tensor([r], device=device) for r in rel_idx]
        t_list = [torch.tensor([r], device=device) for r in rel_idx]
        rel_feat_all = self.relation_encoder(rel_graph, h_list, t_list)
        rel_repr = rel_feat_all[torch.arange(batch_size), rel_idx]

        # 2. Prepare dual signals
        src_signal = rel_repr
        snk_signal = rel_repr

        # 3. Build forward/backward subgraphs
        kg_data_fwd = self._build_fwd_graph(entity_graph)
        kg_data_bwd = self._build_bwd_graph(entity_graph)

        # 4. Bidirectional entity graph propagation
        node_feat_h2t = self.entity_encoder(
            kg_data_fwd, kg_data_bwd, head_idx, tail_idx,
            rel_feat_all, rel_feat_all, src_signal, snk_signal
        )
        t_exp = tail_idx.view(-1, 1, 1).expand(-1, 1, node_feat_h2t.shape[-1])
        emb_h2t = node_feat_h2t.gather(1, t_exp).squeeze(1)

        node_feat_t2h = self.entity_encoder(
            kg_data_bwd, kg_data_fwd, tail_idx, head_idx,
            rel_feat_all, rel_feat_all, snk_signal, src_signal
        )
        h_exp = head_idx.view(-1, 1, 1).expand(-1, 1, node_feat_t2h.shape[-1])
        emb_t2h = node_feat_t2h.gather(1, h_exp).squeeze(1)

        # 5. Triplet scoring
        score_fwd = self.triplet_scorer(emb_h2t).squeeze(-1)
        score_bwd = self.triplet_scorer(emb_t2h).squeeze(-1)
        score_avg = (score_fwd + score_bwd) / 2.0

        # 6. Relation-level representation
        kg_pair_raw = (emb_h2t + emb_t2h) / 2.0
        b_idx = torch.arange(batch_size, device=device)
        head_node_raw = node_feat_h2t[b_idx, head_idx]
        tail_node_raw = node_feat_h2t[b_idx, tail_idx]

        return (kg_pair_raw, self.node_proj(head_node_raw), self.node_proj(tail_node_raw),
                score_fwd, score_bwd, score_avg, rel_feat_all)
