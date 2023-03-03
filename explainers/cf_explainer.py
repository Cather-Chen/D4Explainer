
from math import sqrt

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

EPS = 1e-15


class MetaCFExplainer(torch.nn.Module):
    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 0.5,
    }

    def __init__(self, model, epochs=1000, lr=0.01, log=True, task="gc"):
        super(MetaCFExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.task = task

    def __set_masks__(self, x, edge_index, init="normal"):

        N = x.size(0)
        E = edge_index.size(1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def __loss__(self, log_logits, pred_label):
        loss = -F.nll_loss(log_logits,pred_label)
        # m = self.edge_mask.sigmoid()
        # loss = loss + self.coeffs['edge_size'] * m.sum()
        # ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        # loss = loss + self.coeffs['edge_ent'] * ent.mean()
        return loss

    def explain_graph(self, graph, **kwargs):

        # get the initial prediction.
        with torch.no_grad():
            if self.task == "nc":
                soft_pred, _ = self.model.get_node_pred_subgraph(x=graph.x, edge_index=graph.edge_index,
                                                                 mapping=graph.mapping)
            else:
                soft_pred, _ = self.model.get_pred(x=graph.x, edge_index=graph.edge_index,
                                                   batch=graph.batch)
            pred_label = soft_pred.argmax(dim=-1)

        N = graph.x.size(0)
        E = graph.edge_index.size(1)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        self.to(graph.x.device)
        optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)

        for epoch in range(1, self.epochs + 1):

            optimizer.zero_grad()
            if self.task == "nc":
                output_prob, output_repr= self.model.get_pred_explain(x=graph.x, edge_index=graph.edge_index,
                                                                                edge_mask=self.edge_mask,
                                                                                mapping=graph.mapping)
            else:
                output_prob, output_repr = self.model.get_pred_explain(x=graph.x, edge_index=graph.edge_index,
                                                                                    edge_mask=self.edge_mask,
                                                                                    batch=graph.batch)
            log_logits = F.log_softmax(output_repr, dim=-1)
            loss = self.__loss__(log_logits, pred_label)
            loss.backward()
            optimizer.step()

        edge_mask = self.edge_mask.detach().sigmoid()

        return edge_mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'


import math

import numpy as np

from explainers.base import Explainer


class CF_Explainer(Explainer):

    def __init__(self, device, gnn_model_path, task):
        super(CF_Explainer, self).__init__(device, gnn_model_path, task)

    def explain_graph(self, graph,
                      model=None,
                      epochs=100,
                      lr=1e-2,
                      draw_graph=0,
                      vis_ratio=0.2
                      ):

        if model == None:
            model = self.model

        explainer = MetaCFExplainer(model, epochs=epochs, lr=lr, task=self.task)
        edge_imp = explainer.explain_graph(graph)
        edge_imp = self.norm_imp(edge_imp.cpu().numpy())

        if draw_graph:
            self.visualize(graph, edge_imp, self.name, vis_ratio=vis_ratio)
        self.last_result = (graph, edge_imp)

        return edge_imp

    def pack_explanatory_subgraph(self, top_ratio=0.2,
                                  graph=None, imp=None, relabel=False, if_cf=True):
        ratio_cf = 1-top_ratio
        if graph is None:
            graph, imp = self.last_result
        assert len(imp) == graph.num_edges, 'length mismatch'

        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]
        exp_subgraph = graph.clone()
        exp_subgraph.y = graph.y if self.task == "gc" else graph.self_y
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = min(max(math.ceil(ratio_cf * Gi_n_edge), 1), Gi_n_edge)
            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
        try:
            exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        except:
            pass
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        # .... the nodes.
        exp_subgraph.x = graph.x
        if relabel:
            exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch, exp_subgraph.pos = self.__relabel__(exp_subgraph, exp_subgraph.edge_index)
        return exp_subgraph