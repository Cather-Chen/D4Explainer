import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from explainers.base import Explainer

from .common import EdgeMaskNet

EPS = 1e-6


class PGExplainer(Explainer):
    coeffs = {
        "edge_size": 1e-4,
        "edge_ent": 1e-2,
    }

    def __init__(
        self,
        device,
        gnn_model,
        task,
        n_in_channels=14,
        e_in_channels=3,
        hid=64,
        n_layers=2,
        n_label=2,
    ):
        super(PGExplainer, self).__init__(device, gnn_model, task)

        self.device = device
        self.edge_mask_model = EdgeMaskNet(n_in_channels, e_in_channels, hid=hid, n_layers=n_layers).to(self.device)
        self.epoch = 1000
        self.lr = 0.01

    def __set_masks__(self, mask, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = mask

    def __clear_masks__(self, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

    def __setup_target_label__(self, graph):
        if not hasattr(graph, "hat_y"):
            graph.hat_y = self.model(graph).argmax(-1).to(graph.x.device)
        return graph

    def __reparameterize__(self, log_alpha, beta=0.1, training=True):
        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def __loss__(self, log_logits, mask, pred_label):
        # loss = criterion(log_logits, pred_label)
        idx = [i for i in range(len(pred_label))]
        loss = -log_logits.softmax(dim=1)[idx, pred_label.view(-1)].sum()

        loss = loss + self.coeffs["edge_size"] * mask.mean()
        ent = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        loss = loss + self.coeffs["edge_ent"] * ent.mean()
        return loss

    def __cfloss__(self, pred, mask, pred_label):
        bsz, n = pred.size(0), pred.size(1)
        inf_diag = torch.diag(-torch.ones((n)) / 0).unsqueeze(0).repeat(bsz, 1, 1).to(pred.device)
        neg_prop = (pred.unsqueeze(1).expand(bsz, n, n) + inf_diag).logsumexp(-1) - pred.logsumexp(-1).unsqueeze(
            1
        ).repeat(1, n)
        criterion = torch.nn.NLLLoss()
        loss_cf = criterion(neg_prop, pred_label)
        loss_cf = loss_cf + self.coeffs["edge_size"] * mask.mean()
        ent = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        loss_cf = loss_cf + self.coeffs["edge_ent"] * ent.mean()
        return loss_cf

    # batch version
    def pack_subgraph(self, graph, imp, top_ratio=0.2):
        if abs(top_ratio - 1.0) < EPS:
            return graph, imp

        exp_subgraph = copy.deepcopy(graph)
        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]

        # extract ego graph
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = max(math.ceil(top_ratio * Gi_n_edge), 1)

            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])

        exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        (
            exp_subgraph.x,
            exp_subgraph.edge_index,
            exp_subgraph.batch,
            _,
        ) = self.__relabel__(exp_subgraph, exp_subgraph.edge_index)

        return exp_subgraph, imp[top_idx]

    def get_mask(self, graph):
        # batch version
        graph_map = graph.batch[graph.edge_index[0, :]]
        mask = torch.FloatTensor([]).to(self.device)
        for i in range(graph.num_graphs):
            edge_indicator = (graph_map == i).bool()
            G_i_mask = self.edge_mask_model(graph.x, graph.edge_index[:, edge_indicator]).view(-1)
            mask = torch.cat([mask, G_i_mask])
        return mask

    def get_pos_edge(self, graph, mask, ratio):
        num_edge = [0]
        num_node = [0]
        sep_edge_idx = []
        graph_map = graph.batch[graph.edge_index[0, :]]
        pos_idx = torch.LongTensor([])
        mask = mask.detach().cpu()
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = max(math.ceil(ratio * Gi_n_edge), 1)

            Gi_pos_edge_idx = np.argsort(-mask[edge_indicator])[:topk]

            pos_idx = torch.cat([pos_idx, edge_indicator[Gi_pos_edge_idx]])
            num_edge.append(num_edge[i] + Gi_n_edge)
            num_node.append(num_node[i] + (graph.batch == i).sum().long())
            sep_edge_idx.append(Gi_pos_edge_idx)

        return pos_idx, num_edge, num_node, sep_edge_idx

    def explain_graph(self, graph, model=None, temp=0.1, draw_graph=0, vis_ratio=0.2, train_mode=False):
        """
        Explain the graph using PGExplainer
        :param graph: the graph to be explained
        :param model: the model to be explained
        :param temp: the temperature for the reparameterization trick
        :param draw_graph: whether to draw the graph
        :param vis_ratio: the ratio of edges to be visualized
        :param train_mode: whether to train the explainer
        """
        if model is not None:
            self.model = model
        if self.task == "nc":
            soft_pred, _ = self.model.get_node_pred_subgraph(
                x=graph.x, edge_index=graph.edge_index, mapping=graph.mapping
            )
        else:
            soft_pred, _ = self.model.get_pred(x=graph.x, edge_index=graph.edge_index, batch=graph.batch)
        y_pred = soft_pred.argmax(dim=-1)
        optimizer = torch.optim.Adam(self.edge_mask_model.parameters(), lr=self.lr)
        for _ in range(self.epoch):
            ori_mask = self.get_mask(graph)
            edge_mask = self.__reparameterize__(ori_mask, training=train_mode, beta=temp)
            if self.task == "nc":
                _, output_repr = self.model.get_pred_explain(
                    x=graph.x, edge_index=graph.edge_index, edge_mask=edge_mask, mapping=graph.mapping
                )
            else:
                _, output_repr = self.model.get_pred_explain(
                    x=graph.x, edge_index=graph.edge_index, edge_mask=edge_mask, batch=graph.batch
                )

            log_logits = F.log_softmax(output_repr)
            criterion = torch.nn.NLLLoss()
            loss = criterion(log_logits, y_pred)

            loss.backward()
            optimizer.step()

        imp = edge_mask.detach().cpu().numpy()
        # self.__clear_masks__(self.model)
        if draw_graph:
            self.visualize(graph, imp, self.name, vis_ratio=vis_ratio)
        self.last_result = (graph, imp)

        return imp

    def pack_explanatory_subgraph(self, top_ratio=0.2, graph=None, imp=None, relabel=False, if_cf=False):
        """
        Pack the explanatory subgraph from the original graph
        :param top_ratio: the ratio of edges to be selected
        :param graph: the original graph
        :param imp: the attribution scores for edges
        :param relabel: whether to relabel the nodes in the explanatory subgraph
        :param if_cf: whether to use the CF method
        :return: the explanatory subgraph
        """
        if graph is None:
            graph, imp = self.last_result
        assert len(imp) == graph.num_edges, "length mismatch"

        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]
        exp_subgraph = graph.clone()
        exp_subgraph.y = graph.y if self.task == "gc" else graph.self_y
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = min(max(math.ceil(top_ratio * Gi_n_edge), 1), Gi_n_edge)
            if not if_cf:
                Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            else:
                Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[topk:]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
        try:
            exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        except Exception:
            pass
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]

        exp_subgraph.x = graph.x
        if relabel:
            (
                exp_subgraph.x,
                exp_subgraph.edge_index,
                exp_subgraph.batch,
                exp_subgraph.pos,
            ) = self.__relabel__(exp_subgraph, exp_subgraph.edge_index)
        return exp_subgraph
