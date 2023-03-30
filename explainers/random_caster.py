import random

import numpy as np
import torch

from explainers.base import Explainer


class RandomCaster(Explainer):
    def __init__(self, device, gnn_model_path, task, ds):
        super(RandomCaster, self).__init__(device, gnn_model_path, task)
        self.ds = ds

    def explain_graph(self, graph, model=None, ratio=0.1, draw_graph=0, vis_ratio=0.2):
        """
        Explain the graph using RandomCaster
        :param graph: the graph to be explained.
        :param model: the model to be explained. (not used)
        :param ratio: the ratio of edges to be removed.
        :param draw_graph: whether to draw the graph.
        :param vis_ratio: the ratio of edges to be visualized.
        """
        self.ratio = ratio
        topk = max(int(ratio * graph.num_edges), 1)

        random_edges = random.sample(range(graph.num_edges), topk)

        scores = np.zeros(graph.num_edges)
        scores[random_edges] = topk - np.array(range(topk))
        scores = self.norm_imp(scores)

        if draw_graph:
            self.visualize(graph, scores, self.name, vis_ratio=vis_ratio)
        self.last_result = (graph, scores)

        return scores

    def evaluate_acc(self, top_ratio_list, graph=None, imp=None, if_cf=False):
        """
        Evaluate the accuracy of the model on the graph
        :param top_ratio_list: list of top ratio of edges to be removed
        :param graph: graph to be evaluated
        :param imp: importance of edges
        :param if_cf: whether to evaluate the fidelity of the model
        :return: accuracy and fidelity of the model
        """
        if graph is None:
            assert self.last_result is not None
            graph, imp = self.last_result
        acc = np.array([[]])
        fidelity = np.array([[]])

        num_edges = graph.edge_index.size(1)
        if self.ds not in {"ba3", "mutag"}:
            graph.edge_attr = torch.ones((num_edges, 1)).to(self.device)

        if self.task == "nc":
            soft_pred, _ = self.model.get_node_pred_subgraph(
                x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, mapping=graph.mapping
            )
        else:
            soft_pred, _ = self.model.get_pred(
                x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, batch=graph.batch
            )

        y_pred = soft_pred.argmax(dim=-1)
        for idx, top_ratio in enumerate(top_ratio_list):
            exp_subgraph = self.pack_explanatory_subgraph(top_ratio, graph=graph, imp=imp, if_cf=if_cf)
            if self.task == "nc":
                soft_pred, _ = self.model.get_node_pred_subgraph(
                    x=exp_subgraph.x,
                    edge_index=exp_subgraph.edge_index,
                    edge_attr=exp_subgraph.edge_attr,
                    mapping=exp_subgraph.mapping,
                )
            else:
                soft_pred, _ = self.model.get_pred(
                    x=exp_subgraph.x,
                    edge_index=exp_subgraph.edge_index,
                    edge_attr=exp_subgraph.edge_attr,
                    batch=exp_subgraph.batch,
                )

            res_acc = (y_pred == soft_pred.argmax(dim=-1)).detach().cpu().float().view(-1, 1).numpy()
            labels = torch.LongTensor([[i] for i in y_pred]).to(y_pred.device)
            if not if_cf:
                res_fid = soft_pred.gather(1, labels).detach().cpu().float().view(-1, 1).numpy()
            else:
                res_fid = (1 - soft_pred.gather(1, labels)).detach().cpu().float().view(-1, 1).numpy()
            acc = np.concatenate([acc, res_acc], axis=1)  # [bsz, len_ratio_list]
            fidelity = np.concatenate([fidelity, res_fid], axis=1)

        return acc, fidelity
