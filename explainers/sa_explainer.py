import numpy as np
import torch
from torch.autograd import Variable

from explainers.base import Explainer


class SAExplainer(Explainer):
    def __init__(self, device, gnn_model_path, task, ds):
        super(SAExplainer, self).__init__(device, gnn_model_path, task)
        self.ds = ds

    def explain_graph(self, graph, model=None, draw_graph=0, vis_ratio=0.2):
        if model == None:
            model = self.model
        y = graph.y if self.task == "gc" else graph.self_y
        tmp_graph = graph.clone()
        num_edges = graph.edge_index.size(1)
        if self.ds in {"ba3", "mutag"}:
            tmp_graph.edge_attr = Variable(tmp_graph.edge_attr, requires_grad=True)
        else:
            tmp_graph.edge_attr = torch.ones((num_edges, 1)).to(self.device)
            tmp_graph.edge_attr = Variable(tmp_graph.edge_attr, requires_grad=True)
        tmp_graph.x = Variable(tmp_graph.x, requires_grad=True)

        if self.task == "nc":
            soft_pred, _ = model.get_node_pred_subgraph(
                x=tmp_graph.x,
                edge_index=tmp_graph.edge_index,
                edge_attr=tmp_graph.edge_attr,
                mapping=tmp_graph.mapping,
            )
        else:
            soft_pred, _ = model.get_pred(
                x=tmp_graph.x,
                edge_index=tmp_graph.edge_index,
                edge_attr=tmp_graph.edge_attr,
                batch=tmp_graph.batch,
            )

        soft_pred[0, y].backward()

        edge_grads = pow(tmp_graph.edge_attr.grad, 2).sum(dim=1).cpu().numpy()
        edge_imp = self.norm_imp(edge_grads)

        if draw_graph:
            self.visualize(graph, edge_imp, self.name, vis_ratio=vis_ratio)
        self.last_result = (graph, edge_imp)

        return edge_imp

    def evaluate_acc(self, top_ratio_list, graph=None, imp=None, if_cf=False):
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
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                mapping=graph.mapping,
            )
        else:
            soft_pred, _ = self.model.get_pred(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                batch=graph.batch,
            )

        y_pred = soft_pred.argmax(dim=-1)
        for idx, top_ratio in enumerate(top_ratio_list):
            exp_subgraph = self.pack_explanatory_subgraph(
                top_ratio, graph=graph, imp=imp, if_cf=if_cf
            )
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
            # soft_pred: [bsz, num_class]
            res_acc = (
                (y_pred == soft_pred.argmax(dim=-1))
                .detach()
                .cpu()
                .float()
                .view(-1, 1)
                .numpy()
            )
            labels = torch.LongTensor([[i] for i in y_pred]).to(y_pred.device)
            if if_cf == False:
                res_fid = (
                    soft_pred.gather(1, labels)
                    .detach()
                    .cpu()
                    .float()
                    .view(-1, 1)
                    .numpy()
                )
            else:
                res_fid = (
                    (1 - soft_pred.gather(1, labels))
                    .detach()
                    .cpu()
                    .float()
                    .view(-1, 1)
                    .numpy()
                )
            acc = np.concatenate([acc, res_acc], axis=1)  # [bsz, len_ratio_list]
            fidelity = np.concatenate([fidelity, res_fid], axis=1)
        return acc, fidelity
