

import torch
from torch_geometric.nn import NNConv
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softmax
from explainers.base import Explainer

import warnings
warnings.filterwarnings("ignore")


class CXPlain(Explainer):
    
    def __init__(self, device, gnn_model_path, task):
        super(CXPlain, self).__init__(device, gnn_model_path, task)
        
    def explain_graph(self, graph,
                      model=None,
                      epoch=100,
                      lr=0.01,
                      draw_graph=0,
                      vis_ratio=0.2):

        y = graph.y if self.task == "gc" else graph.self_y
        if model == None:
            model = self.model
        if self.task == "nc":
            soft_pred, _ = self.model.get_node_pred_subgraph(x=graph.x, edge_index=graph.edge_index,
                                                             mapping=graph.mapping)
        else:
            soft_pred, _ = self.model.get_pred(x=graph.x, edge_index=graph.edge_index,
                                               batch=graph.batch)
        orig_pred = soft_pred[0, y]

        granger_imp = []
        for e_id in range(graph.num_edges):
            edge_mask = torch.ones(graph.num_edges, dtype=torch.bool)
            edge_mask[e_id] = False
            tmp_g = graph.clone()
            tmp_g.edge_index = graph.edge_index[:, edge_mask]
            try:
                tmp_g.edge_attr = graph.edge_attr[edge_mask]
            except:
                pass
            if self.task == "nc":
                soft_pred, _ = self.model.get_node_pred_subgraph(x=tmp_g.x, edge_index=tmp_g.edge_index,
                                                                 mapping=tmp_g.mapping)
            else:
                soft_pred, _ = self.model.get_pred(x=tmp_g.x, edge_index=tmp_g.edge_index,
                                                   batch=tmp_g.batch)

            masked_pred = soft_pred[0, y]
            granger_imp.append(float(orig_pred - masked_pred))

        granger_imp = torch.FloatTensor(granger_imp)
        scores = self.norm_imp(granger_imp).to(self.device)
        
        explainer = CX_Model(graph, h_dim=32).to(self.device) 
        optimizer = torch.optim.Adam(explainer.parameters(), lr=lr)

        for i in range(1, epoch + 1):
            optimizer.zero_grad()
            out = explainer(graph)
            out = F.softmax(out)
            loss = F.kl_div(scores, out)
            loss.backward()
            optimizer.step()
            
        out = out.detach().cpu().numpy()

        if draw_graph:
            self.visualize(graph, out, self.name, vis_ratio=vis_ratio)
        self.last_result = (graph, out)

        return out


class CX_Model(torch.nn.Module):

    def __init__(self, graph, h_dim):
        super(CX_Model, self).__init__()
        # node encoder
        self.lin0 = Lin(graph.x.size(-1), h_dim)
        self.relu0 = ReLU()
        self.edge_nn = Seq(Lin(in_features=1, out_features=h_dim),
                           ReLU(),
                           Lin(in_features=h_dim, out_features=h_dim * h_dim))
        self.conv = NNConv(in_channels = h_dim,
                           out_channels = h_dim,
                           nn = self.edge_nn)
        self.lin1 = torch.nn.Linear(h_dim, 8)
        self.relu1 = ReLU()
        self.lin2 = torch.nn.Linear(8, 1)

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = torch.ones((edge_index.size(1), 1)).to(edge_index.device)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.relu0(self.lin0(x))
        x = self.conv(x, edge_index,edge_attr=edge_attr)
        edge_emb = x[edge_index[0,:]] * x[edge_index[1,:]]
        edge_emb = self.relu1(self.lin1(edge_emb))
        edge_score = self.lin2(edge_emb)
        edge_score = edge_score.view(-1)

        return edge_score
