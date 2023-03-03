import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_undirected
from tqdm import tqdm

from explainers import *
from explainers.visual import *
from gnns import *
from utils.dataset import get_datasets

feature_dict = {"BA_Community": 10, "BA_shapes": 10, "Tree_Cycle": 10, "Tree_Grids": 10,
                "mutag": 14, "ba3": 4, "mnist": 1, "tox21": 9, "reddit": 1, "bbbp": 9, "NCI1": 37}
task_type = {"BA_Community": "nc", "BA_shapes": "nc", "Tree_Cycle": "nc", "Tree_Grids":"nc",
                "mutag": "gc", "ba3": "gc", "mnist": "gc", "tox21": "gc", "reddit": "gc", "bbbp": "gc", "NCI1": "gc"}


def parse_args():
    parser = argparse.ArgumentParser(description="explanation visualization")
    parser.add_argument('--cuda', type=int, default=5,
                        help='GPU device.')
    parser.add_argument('--root', type=str, default="results/",
                        help='Result directory.')
    parser.add_argument('--dataset', type=str, default='ba3',
                        choices=["BA_shapes", "Tree_Cycle", "Tree_Grids", 'mutag', 'ba3', "bbbp","NCI1"])
    parser.add_argument('--explainer', type=str, default="CXPlain",
                        choices=['GNNExplainer', 'PGExplainer', 'PGMExplainer',
                                 'CXPlain', 'CF_Explainer'])
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--task', type=str, default="nc")
    parser.add_argument('--if_cf', type=bool, default=True)
    parser.add_argument('--num_start', type=int, default=0)
    parser.add_argument('--num_end', type=int, default=-1)
    parser.add_argument('--normalization', type=str, default="instance")
    parser.add_argument('--num_layers', type=int, default=6)            ####
    parser.add_argument('--layers_per_conv', type=int, default=1)
    parser.add_argument('--sigma_length', type=int, default=5)
    parser.add_argument('--feature_in', type=int)
    parser.add_argument('--n_hidden', type=int, default=128)              ####
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.001)
    parser.add_argument('--prob_low', type=float, default=0.0)
    parser.add_argument('--prob_high', type=float, default=0.5)
    parser.add_argument('--cat_output', type=bool, default=True)
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--noise_mlp', type=bool, default=True)
    parser.add_argument('--simplified', type=bool, default=False)
    return parser.parse_args()


def visualize(args, graph, edge_imp, edge_index_diff, explainer, dataset, vis_ratio, modification_ratio,  label, pred_cf, pred_diff, save=True,):
    vis_setting = vis_dict[dataset] if dataset in vis_dict.keys() else vis_dict["default"]
    if explainer in ['CF_Explainer', 'PGExplainer']:
        topk = min(max(math.ceil((1-vis_ratio) * graph.num_edges), 1), graph.num_edges)
        idx = np.argsort(-edge_imp)[:topk]
    else:
        topk = min(max(math.ceil(vis_ratio * graph.num_edges), 1), graph.num_edges)
        if args.if_cf == False:
            idx = np.argsort(-edge_imp)[:topk]
        else:
            idx = np.argsort(-edge_imp)[topk:]
    G = nx.DiGraph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(list(graph.edge_index.cpu().numpy().T))
    G_diff = nx.DiGraph()
    G_diff.add_nodes_from(range(graph.num_nodes))
    G_diff.add_edges_from(list(edge_index_diff.cpu().numpy().T))
    folder = Path(r'visual/image/%s/%s' % (dataset, explainer))
    if save and not os.path.exists(folder):
        os.makedirs(folder)
    edge_pos_mask = np.zeros(graph.num_edges, dtype=np.bool_)
    edge_pos_mask[idx] = True
    # print(edge_pos_mask, graph.x.size(0))
    vmax = sum(edge_pos_mask)
    node_pos_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    node_neg_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    node_pos_idx = np.unique(graph.edge_index[:, edge_pos_mask].cpu().numpy()).tolist()
    node_neg_idx = list(set([i for i in range(graph.num_nodes)]) - set(node_pos_idx))
    node_pos_mask[node_pos_idx] = True
    node_neg_mask[node_neg_idx] = True

    if dataset == "mutag":
        x = graph.x
        node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7:"S", 8:"P",9:"I",10:"Na", 11:"K",
                     12:"Li", 13:"Ca"}
        node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
        node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
        node_color = ['gold', 'lightgreen', 'azure', 'bisque', 'lightgoldenrodyellow', 'mistyrose', 'thistle',
                      'lemonchiffon', 'lightsteelblue', 'pink', "lavender", "wheat", "plum", "lightblue"]
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        # colors_pos = [colors[i] for i in node_pos_idx]
        # color_neg = [colors[i] for i in node_neg_idx]
        pos = nx.kamada_kawai_layout(G)
        fig = plt.figure(figsize=(24, 6))
        ax = fig.subplots(nrows=1, ncols=3)
        nx.draw_networkx_nodes(G, pos=pos,
                               node_size=vis_setting['node_size'],
                               node_color=colors,
                               alpha=1, cmap='winter',
                               linewidths=vis_setting['linewidths'],
                               edgecolors='grey',
                               ax=ax[0])
        nx.draw_networkx_labels(G, pos=pos, labels=node_labels,ax=ax[0])
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index.cpu().numpy().T),
                               edge_color="black",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[0])

        ###################################
        nx.draw_networkx_nodes(G, pos=pos,
                               node_size=vis_setting['node_size'],
                               node_color=colors,
                               alpha=1, cmap='winter',
                               linewidths=vis_setting['linewidths'],
                               edgecolors='grey',
                               ax=ax[1])
        nx.draw_networkx_labels(G, pos=pos, labels=node_labels,ax=ax[1])

        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index.cpu().numpy().T),
                               edge_color='grey',
                               width=vis_setting['width'],
                               arrows=False,
                               style="dotted",
                               ax=ax[1])
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                               edge_color="black",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[1])

        ################################
        diff_adj = to_dense_adj(edge_index_diff, max_num_nodes=graph.x.size(0))
        ori_adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.x.size(0))
        add_adj = diff_adj - ori_adj
        add_adj = torch.where(add_adj > 0, 1, 0).to(edge_index_diff.device)
        add_edge_index = dense_to_sparse(add_adj)[0]
        nx.draw_networkx_nodes(G, pos=pos,
                               node_size=vis_setting['node_size'],
                               node_color=colors,
                               alpha=1, cmap='winter',
                               linewidths=vis_setting['linewidths'],
                               edgecolors='grey',
                               ax=ax[2])
        nx.draw_networkx_labels(G, pos=pos, labels=node_labels,ax=ax[2])

        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index.cpu().numpy().T),
                               edge_color='grey',
                               width=vis_setting['width'],
                               arrows=False,
                               style="dotted",
                               ax=ax[2])
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(edge_index_diff.cpu().numpy().T),
                               edge_color="black",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[2])

        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(add_edge_index.cpu().numpy().T),
                               edge_color="red",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[2])

        plt.title(f'label: {label.item()}; mr(cf)={vis_ratio}; pred(cf):{pred_cf.item()}; mr(diff)={modification_ratio}; pred(diff)={pred_diff}', fontsize=10)
        if save:
            plt.savefig(folder/Path(f'{graph.name[0]}.pdf'), dpi=500)
        plt.show()

    if dataset == "ba3":
        pos = graph.pos[0]
        fig = plt.figure(figsize=(24, 6))
        ax = fig.subplots(nrows=1, ncols=3)
        nx.draw_networkx_nodes(G, pos=pos,
                               node_size=vis_setting['node_size'],
                               node_color=graph.z[0],
                               alpha=1, cmap='winter',
                               linewidths=vis_setting['linewidths'],
                               edgecolors='grey',
                               ax=ax[0])
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index.cpu().numpy().T),
                               edge_color="black",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[0])

        ###################################
        nx.draw_networkx_nodes(G, pos=pos,
                               node_size=vis_setting['node_size'],
                               node_color=graph.z[0],
                               alpha=1, cmap='winter',
                               linewidths=vis_setting['linewidths'],
                               edgecolors='grey',ax=ax[1]
                               )
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index.cpu().numpy().T),
                               edge_color='grey',
                               width=vis_setting['width'],
                               arrows=False,
                               style="dotted",
                               ax=ax[1])
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                               edge_color="black",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[1])

        ################################
        diff_adj = to_dense_adj(to_undirected(edge_index_diff), max_num_nodes=graph.x.size(0))
        ori_adj = to_dense_adj(to_undirected(graph.edge_index), max_num_nodes=graph.x.size(0))
        add_adj = diff_adj - ori_adj
        add_adj = torch.where(add_adj > 0.0001, 1, 0).to(edge_index_diff.device)
        add_edge_index = dense_to_sparse(add_adj)[0]
        nx.draw_networkx_nodes(G, pos=pos,
                               node_size=vis_setting['node_size'],
                               node_color=graph.z[0],
                               alpha=1, cmap='winter',
                               linewidths=vis_setting['linewidths'],
                               edgecolors='grey',
                               ax=ax[2])

        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index.cpu().numpy().T),
                               edge_color='grey',
                               width=vis_setting['width'],
                               arrows=False,
                               style="dotted",
                               ax=ax[2])
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(edge_index_diff.cpu().numpy().T),
                               edge_color="black",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[2])

        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(add_edge_index.cpu().numpy().T),
                               edge_color="red",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[2])

        plt.title(f'label: {label.item()}; mr(cf)={vis_ratio}; pred(cf):{pred_cf.item()}; mr(diff)={modification_ratio}; pred(diff)={pred_diff.item()}', fontsize=10)
        if save:
            plt.savefig(folder/Path(f'{graph.name[0]}.pdf'), dpi=500)
        plt.show()

    if dataset == "NCI1":
        pos = graph.pos[0]
        fig = plt.figure(figsize=(24, 6))
        ax = fig.subplots(nrows=1, ncols=3)
        nx.draw_networkx_nodes(G, pos=pos,
                               node_size=vis_setting['node_size'],
                               node_color=graph.z[0],
                               alpha=1, cmap='winter',
                               linewidths=vis_setting['linewidths'],
                               edgecolors='grey',
                               ax=ax[0])
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index.cpu().numpy().T),
                               edge_color="black",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[0])

        ###################################
        nx.draw_networkx_nodes(G, pos=pos,
                               node_size=vis_setting['node_size'],
                               node_color=graph.z[0],
                               alpha=1, cmap='winter',
                               linewidths=vis_setting['linewidths'],
                               edgecolors='grey',ax=ax[1]
                               )
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index.cpu().numpy().T),
                               edge_color='grey',
                               width=vis_setting['width'],
                               arrows=False,
                               style="dotted",
                               ax=ax[1])
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                               edge_color="black",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[1])

        ################################
        diff_adj = to_dense_adj(edge_index_diff, max_num_nodes=graph.x.size(0))
        ori_adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.x.size(0))
        add_adj = diff_adj - ori_adj
        add_adj = torch.where(add_adj > 0, 1, 0).to(edge_index_diff.device)
        add_edge_index = dense_to_sparse(add_adj)[0]
        nx.draw_networkx_nodes(G, pos=pos,
                               node_size=vis_setting['node_size'],
                               node_color=graph.z[0],
                               alpha=1, cmap='winter',
                               linewidths=vis_setting['linewidths'],
                               edgecolors='grey',
                               ax=ax[2])

        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index.cpu().numpy().T),
                               edge_color='grey',
                               width=vis_setting['width'],
                               arrows=False,
                               style="dotted",
                               ax=ax[2])
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(edge_index_diff.cpu().numpy().T),
                               edge_color="black",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[2])

        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(add_edge_index.cpu().numpy().T),
                               edge_color="red",
                               width=vis_setting['width'],
                               arrows=False,
                               ax=ax[2])

        plt.title(f'label: {label.item()}; mr(cf)={vis_ratio}; pred(cf):{pred_cf.item()}; mr(diff)={modification_ratio}; pred(diff)={pred_diff.item()}', fontsize=10)
        if save:
            plt.savefig(folder/Path(f'{graph.name[0]}.pdf'), dpi=500)
        plt.show()

ex = "CF_Explainer"
mr = 0.1
args = parse_args()
args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
args.noise_list = None
args.feature_in = feature_dict[args.dataset]
args.task = task_type[args.dataset]
_,_, test_dataset = get_datasets(name=args.dataset)
test_loader = DataLoader(test_dataset[args.num_start:args.num_end], batch_size=1, shuffle=False, drop_last=False)
gnn_path = f'param/gnns/{args.dataset}_{args.gnn_type}.pt'
args.explainer = ex
if args.explainer in ["PGExplainer"]:
    exec(f"Explainer = {args.explainer}(args.device, gnn_path, task=args.task, n_in_channels=args.feature_in)")
else:
    exec(f"Explainer = {args.explainer}(args.device, gnn_path, task=args.task)")
diff_e = DiffExplainer(args.device, gnn_path)
for graph in tqdm(iter(test_loader), total=len(test_loader)):
    y = graph.y if args.task == "gc" else graph.self_y
    graph.to(args.device)
    edge_imp = Explainer.explain_graph(graph)
    exp_subgraph = Explainer.pack_explanatory_subgraph(top_ratio=mr, graph=graph, imp=edge_imp, if_cf=True)
    if args.task == "nc":
        output_prob, _ = Explainer.model.get_node_pred_subgraph(x=exp_subgraph.x, edge_index=exp_subgraph.edge_index,
                                                           mapping=exp_subgraph.mapping)
    else:
        output_prob, _ = Explainer.model.get_pred(x=exp_subgraph.x, edge_index=exp_subgraph.edge_index,
                                             batch=exp_subgraph.batch)
    y_pred_cf = output_prob.argmax(dim=-1)
    edge_index_diff, _, y_exp_diff, modif_r = diff_e.explain_evaluation(args, graph)
    visualize(args, graph, edge_imp, edge_index_diff, explainer=ex, dataset=args.dataset, vis_ratio=mr, modification_ratio=modif_r,
              label=y, pred_cf=y_pred_cf, pred_diff=y_exp_diff,save=True)