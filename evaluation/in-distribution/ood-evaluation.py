import argparse
import pickle

import numpy as np
import torch
from ood_stat import *
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from explainers import *
from gnns import *
from utils.dataset import get_datasets

feature_dict = {"BA_Community": 10, "BA_shapes": 10, "Tree_Cycle": 10, "Tree_Grids": 10,"cornell": 1703, "cora":1433,
                "mutag": 14, "ba3": 4, "mnist": 1, "tox21": 9, "reddit": 1, "bbbp": 9, "NCI1": 37}
task_type = {"BA_Community": "nc", "BA_shapes": "nc", "Tree_Cycle": "nc", "Tree_Grids":"nc","cornell": "nc", "cora":"nc",
                "mutag": "gc", "ba3": "gc", "mnist": "gc", "tox21": "gc", "reddit": "gc", "bbbp": "gc", "NCI1": "gc"}

def parse_args():
    parser = argparse.ArgumentParser(description="Train explainers")
    parser.add_argument('--cuda', type=int, default=5,
                        help='GPU device.')
    parser.add_argument('--root', type=str, default="results/",
                        help='Result directory.')
    parser.add_argument('--dataset', type=str, default='Tree_Cycle',
                        choices=["BA_shapes", "Tree_Cycle", "Tree_Grids", "cora", "cornell"
                                 'mutag', 'ba3', 'graphsst2', "tox21", 'mnist','vg', 'reddit', "bbbp", "NCI1"])
    parser.add_argument('--explainer', type=str, default="PGExplainer",
                        choices=['GNNExplainer', 'PGExplainer', 'PGMExplainer',
                                 'CXPlain', 'CF_Explainer'])
    # gflow explainer related parametersn
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--task', type=str, default="nc")
    parser.add_argument('--verbose', type=int, default=10)
    parser.add_argument('--train_batchsize', type=int, default=16)            ####
    parser.add_argument('--test_batchsize', type=int, default=16)            ####
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--feature_in', type=int)
    parser.add_argument('--n_hidden', type=int, default=128)              ####
    parser.add_argument('--num_test', type=int, default=50)


    parser.add_argument('--dropout', type=float, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=1e-3)    ####
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--if_cf', type=bool, default=True)

    return parser.parse_args()


args = parse_args()
args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
mr = [0.2]
results = {}
for dataset in ["BA_shapes", "Tree_Cycle", "Tree_Grids", "cornell", 'mutag', 'ba3', "bbbp", "NCI1"]:
    data_wise_resutls = {}
    args.dataset = dataset
    args.feature_in = feature_dict[args.dataset]
    args.task = task_type[args.dataset]
    train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset)
    test_loader = DataLoader(test_dataset[:args.num_test], batch_size=1, shuffle=False, drop_last=False)
    gnn_path = f'param/gnns/{args.dataset}_{args.gnn_type}.pt'
    for ex in [ 'CF_Explainer', 'GNNExplainer', 'PGMExplainer', 'CXPlain', 'PGExplainer']:
        print(f"-------{dataset} with {ex}----------")
        test_graph = []
        pred_graph = []
        args.explainer = ex
        if args.explainer in ["PGExplainer"]:
            exec(f"Explainer = {args.explainer}(args.device, gnn_path, task=args.task, n_in_channels=args.feature_in)")
        else:
            exec(f"Explainer = {args.explainer}(args.device, gnn_path, task=args.task)")
        # for graph in tqdm(iter(test_loader), total=len(test_loader)):
        for graph in test_loader:
            graph.to(args.device)
            edge_imp = Explainer.explain_graph(graph)
            exp_subgraph = Explainer.pack_explanatory_subgraph(top_ratio=0.2, graph=graph, imp=edge_imp, if_cf=True)
            G_ori = to_networkx(graph, to_undirected=True)
            G_pred = to_networkx(exp_subgraph, to_undirected=True)
            test_graph.append(G_ori)
            pred_graph.append(G_pred)
        MMD = eval_graph_list(test_graph,pred_graph)
        data_wise_resutls[ex] = MMD
    results[dataset] = data_wise_resutls


    gnn_path = f'param/gnns/{args.dataset}_{args.gnn_type}_attr.pt'
    for ex in ['SAExplainer', 'GradCam', 'IGExplainer', "RandomCaster"]:
        print(f"-------{dataset} with {ex}----------")
        test_graph = []
        pred_graph = []
        args.explainer = ex
        exec(f"Explainer = {args.explainer}(args.device, gnn_path, task=args.task, ds=args.dataset)")
        for graph in tqdm(iter(test_loader), total=len(test_loader)):
            graph.to(args.device)
            edge_imp = Explainer.explain_graph(graph)
            exp_subgraph = Explainer.pack_explanatory_subgraph(top_ratio=0.2, graph=graph, imp=edge_imp, if_cf=True)
            G_ori = to_networkx(graph, to_undirected=True)
            G_pred = to_networkx(exp_subgraph, to_undirected=True)
            test_graph.append(G_ori)
            pred_graph.append(G_pred)
        MMD = eval_graph_list(test_graph,pred_graph)
        data_wise_resutls[ex] = MMD
    results[dataset] = data_wise_resutls
