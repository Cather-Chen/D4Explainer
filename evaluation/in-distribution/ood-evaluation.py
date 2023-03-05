import argparse

import torch
from ood_stat import eval_graph_list
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from constants import add_dataset_args, dataset_choices, feature_dict, task_type
from explainers import (
    CF_Explainer,
    CXPlain,
    GNNExplainer,
    GradCam,
    IGExplainer,
    PGExplainer,
    PGMExplainer,
    RandomCaster,
    SAExplainer,
)
from gnns import *
from utils.dataset import get_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Train explainers")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument(
        "--root", type=str, default="results/", help="Result directory."
    )
    parser = add_dataset_args(parser)
    # gflow explainer related parametersn
    parser.add_argument("--gnn_type", type=str, default="gcn")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--train_batchsize", type=int, default=16)
    parser.add_argument("--test_batchsize", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--feature_in", type=int)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--num_test", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.001)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--if_cf", type=bool, default=True)
    return parser.parse_args()


args = parse_args()
args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
mr = [0.2]
results = {}
for dataset in dataset_choices:
    data_wise_resutls = {}
    args.dataset = dataset
    args.feature_in = feature_dict[args.dataset]
    args.task = task_type[args.dataset]
    train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset)
    test_loader = DataLoader(
        test_dataset[: args.num_test], batch_size=1, shuffle=False, drop_last=False
    )
    gnn_path = f"param/gnns/{args.dataset}_{args.gnn_type}.pt"
    for ex in [
        CF_Explainer,
        GNNExplainer,
        PGMExplainer,
        CXPlain,
        PGExplainer,
    ]:
        print(f"-------{dataset} with {ex}----------")
        test_graph = []
        pred_graph = []
        if ex in [PGExplainer]:
            explainer = ex(
                args.device, gnn_path, task=args.task, n_in_channels=args.feature_in
            )
        else:
            explainer = ex(args.device, gnn_path, task=args.task)

        for graph in test_loader:
            graph.to(args.device)
            edge_imp = explainer.explain_graph(graph)
            exp_subgraph = explainer.pack_explanatory_subgraph(
                top_ratio=0.2, graph=graph, imp=edge_imp, if_cf=True
            )
            G_ori = to_networkx(graph, to_undirected=True)
            G_pred = to_networkx(exp_subgraph, to_undirected=True)
            test_graph.append(G_ori)
            pred_graph.append(G_pred)
        MMD = eval_graph_list(test_graph, pred_graph)
        data_wise_resutls[ex] = MMD
    results[dataset] = data_wise_resutls

    gnn_path = f"param/gnns/{args.dataset}_{args.gnn_type}_attr.pt"
    for ex in [SAExplainer, GradCam, IGExplainer, RandomCaster]:
        print(f"-------{dataset} with {ex}----------")
        test_graph = []
        pred_graph = []
        explainer = ex(args.device, gnn_path, task=args.task, ds=args.dataset)
        for graph in tqdm(iter(test_loader), total=len(test_loader)):
            graph.to(args.device)
            edge_imp = explainer.explain_graph(graph)
            exp_subgraph = explainer.pack_explanatory_subgraph(
                top_ratio=0.2, graph=graph, imp=edge_imp, if_cf=True
            )
            G_ori = to_networkx(graph, to_undirected=True)
            G_pred = to_networkx(exp_subgraph, to_undirected=True)
            test_graph.append(G_ori)
            pred_graph.append(G_pred)
        MMD = eval_graph_list(test_graph, pred_graph)
        data_wise_resutls[ex] = MMD
    results[dataset] = data_wise_resutls
