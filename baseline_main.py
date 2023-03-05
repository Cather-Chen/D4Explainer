import argparse

import numpy as np
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from constants import add_dataset_args, add_explainer_args, feature_dict, task_type
from explainers import *
from gnns import *
from utils.dataset import get_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Train explainers")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument(
        "--root", type=str, default="results/", help="Result directory."
    )
    parser = add_dataset_args(parser)
    parser = add_explainer_args(parser)

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
mr = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
results = {}
for dataset in [
    "BA_shapes",
    "Tree_Cycle",
    "Tree_Grids",
    "cornell",
    "mutag",
    "ba3",
    "bbbp",
    "NCI1",
]:
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
        "PGExplainer",
        "CF_Explainer",
        "GNNExplainer",
        "PGMExplainer",
        "CXPlain",
    ]:
        data_wise_resutls[ex] = []
        args.explainer = ex
        if args.explainer in ["PGExplainer"]:
            explainer = eval(args.explainer)(
                args.device, gnn_path, task=args.task, n_in_channels=args.feature_in
            )
        else:
            explainer = eval(args.explainer)(args.device, gnn_path, task=args.task)
        acc_logger, fid_logger = [], []
        # for graph in tqdm(iter(test_loader), total=len(test_loader)):
        for graph in tqdm(iter(test_loader), total=len(test_loader)):
            graph.to(args.device)
            edge_imp = explainer.explain_graph(graph)
            acc, fidelity = explainer.evaluate_acc(
                mr, graph=graph, imp=edge_imp, if_cf=True
            )
            acc_logger.append(acc)
            fid_logger.append(fidelity)
        S_A = np.mean(acc_logger, axis=0, keepdims=False)[0]
        S_F = np.mean(fid_logger, axis=0, keepdims=False)[0]
        auc = np.array(acc_logger).mean(axis=0).mean()
        data_wise_resutls[ex].append(S_A)
        data_wise_resutls[ex].append(S_F)
        data_wise_resutls[ex].append(auc)
        print(f"--------{dataset} with {ex}----------")
        print("Sparsity V.S. Acc:", np.mean(acc_logger, axis=0, keepdims=False)[0])
        print("AUC: %.3f" % np.array(acc_logger).mean(axis=0).mean())
        print("Sparsity V.S. Fidelity:", np.mean(fid_logger, axis=0, keepdims=False)[0])
        print("AUC-fid: %.3f" % np.array(fid_logger).mean(axis=0).mean())
    results[dataset] = data_wise_resutls

    gnn_path = f"param/gnns/{args.dataset}_{args.gnn_type}_attr.pt"
    for ex in ["SAExplainer", "GradCam", "IGExplainer", "RandomCaster"]:
        data_wise_resutls[ex] = []
        args.explainer = ex
        explainer = eval(args.explainer)(
            args.device, gnn_path, task=args.task, ds=args.dataset
        )

        acc_logger, fid_logger = [], []
        # for graph in tqdm(iter(test_loader), total=len(test_loader)):
        for graph in tqdm(iter(test_loader), total=len(test_loader)):
            graph.to(args.device)
            edge_imp = explainer.explain_graph(graph)
            acc, fidelity = explainer.evaluate_acc(
                mr, graph=graph, imp=edge_imp, if_cf=True
            )
            acc_logger.append(acc)
            fid_logger.append(fidelity)
        S_A = np.mean(acc_logger, axis=0, keepdims=False)[0]
        S_F = np.mean(fid_logger, axis=0, keepdims=False)[0]
        auc = np.array(acc_logger).mean(axis=0).mean()
        data_wise_resutls[ex].append(S_A)
        data_wise_resutls[ex].append(S_F)
        data_wise_resutls[ex].append(auc)
        print(f"--------{dataset} with {ex}----------")
        print("Sparsity V.S. Acc:", np.mean(acc_logger, axis=0, keepdims=False)[0])
        print("AUC: %.3f" % np.array(acc_logger).mean(axis=0).mean())
        print("Sparsity V.S. Fidelity:", np.mean(fid_logger, axis=0, keepdims=False)[0])
        print("AUC-fid: %.3f" % np.array(fid_logger).mean(axis=0).mean())
    results[dataset] = data_wise_resutls
