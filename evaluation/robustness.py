import argparse
import copy
import math
import os

import numpy as np
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from constants import add_standard_args, feature_dict, task_type
from explainers import CF_Explainer, CXPlain, GNNExplainer, PGExplainer, PGMExplainer
from explainers.base import Explainer as BaseExplainer
from explainers.diff_explainer import Powerful, sparsity
from explainers.diffusion.graph_utils import (
    gen_list_of_data_single,
    generate_mask,
    graph2tensor,
)
from utils.dataset import get_datasets


class DiffExplainer(BaseExplainer):
    def __init__(self, device, gnn_model_path, task, args):
        super(DiffExplainer, self).__init__(device, gnn_model_path, task)
        self.device = device
        self.model = Powerful(args).to(args.device)
        exp_dir = f"{args.root}/{args.dataset}/"
        self.model.load_state_dict(
            torch.load(os.path.join(exp_dir, "best_model.pth"), map_location="cuda:0")[
                "model"
            ]
        )
        self.model.eval()

    def explain_graph(self, model, graph, adj_b, x_b, node_flag_b, sigma_list, args):
        sigma_list = [sigma / 20 for sigma in sigma_list]
        _, _, _, test_noise_adj_b, _ = gen_list_of_data_single(
            x_b, adj_b, node_flag_b, sigma_list, args
        )
        test_noise_adj_b_chunked = test_noise_adj_b.chunk(len(sigma_list), dim=0)
        score = []
        mask = generate_mask(node_flag_b)
        for i, sigma in enumerate(sigma_list):
            # [1, N, N, 1]
            score_batch = self.model(
                A=test_noise_adj_b_chunked[i].to(args.device),
                node_features=x_b,
                mask=mask,
                noiselevel=sigma,
            ).to(args.device)
            score.append(score_batch)
        score_tensor = torch.stack(score, dim=0)  # [len_sigma_list, 1, N, N, 1]
        score_tensor = torch.mean(score_tensor, dim=0)  # [1, N, N, 1]
        y_exp = None  # output_prob_cont.argmax(dim=-1)

        modif_r = sparsity([score_tensor], adj_b, mask)

        score_tensor = score_tensor[0, :, :, 0]
        score_tensor[score_tensor < 0] = 0
        return score_tensor, modif_r, y_exp


class RandomExplainer(BaseExplainer):
    def explain_graph(self, graph):
        return torch.randn(graph.edge_index.shape[1])


Explainer = (
    DiffExplainer
    | PGExplainer
    | RandomExplainer
    | GNNExplainer
    | PGMExplainer
    | CXPlain
    | CF_Explainer
)


def parse_args():
    parser = argparse.ArgumentParser(description="Robustness Experiment")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument(
        "--root", type=str, default="results/distribution", help="Result directory."
    )
    parser.add_dataset_args(parser)
    parser.add_explainer_args(parser)
    parser.add_argument(
        "--mod-ratio", type=float, default=0.2, help="Modification Ratio"
    )
    parser.add_argument("--k", type=int, default=8, help="Top-K")
    parser = add_standard_args(parser)
    return parser.parse_args()


args = parse_args()
args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
for args.dataset in ["NCI1"]:
    args.feature_in = feature_dict[args.dataset]
    args.task = task_type[args.dataset]

    train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    gnn_path = f"param/gnns/{args.dataset}_gcn.pt"
    model = torch.load(gnn_path, map_location="cuda:0").to(args.device)

    def get_graph_pred(graph):
        if args.task == "nc":
            output_prob, _ = model.get_node_pred_subgraph(
                x=graph.x, edge_index=graph.edge_index, mapping=graph.mapping
            )
        else:
            output_prob, _ = model.get_pred(
                x=graph.x, edge_index=graph.edge_index, batch=graph.batch
            )
        orig_pred = output_prob.argmax(dim=-1)
        return orig_pred.item()

    if args.explainer == "DiffExplainer":
        explainer: Explainer = DiffExplainer(
            device=args.device, gnn_model_path=gnn_path, task=args.task, args=args
        )
    elif args.explainer == "PGExplainer":
        explainer: Explainer = PGExplainer(
            device=args.device,
            gnn_model=gnn_path,
            task=args.task,
            n_in_channels=args.feature_in,
        )
    elif args.explainer == "Random":
        explainer: Explainer = RandomExplainer(
            device=args.device, gnn_model_path=gnn_path, task=args.task
        )
    else:
        explainer: Explainer = eval(args.explainer)(
            device=args.device, gnn_model_path=gnn_path, task=args.task
        )

    print(f"--------{args.dataset} with {args.explainer}----------")

    for sigma in range(0, 11):
        sigma /= 100
        acc_logger = []
        orig_modif_r_arr = []
        noisy_modif_r_arr = []

        noisy_graph_same_class = []
        for graph in tqdm(iter(test_loader), total=len(test_loader)):
            adj_b, x_b = graph2tensor(graph.to(args.device), device=args.device)
            x_b = x_b.to(args.device)
            node_flag_b = adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
            _, _, _, noisy_adj_b, _ = gen_list_of_data_single(
                x_b, adj_b, node_flag_b, [sigma], args
            )

            noisy_graph = copy.deepcopy(graph)
            noisy_graph.edge_index = noisy_adj_b[0].nonzero().t()

            orig_pred = get_graph_pred(graph)
            noisy_pred = get_graph_pred(noisy_graph)

            if args.explainer == "DiffExplainer":
                # 2D arrays [N, N] of the importance of full adjacency matrix
                sigma_list = list(
                    np.random.uniform(
                        low=args.prob_low, high=args.prob_high, size=args.sigma_length
                    )
                )
                orig_edge_imp, orig_modif_r, exp_pred = Explainer.explain_graph(
                    model, graph, adj_b, x_b, node_flag_b, sigma_list, args
                )
                noisy_edge_imp, noisy_modif_r, noisy_exp_pred = Explainer.explain_graph(
                    model, noisy_graph, noisy_adj_b, x_b, node_flag_b, sigma_list, args
                )

                orig_modif_r_arr.append(orig_modif_r.item())
                noisy_modif_r_arr.append(noisy_modif_r.item())

                n_nodes = orig_edge_imp.shape[0]

                # K edges with largest counterfactual importance
                try:
                    _, indices = orig_edge_imp.flatten().topk(8)
                except Exception:
                    t = int(n_nodes * n_nodes)
                    _, indices = orig_edge_imp.flatten().topk(t)

                top_k_orig_exp_edges = torch.stack(
                    [indices // n_nodes, indices % n_nodes]
                )

                # all edges with positive counterfactual importance
                noisy_exp_edges = noisy_edge_imp.nonzero().T
            else:
                # 1D arrays of the importance of the edge_index
                orig_edge_imp = explainer.explain_graph(graph)
                noisy_edge_imp = explainer.explain_graph(noisy_graph)

                n_edges = noisy_graph.edge_index.shape[1]

                # K edges with the smallest importance (largest counterfactual importance)
                try:
                    top_k_orig_exp_edges = graph.edge_index[
                        :, np.argsort(orig_edge_imp)[: args.k]
                    ]
                except Exception:
                    top_k_orig_exp_edges = graph.edge_index[
                        :, np.argsort(orig_edge_imp)
                    ]

                # (1-mod%) % edges with the smallest importance (largest counterfactual importance)
                mod_top_k = min(math.ceil(args.mod_ratio * n_edges), n_edges)
                noisy_exp_edges = noisy_graph.edge_index[
                    :, np.argsort(-noisy_edge_imp)[mod_top_k:]
                ]

            noisy_graph_same_class.append(1 if orig_pred == noisy_pred else 0)

            num_intersect = 0
            n_orig_edges = min(args.k, top_k_orig_exp_edges.shape[1])
            for i in range(n_orig_edges):
                if (
                    (top_k_orig_exp_edges[:, i].unsqueeze(1) == noisy_exp_edges)
                    .all(0)
                    .any()
                ):
                    num_intersect += 1
            acc = num_intersect / n_orig_edges
            acc_logger.append(acc)

        print("Sigma", sigma)
        print("Top K Accuracy", round(np.array(acc_logger).mean(), 5))
        print("Noisy Graph Same Class", np.array(noisy_graph_same_class).mean())

        if args.explainer == "DiffExplainer":
            print(
                "Modification Ratio",
                np.array(orig_modif_r_arr).mean(),
                np.array(noisy_modif_r_arr).mean(),
            )
