import argparse
import os.path as osp
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch_geometric.data import DataLoader

from gnns import *
from utils.dataset import get_datasets

EPS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mutag Model")

    parser.add_argument(
        "--data_name", nargs="?", default="Tree_Cycle", help="Input data path."
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        default=osp.join(osp.dirname(__file__), "param", "gnns"),
        help="path for saving trained model.",
    )
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--epoch", type=int, default=5000, help="Number of epoch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=128, help="hiden size.")
    parser.add_argument(
        "--verbose", type=int, default=10, help="Interval of evaluation."
    )
    parser.add_argument(
        "--num_unit", type=int, default=3, help="number of Convolution layers(units)"
    )
    parser.add_argument(
        "--random_label",
        type=bool,
        default=False,
        help="train a model under label randomization for sanity check",
    )

    return parser.parse_args()


if __name__ == "__main__":
    model = torch.load("param/gnns/Tree_Cycle_gcn.pt")
    train, val, test = get_datasets(name="Tree_Cycle")
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    X = []
    Y = []

    for graph in test_loader:
        # graph.to(device)
        node_x = model.get_node_reps(graph.x, graph.edge_index).squeeze(0)
        X.append(node_x)
        Y.append(graph.y)

    X = torch.cat(X, dim=0)  # [num_nodes, n_hid]
    Y = torch.cat(Y, dim=0)  # [num_nodes]
    print(X.size(), Y.size())
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    Z = tsne.fit_transform(X)

    df = pd.DataFrame()
    df["y"] = Y
    df["comp-1"] = Z[:, 0]
    df["comp-2"] = Z[:, 1]

    sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue=df.y.tolist(),
        palette=sns.color_palette("hls", 2),
        data=df,
    )
    plt.title("T-SNE projection of Tree-Cycle", fontsize=20)
    plt.xlabel("Dimension 1", fontsize=10)
    plt.ylabel("Dimension 2", fontsize=10)
    plt.savefig("tsne.pdf", dpi=500)
    plt.show()
