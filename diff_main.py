import argparse

import torch

from explainers import *
from gnns import *
from utils.dataset import get_datasets

feature_dict = {"BA_Community": 10, "BA_shapes": 10, "Tree_Cycle": 10, "Tree_Grids": 10,"cornell": 1703, "cora":1433,
                "mutag": 14, "ba3": 4, "mnist": 1, "tox21": 9, "reddit": 1, "bbbp": 9, "NCI1": 37}
task_type = {"BA_Community": "nc", "BA_shapes": "nc", "Tree_Cycle": "nc", "Tree_Grids":"nc","cornell": "nc", "cora":"nc",
                "mutag": "gc", "ba3": "gc", "mnist": "gc", "tox21": "gc", "reddit": "gc", "bbbp": "gc", "NCI1": "gc"}

def parse_args():
    parser = argparse.ArgumentParser(description="Train explainers")
    parser.add_argument('--cuda', type=int, default=3,
                        help='GPU device.')
    parser.add_argument('--root', type=str, default="results/distribution/",
                        help='Result directory.')
    parser.add_argument('--dataset', type=str, default='NCI1',
                        choices=["BA_shapes", "Tree_Cycle", "Tree_Grids","cornell"
                                 'mutag', 'ba3',  "bbbp", "NCI1"])

    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--task', type=str, default="nc")
    parser.add_argument('--normalization', type=str, default="instance")
    parser.add_argument('--verbose', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--layers_per_conv', type=int, default=1)
    parser.add_argument('--train_batchsize', type=int, default=32)
    parser.add_argument('--test_batchsize', type=int, default=32)
    parser.add_argument('--sigma_length', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=800)
    parser.add_argument('--feature_in', type=int)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--data_size', type=int, default=-1)

    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--alpha_cf', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--prob_low', type=float, default=0.0)
    parser.add_argument('--prob_high', type=float, default=0.4)
    parser.add_argument('--sparsity_level', type=float, default=2.5)

    parser.add_argument('--cat_output', type=bool, default=True)
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--noise_mlp', type=bool, default=True)
    parser.add_argument('--simplified', type=bool, default=False)
    return parser.parse_args()


args = parse_args()
args.noise_list = None

args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
args.feature_in = feature_dict[args.dataset]
args.task = task_type[args.dataset]
train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset)

train_dataset = train_dataset[:args.data_size]
gnn_path = f'param/gnns/{args.dataset}_{args.gnn_type}.pt'
explainer = DiffExplainer(args.device, gnn_path)
explainer.explain_graph_task(args, train_dataset, test_dataset)