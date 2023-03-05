feature_dict = {
    "BA_shapes": 10,
    "Tree_Cycle": 10,
    "Tree_Grids": 10,
    "cornell": 1703,
    "mutag": 14,
    "ba3": 4,
    "bbbp": 9,
    "NCI1": 37,
}

task_type = {
    "BA_shapes": "nc",
    "Tree_Cycle": "nc",
    "Tree_Grids": "nc",
    "cornell": "nc",
    "mutag": "gc",
    "ba3": "gc",
    "bbbp": "gc",
    "NCI1": "gc",
}

dataset_choices = list(task_type.keys())


explainer_choices = [
    "DiffExplainer",
    "GNNExplainer",
    "PGExplainer",
    "PGMExplainer",
    "CXPlain",
    "CF_Explainer",
    "SAExplainer",
    "GradCam",
    "IGExplainer",
    "RandomCaster",
]


def add_dataset_args(parser):
    parser.add_argument("--dataset", type=str, default="NCI1", choices=dataset_choices)
    return parser


def add_explainer_args(parser):
    parser.add_argument(
        "--explainer", type=str, default="DiffExplainer", choices=explainer_choices
    )
    return parser


def add_standard_args(parser):
    # gflow explainer related parameters
    parser.add_argument("--gnn_type", type=str, default="gcn")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--normalization", type=str, default="instance")
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--layers_per_conv", type=int, default=1)
    parser.add_argument("--train_batchsize", type=int, default=32)
    parser.add_argument("--test_batchsize", type=int, default=32)
    parser.add_argument("--sigma_length", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--feature_in", type=int)
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--data_size", type=int, default=-1)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha_cf", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.001)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--prob_low", type=float, default=0.0)
    parser.add_argument("--prob_high", type=float, default=0.2)
    parser.add_argument("--sparsity_level", type=float, default=2.5)

    parser.add_argument("--cat_output", type=bool, default=True)
    parser.add_argument("--residual", type=bool, default=False)
    parser.add_argument("--noise_mlp", type=bool, default=True)
    parser.add_argument("--simplified", type=bool, default=False)

    return parser
