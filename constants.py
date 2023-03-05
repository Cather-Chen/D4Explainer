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
