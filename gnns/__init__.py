
# from .mnist_gnn import MNISTNet
from .ba3motif_gnn import BA3MotifNet, BA3MotifNet_attr
from .mutag_gnn import MutagNet, Mutag_GCN
from .syn_gnn import Syn_GCN, Syn_GCN_attr
from .syn_tree_grids_gcn import Syn_GCN_TG, Syn_GCN_TG_attr
from .tox21_gnn import Tox_GCN
from .bbbp_gnn import BBBP_GCN, BBBP_GCN_attr
from .mnist_gnn import MnistGCN,MNISTNet
from .nci1_gnn import NCI1GCN, NCI1GCN_attr
from .web_gnn import *
from .cora_gnn import CoraGCN
import os
if os.path.exists("../visual_genome"):
    from .vg_gnn import VGNet