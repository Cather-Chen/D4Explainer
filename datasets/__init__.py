import os

from .ba3motif_dataset import BA3Motif
from .load_datasets import *
from .mutag_dataset import Mutagenicity
from .NCI1_dataset import NCI1
from .sup_dataset import Tox, bbbp
from .Syn_dataset import SynGraphDataset
from .web_dataset import WebDataset

# if os.path.exists("../visual_genome"):
#     from .vg_dataset import Visual_Genome