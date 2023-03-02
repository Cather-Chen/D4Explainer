from .mutag_dataset import Mutagenicity
from .ba3motif_dataset import BA3Motif
from .load_datasets import *
import os
from .vg_dataset import Visual_Genome
from .reddit5k_dataset import Reddit5k
from .Syn_dataset import SynGraphDataset
from .sup_dataset import Tox, bbbp
from .NCI1_dataset import NCI1
from .web_dataset import WebDataset
# if os.path.exists("../visual_genome"):
#     from .vg_dataset import Visual_Genome