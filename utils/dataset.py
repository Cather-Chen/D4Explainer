import os

import numpy as np
import torch

from datasets import *
from gnns import *

if os.path.exists("visual_genome"):
    from datasets.vg_dataset import Visual_Genome

from torch_geometric.datasets import MNISTSuperpixels


class MNISTTransform(object):
    
    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[col] - pos[row]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if self.norm and cart.numel() > 0:
            max_value = cart.abs().max() if self.max is None else self.max
            cart = cart / (2 * max_value) + 0.5

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart
            
        row, col = data.edge_index
        data.ground_truth_mask = (data.x[row] > 0).view(-1).bool() * (data.x[col] > 0).view(-1).bool()
        return data

    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__,
                                                  self.norm, self.max)


def get_datasets(name, root='data/'):
    if name == "mutag":
        folder = os.path.join(root, 'MUTAG')
        train_dataset = Mutagenicity(folder, mode='training')
        test_dataset = Mutagenicity(folder, mode='testing')
        val_dataset = Mutagenicity(folder, mode='evaluation')
    elif name == "NCI1":
        folder = os.path.join(root, 'NCI1')
        train_dataset = NCI1(folder, mode='training')
        test_dataset = NCI1(folder, mode='testing')
        val_dataset = NCI1(folder, mode='evaluation')
    elif name == "ba3":
        folder = os.path.join(root, 'BA3')
        train_dataset = BA3Motif(folder, mode='training')
        test_dataset = BA3Motif(folder, mode='testing')
        val_dataset = BA3Motif(folder, mode='evaluation')
    elif name == "mnist":
        folder = os.path.join(root, 'MNIST')
        transform = MNISTTransform(cat=False, max_value=9)
        train_dataset = MNISTSuperpixels(folder, True, transform=transform)
        test_dataset = MNISTSuperpixels(folder, False, transform=transform)
        # Reduced dataset
        train_dataset = train_dataset[:6000]
        val_dataset = test_dataset[1000:2000]
        test_dataset = test_dataset[:1000]
    elif name == "vg":
        folder = os.path.join(root, 'VG')
        test_dataset = Visual_Genome(folder, mode='testing')
        val_dataset = Visual_Genome(folder, mode='evaluation')
        train_dataset = Visual_Genome(folder, mode='training')
    elif name == "reddit":
        folder = os.path.join(root, 'reddit')
        test_dataset = Reddit5k(folder, mode='testing')
        val_dataset = Reddit5k(folder, mode='evaluation')
        train_dataset = Reddit5k(folder, mode='training')
    elif name == "BA_shapes":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode='testing',name= "BA_shapes")
        val_dataset = SynGraphDataset(folder, mode='evaluating',name= "BA_shapes")
        train_dataset = SynGraphDataset(folder, mode='training',name= "BA_shapes")
    elif name == "BA_Community":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode='testing',name= "BA_Community")
        val_dataset = SynGraphDataset(folder, mode='evaluating',name= "BA_Community")
        train_dataset = SynGraphDataset(folder, mode='training',name= "BA_Community")
    elif name == "Tree_Cycle":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode='testing',name= "Tree_Cycle")
        val_dataset = SynGraphDataset(folder, mode='evaluating',name= "Tree_Cycle")
        train_dataset = SynGraphDataset(folder, mode='training',name= "Tree_Cycle")
    elif name == "Tree_Grids":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode='testing',name= "Tree_Grids")
        val_dataset = SynGraphDataset(folder, mode='evaluating',name= "Tree_Grids")
        train_dataset = SynGraphDataset(folder, mode='training',name= "Tree_Grids")
    elif name == "bbbp":
        folder = os.path.join(root, 'bbbp')
        dataset = bbbp(folder)
        test_dataset = dataset[:200]
        val_dataset = dataset[200:400]
        train_dataset = dataset[400:]
    elif name == "tox21":
        folder = os.path.join(root, 'tox21')
        dataset = Tox(folder)
        test_dataset = dataset[:500]
        val_dataset = dataset[500:1000]
        train_dataset = dataset[1000:]
    elif name in ['cornell', 'texas', 'wisconsin', 'cora', 'citeseer']:
        folder = os.path.join(root)
        test_dataset = WebDataset(folder, mode='testing',name=name)
        val_dataset = WebDataset(folder, mode='evaluating',name=name)
        train_dataset = WebDataset(folder, mode='training',name=name)
    else:
        raise ValueError
    return train_dataset, val_dataset, test_dataset


