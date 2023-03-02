import torch
import numpy as np


def bool2str(array: torch.Tensor):
    string = ''
    for i in array.int():
        string += np.array(['0', '1'])[i]
    return string

