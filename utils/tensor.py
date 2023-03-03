import numpy as np
import torch


def bool2str(array: torch.Tensor):
    string = ''
    for i in array.int():
        string += np.array(['0', '1'])[i]
    return string

