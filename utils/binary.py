import numpy as np
import torch


def tensor2order(array: torch.Tensor):
    string = ""
    for i in array.int():
        string += np.array(["0", "1"])[i]
    return int(string, 2)


def order2tensor(order: int, length: int):
    binary_str = bin(order)[2:]
    # filled with zeros
    binary_str = "0" * (length - len(binary_str)) + binary_str
    return torch.Tensor([int(i) for i in binary_str]).bool()
