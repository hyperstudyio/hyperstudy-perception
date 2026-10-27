# Trimmed from gazelle/utils.py at fkryan/gazelle@2d78f9f3 (MIT, ../LICENSE): only the two functions model.py calls.
# The upstream module also holds training and evaluation helpers that import matplotlib and scikit-learn at module level.
import torch

def repeat_tensors(tensor, repeat_counts):
    repeated_tensors = [tensor[i:i+1].repeat(repeat, *[1] * (tensor.ndim - 1)) for i, repeat in enumerate(repeat_counts)]
    return torch.cat(repeated_tensors, dim=0)

def split_tensors(tensor, split_counts):
    indices = torch.cumsum(torch.tensor([0] + split_counts), dim=0)
    return [tensor[indices[i]:indices[i+1]] for i in range(len(split_counts))]
