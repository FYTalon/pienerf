import torch
from kornia.utils.grid import create_meshgrid3d

a = torch.tensor([0, 0, 1, 1, 2, 2])
mass = torch.tensor([1, 1, 1, 1, 1, 1])
x = torch.tensor([0, 0, 0])
x[a] += mass
print(x)
