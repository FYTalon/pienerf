import warp as wp
import warp.sparse as wps
import torch
import warp.torch
import numpy as np

wp.init()

a = torch.ones(3, dtype=torch.int32)

n = a.size(0) + a.sum()

mat = wps.bsr_zeros(n, n, block_type=wp.float64)

a = wp.array(shape=n, dtype=wp.float64).zero_()

x = wps.bsr_mv(mat, a)

print(x)
