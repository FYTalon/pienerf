import os
pienerf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import warp as wp
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
wp.init()

from plyfile import PlyData, PlyElement
import numpy as np
def write_ply(filename, points):
    vertex = np.array([tuple(v) for v in points], dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])  # float64
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(filename)

from simulator.solver import Simulator
import torch

def main():

    if not os.path.exists(pienerf_dir + "/outputs"):
        os.mkdir(pienerf_dir + "/outputs")

    sim = Simulator(
        dt=1e-2,
        iters=10,
        bbox=torch.tensor([2, 2, 2], dtype=torchfloat),
        dx=0.05,
        stiff=1e3,
        base=torch.tensor([-1, -1, -1], dtype=torchfloat)
    )

    sim.InitializeFromPly(pienerf_dir + "/assets/chair.ply")
    sim.OutputToPly(pienerf_dir + "/outputs/0.ply")

    if not os.path.exists(pienerf_dir + "/debug"):
        os.mkdir(pienerf_dir + "/debug")
    write_ply(pienerf_dir + "/debug/ip_0.ply", sim.IP_pos.cpu().numpy())

    cost = time.time()

    with torch.no_grad():
        for i in range(1, 1001):
            print(i)
            sim.stepforward()
            sim.OutputToPly(pienerf_dir + f"/outputs/{i}.ply")
            pos, _, _ = sim.get_IP_info()
            write_ply(pienerf_dir + f"/debug/ip_{i}.ply", pos.cpu().numpy())
    print(time.time() - cost)

if __name__ == "__main__":
    main()

