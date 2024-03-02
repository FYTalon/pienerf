import os
import warp as wp

wp.init()

from solver import Simulator
import torch

def main():

    if not os.path.exists("./outputs"):
        os.mkdir("./outputs")

    sim = Simulator(
        dt=1e-3,
        iters=2,
        res=torch.tensor([16, 3, 3]),
        dx=1,
        subspace=10
    )

    sim.InitializeFromPly("./assets/cube.ply")

    sim.OutputToPly("./outputs/0.ply")

    with torch.no_grad():
        for i in range(1, 201):
            sim.stepforward()
            sim.OutputToPly(f"./outputs/{i}.ply")

if __name__ == "__main__":
    main()

