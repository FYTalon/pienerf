import os
import warp as wp
import time

wp.init()

from solver import Simulator
import torch

def main():

    if not os.path.exists("./outputs"):
        os.mkdir("./outputs")

    sim = Simulator(
        dt=1e-2,
        iters=10,
        res=torch.tensor([90, 30, 30]),
        dx=0.05,
        stiff=1e7,
        base=torch.tensor([-0.5, -0.5, -0.5])
    )

    sim.InitializeFromPly("./assets/cube.ply")

    sim.OutputToPly("./outputs/0.ply")

    cost = time.time()

    with torch.no_grad():
        for i in range(1, 1001):

            sim.stepforward()
            sim.OutputToPly(f"./outputs/{i}.ply")
    print(time.time() - cost)

if __name__ == "__main__":
    main()

