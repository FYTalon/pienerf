from plyfile import PlyData, PlyElement
import numpy as np

def main():
    pos = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    pos = np.array(pos)

    vertex = np.array([tuple(v) for v in pos.astype(np.float64)], dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    

if __name__ == "__main__":
    main()