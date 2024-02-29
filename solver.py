import torch
import warp as wp
import warp.torch
import numpy as np
from plyfile import PlyData, PlyElement
from kornia.utils.grid import create_meshgrid3d
import cuda_utils
from cuda_utils import torchfloat, npfloat, wpfloat, vec3, vec10, vec8i, mat10, mat3

class Simulator:
    def __init__(
            self,
            dt=1e-2,
            iters=20,
            res=torch.tensor([32, 32, 32], dtype=torch.int32),
            dx=1,
            subspace=100,
            gravity=torch.tensor([0.0, -9.8, 0.0]),
            base=torch.tensor([-0.5, -0.5, -0.5])
    ):
        self.dt = dt
        self.iters = iters
        self.res = res
        self.dx = dx
        self.subspace = subspace
        self.base = base

        self.gravity = gravity

        # point
        self.pos = None
        self.pos_rest = None
        self.grid_idx = None
        self.vel = None

        self.mass = None
        self.mu = None
        self.lam = None
        self.is_pin = None
        self.pts_IP = None
        self.pts_kernel = None
        self.pts_G = None

        self.pts_Nx = None
        self.pts_dNx = None
        self.pts_ddNx = None

        # IP
        # grid_mask
        self.IP_mask = None
        # grid 2 idx
        self.IP_idx = None
        # idx_mu
        self.IP_mu = None
        # idx_lam
        self.IP_lam = None
        # idx_rho
        self.IP_rho = None
        # idx idx
        self.IP_kernel = None
        # idx 2 grid
        self.IP_grid = None
        # idx_pos
        self.IP_pos = None
        # idx_G
        self.IP_G = None

        self.IP_Nx = None
        self.IP_dNx = None
        self.IP_ddNx = None

        # kernels
        # grid_mask
        self.kernel_mask = None
        # grid 2 idx
        self.kernel_idx = None
        # idx 2 grid
        self.kernel_grid = None
        # idx_pos
        self.kernel_pos = None


    def InitializeFromPly(self, path):
        plydata = PlyData.read(path)

        self.pos = torch.from_numpy(
            np.stack((np.asarray(plydata.elements[0]["x"]),
                      np.asarray(plydata.elements[0]["y"]),
                      np.asarray(plydata.elements[0]["z"])), axis=1).astype(npfloat)
        )

        self.pos_rest = torch.from_numpy(
            np.stack((np.asarray(plydata.elements[0]["rest_x"]),
                      np.asarray(plydata.elements[0]["rest_y"]),
                      np.asarray(plydata.elements[0]["rest_z"])), axis=1).astype(npfloat)
        )

        self.vel = torch.from_numpy(
            np.stack((np.asarray(plydata.elements[0]["vel_x"]),
                      np.asarray(plydata.elements[0]["vel_y"]),
                      np.asarray(plydata.elements[0]["vel_z"])), axis=1).astype(npfloat)
        )

        self.grid_idx = torch.from_numpy(
            np.stack((np.asarray(plydata.elements[0]["idx_x"]),
                      np.asarray(plydata.elements[0]["idx_y"]),
                      np.asarray(plydata.elements[0]["idx_z"])), axis=1).astype(np.int)
        )

        self.mass = torch.from_numpy(
            np.asarray(plydata.elements[0]["mass"]).astype(npfloat)
        )
        self.mu = torch.from_numpy(
            np.asarray(plydata.elements[0]["mu"]).astype(npfloat)
        )
        self.is_pin = torch.from_numpy(
            np.asarray(plydata.elements[0]["pin"]).astype(np.int)
        )

        self.initialize()

    def initialize(self):
        self.IP_mask = torch.zeros(
            (self.res[0], self.res[1], self.res[2]),
            dtype=torch.bool
        )
        self.IP_mask[self.grid_idx[:, 0], self.grid_idx[:, 1], self.grid_idx[:, 2]] = True

        self.IP_idx = torch.zeros(
            (self.res[0], self.res[1], self.res[2]),
            dtype=torch.int32
        )
        self.IP_idx[self.IP_mask] = torch.arange(0, self.IP_mask.sum(), 1, dtype=torch.int32)

        self.pts_IP = torch.zeros(
            self.pos.size(0),
            dtype=torch.int32
        )
        self.pts_IP = self.IP_idx[self.grid_idx[:, 0], self.grid_idx[:, 1], self.grid_idx[:, 2]]

        IP_pos = create_meshgrid3d(
            self.res[0],
            self.res[1],
            self.res[2],
            False,
            dtype=torch.int32
        ).reshape(self.res[0], self.res[1], self.res[2], 3)

        self.IP_grid = torch.zeros(
            (self.IP_mask.sum(), 3),
            dtype=torch.int32
        )
        self.IP_grid = IP_pos[self.IP_mask, :]

        self.IP_pos = self.IP_grid * self.dx + self.base + 0.5


        self.kernel_mask = torch.zeros(
            (self.res[0] + 1, self.res[1] + 1, self.res[2] + 1),
            dtype=torch.bool
        )
        self.IP_kernel = torch.zeros(
            (self.IP_mask.sum(), 8),
            dtype=torch.int32
        )
        for S in range(8):
            x = S >> 2 & 1
            y = S >> 1 & 1
            z = S & 1

            self.kernel_mask[self.IP_grid[:, 0] + x, self.IP_grid[:, 1] + y, self.IP_grid[:, 2] + z] |= True

        self.kernel_idx = torch.zeros(
            (self.res[0] + 1, self.res[1] + 1, self.res[2] + 1),
            dtype=torch.int32
        )
        self.kernel_idx[self.kernel_mask] = torch.arange(0, self.kernel_mask.sum(), 1, dtype=torch.int32)

        self.pts_kernel = torch.zeros(
            (self.pos.size(0), 8),
            dtype=torch.int32
        )

        for S in range(8):
            x = S >> 2 & 1
            y = S >> 1 & 1
            z = S & 1
            self.IP_kernel[:, S] = \
                self.kernel_idx[self.IP_grid[:, 0] + x, self.IP_grid[:, 1] + y, self.IP_grid[:, 2] + z]
            self.pts_kernel[:, S] = \
                self.kernel_idx[self.grid_idx[:, 0] + x, self.grid_idx[:, 1] + y, self.grid_idx[:, 2] + z]

        kernel_pos = create_meshgrid3d(
            self.res[0] + 1,
            self.res[1] + 1,
            self.res[2] + 1,
            False,
            dtype=torch.int32
        ).reshape(self.res[0] + 1, self.res[1] + 1, self.res[2] + 1, 3)

        self.kernel_grid = torch.zeros(
            (self.kernel_mask.sum(), 3),
            dtype=torch.int32
        )
        self.kernel_grid = kernel_pos[self.kernel_mask, :]

        self.kernel_pos = self.kernel_grid * self.dx + self.base

        self.pts_Nx, self.pts_dNx, self.pts_ddNx = self.init_GMLS(self.pos, self.pts_kernel)

        self.IP_Nx, self.IP_dNx, self.IP_ddNx = self.init_GMLS(self.IP_pos, self.IP_kernel)

        self.IP_mu, self.IP_lam, self.IP_rho = self.collect_IP()


    def init_GMLS(self, pos, topo):
        n_pts = pos.size(0)
        pos = wp.from_torch(pos)
        topo = wp.from_torch(topo)
        pts_G = wp.array(shape=n_pts, dtype=mat10)
        pts_dG = wp.array(shape=(n_pts, 3), dtype=mat10)
        pts_ddG = wp.array(shape=(n_pts, 3, 3), dtype=mat10)

        wp.launch(
            kernel=cuda_utils.calc_G,
            dim=(n_pts, ),
            inputs=[
                self.dx,
                pos,
                topo,
                wp.from_torch(self.kernel_pos),
                pts_G, pts_dG, pts_ddG
            ]
        )

        pts_G_inv = wp.from_torch(torch.linalg.inv(wp.to_torch(pts_G)))

        pts_Gp = wp.array(shape=n_pts, dtype=vec10)
        pts_dGp = wp.array(shape=(n_pts, 3), dtype=vec10)
        pts_ddGp = wp.array(shape=(n_pts, 3, 3), dtype=vec10)

        wp.launch(
            kernel=cuda_utils.calc_Gp,
            dim=(n_pts, ),
            inputs=[
                pos,
                pts_G,
                pts_G_inv,
                pts_dG,
                pts_ddG,
                pts_Gp, pts_dGp, pts_ddGp
            ]
        )

        pts_Nx = wp.zeros(shape=(n_pts, 8), dtype=vec10)
        pts_dNx = wp.zeros(shape=(n_pts, 8, 3), dtype=vec10)
        pts_ddNx = wp.zeros(shape=(n_pts, 8, 3, 3), dtype=vec10)

        wp.launch(
            kernel=cuda_utils.calc_weight,
            dim=(n_pts, ),
            inputs=[
                self.dx,
                pos,
                topo,
                wp.from_torch(self.kernel_pos),
                pts_Gp,
                pts_dGp,
                pts_ddGp,
                pts_Nx, pts_dNx, pts_ddNx
            ]
        )

        return wp.to_torch(pts_Nx), wp.to_torch(pts_dNx), wp.to_torch(pts_ddNx)


    def collect_IP(self):
        n_IP = self.IP_pos.size(0)

        IP_mu = wp.zeros(shape=n_IP, dtype=wpfloat)
        IP_lam = wp.zeros(shape=n_IP, dtype=wpfloat)
        IP_rho = wp.zeros(shape=n_IP, dtype=wpfloat)

        wp.launch(
            kernel=cuda_utils.collect_param,
            dim=(self.pos.size(0), ),
            inputs=[
                wp.from_torch(self.pts_IP),
                wp.from_torch(self.mu),
                wp.from_torch(self.lam),
                IP_mu, IP_lam, IP_rho
            ]
        )

        IP_mu = wp.to_torch(IP_mu)
        IP_lam = wp.to_torch(IP_lam)
        IP_rho = wp.to_torch(IP_rho)

        return IP_mu / IP_rho, IP_lam / IP_rho, IP_rho / (self.dx ** 3)


    def build_global(self):












