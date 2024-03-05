import os.path

import torch
import warp as wp
import warp.torch
import warp.sparse as wps
import numpy as np
from plyfile import PlyData, PlyElement
from kornia.utils.grid import create_meshgrid3d
import cuda_utils
import cpu_utils
from cuda_utils import torchfloat, npfloat, wpfloat, vec3, vec10, vec8i, mat10, mat3
import scipy
from scipy.sparse import csc_matrix

class Simulator:
    def __init__(
            self,
            dt=1e-2,
            iters=20,
            res=torch.tensor([32, 32, 32], dtype=torch.int32),
            kres=7,
            dx=1,
            subspace=100,
            gravity=torch.tensor([0.0, -9.8, 0.0]),
            stiff=1e5,
            base=torch.tensor([-0.5, -0.5, -0.5])
    ):
        self.dt = dt
        self.iters = iters
        self.res = res
        self.dx = dx
        self.subspace = subspace
        self.base = base

        self.kres = kres

        self.gravity = gravity
        self.stiff = stiff

        # point
        self.pos = None
        self.grid_idx = None

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

        self.S = None
        self.U = None

        self.global_matrix = None
        self.global_non_diag = None
        self.diag = None

        self.mass_matrix_invt2 = None

        self.rhs_rest = None

        self.dof = None
        self.dof_rest = None
        self.dof_tilde = None
        self.dof_vel = None
        self.dof_f = None

        self.kdx = None


    def OutputToPly(self, path):
        self.update_pos()
        vertex = np.array([tuple(v) for v in self.pos.cpu().numpy().astype(np.float64)], dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])  # float64
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(path)



    def InitializeFromPly(self, path):
        plydata = PlyData.read(path)

        self.pos = torch.from_numpy(
            np.stack((np.asarray(plydata.elements[0]["x"]),
                      np.asarray(plydata.elements[0]["y"]),
                      np.asarray(plydata.elements[0]["z"])), axis=1).astype(npfloat)
        ).cuda()

        self.grid_idx = torch.from_numpy(
            np.stack((np.asarray(plydata.elements[0]["idx_x"]),
                      np.asarray(plydata.elements[0]["idx_y"]),
                      np.asarray(plydata.elements[0]["idx_z"])), axis=1).astype(np.int32)
        ).cuda()

        self.mass = torch.from_numpy(
            np.asarray(plydata.elements[0]["mass"]).astype(npfloat)
        ).cuda()
        self.mu = torch.from_numpy(
            np.asarray(plydata.elements[0]["mu"]).astype(npfloat)
        ).cuda()
        self.lam = torch.from_numpy(
            np.asarray(plydata.elements[0]["lam"]).astype(npfloat)
        ).cuda()
        self.is_pin = torch.from_numpy(
            np.asarray(plydata.elements[0]["pin"]).astype(bool)
        ).cuda()

        self.initialize()

    def initialize(self):
        self.IP_mask = torch.zeros(
            (self.res[0], self.res[1], self.res[2]),
            dtype=torch.bool
        )
        self.IP_mask[self.grid_idx[:, 0], self.grid_idx[:, 1], self.grid_idx[:, 2]] = True

        self.IP_idx = -torch.ones(
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
        IP_pos[:, :, :, [1, 2]] = IP_pos[:, :, :, [2, 1]]

        self.IP_grid = torch.zeros(
            (self.IP_mask.sum(), 3),
            dtype=torch.int32
        )
        self.IP_grid = IP_pos[self.IP_mask, :]

        self.IP_pos = self.IP_grid * self.dx + self.base + 0.5

        self.kernel_mask = torch.zeros(
            # (self.res[0] + 1, self.res[1] + 1, self.res[2] + 1),
            (self.kres, self.kres, self.kres),
            dtype=torch.bool
        )

        self.kdx = self.res.max() * self.dx / (self.kres - 1)

        self.IP_kernel = torch.zeros(
            (self.IP_mask.sum(), 8),
            dtype=torch.int32
        )
        IP2K = (self.IP_pos - self.base) // self.kdx
        IP2K = IP2K.to(dtype=torch.int32)
        for S in range(8):
            x = S >> 2 & 1
            y = S >> 1 & 1
            z = S & 1

            self.kernel_mask[
                IP2K[:, 0] + x,
                IP2K[:, 1] + y,
                IP2K[:, 2] + z
            ] |= True

        self.kernel_idx = torch.zeros(
            # (self.res[0] + 1, self.res[1] + 1, self.res[2] + 1),
            (self.kres, self.kres, self.kres),
            dtype=torch.int32
        )
        self.kernel_idx[self.kernel_mask] = torch.arange(0, self.kernel_mask.sum(), 1, dtype=torch.int32)

        self.pts_kernel = torch.zeros(
            (self.pos.size(0), 8),
            dtype=torch.int32
        )

        pts2K = (self.pos - self.base) // self.kdx
        pts2K = pts2K.to(dtype=torch.int32)

        for S in range(8):
            x = S >> 2 & 1
            y = S >> 1 & 1
            z = S & 1
            self.IP_kernel[:, S] = \
                self.kernel_idx[
                    IP2K[:, 0] + x,
                    IP2K[:, 1] + y,
                    IP2K[:, 2] + z
                ]
            self.pts_kernel[:, S] = \
                self.kernel_idx[
                    pts2K[:, 0] + x,
                    pts2K[:, 1] + y,
                    pts2K[:, 2] + z
                ]

        kernel_pos = create_meshgrid3d(
            self.kres, self.kres, self.kres,
            False,
            dtype=torch.int32
        ).reshape(self.kres, self.kres, self.kres, 3)
        kernel_pos[:, :, :, [1, 2]] = kernel_pos[:, :, :, [2, 1]]

        self.kernel_grid = torch.zeros(
            (self.kernel_mask.sum(), 3),
            dtype=torch.int32
        )
        self.kernel_grid = kernel_pos[self.kernel_mask, :]

        self.kernel_pos = self.kernel_grid * self.kdx + self.base

        self.pts_Nx, self.pts_dNx, self.pts_ddNx = self.init_GMLS(self.pos, self.pts_kernel)

        self.IP_Nx, self.IP_dNx, self.IP_ddNx = self.init_GMLS(self.IP_pos, self.IP_kernel)

        self.IP_mu, self.IP_lam, self.IP_rho = self.collect_IP()

        self.build_global()

        self.dof = torch.zeros(
            (self.kernel_mask.sum() * 30)
        )

        k_idx = torch.arange(0, self.kernel_mask.sum(), 1, dtype=torch.int32)
        for x in range(3):
            self.dof[k_idx * 30 + x] = self.kernel_pos[:, x]
            self.dof[k_idx * 30 + 3 + x * 3 + x] = 1
        self.dof_tilde = self.dof.clone()
        self.dof_rest = self.dof.clone()

        self.dof_vel = torch.zeros(
            (self.kernel_mask.sum() * 30)
        )
        self.dof_f = torch.zeros(
            (self.kernel_mask.sum() * 30)
        )

        self.rhs_rest = self.build_rhs() + self.mass_matrix_invt2 @ self.dof

        print(self.global_matrix @ self.dof - self.rhs_rest)


    def init_GMLS(self, pos, topo):
        n_pts = pos.size(0)
        pos = wp.from_torch(pos, dtype=vec3).to("cpu")
        topo = wp.from_torch(topo).to("cpu")
        pts_G = wp.array(shape=n_pts, dtype=mat10).to("cpu")
        pts_dG = wp.array(shape=(n_pts, 3), dtype=mat10).to("cpu")
        pts_ddG = wp.array(shape=(n_pts, 3, 3), dtype=mat10).to("cpu")

        wp.launch(
            kernel=cpu_utils.calc_G,
            dim=(n_pts, ),
            inputs=[
                self.kdx,
                pos,
                topo.to("cpu"),
                wp.from_torch(self.kernel_pos, dtype=vec3).to("cpu"),
                pts_G, pts_dG, pts_ddG
            ],
            device="cpu"
        )

        wp.synchronize()

        pts_G_inv = wp.from_torch(torch.linalg.inv(wp.to_torch(pts_G).cuda()).contiguous(), dtype=mat10).to("cpu")

        pts_Gp = wp.array(shape=n_pts, dtype=vec10).to("cpu")
        pts_dGp = wp.array(shape=(n_pts, 3), dtype=vec10).to("cpu")
        pts_ddGp = wp.array(shape=(n_pts, 3, 3), dtype=vec10).to("cpu")

        wp.launch(
            kernel=cpu_utils.calc_Gp,
            dim=(n_pts, ),
            inputs=[
                pos,
                pts_G,
                pts_G_inv,
                pts_dG,
                pts_ddG,
                pts_Gp, pts_dGp, pts_ddGp
            ],
            device="cpu"
        )

        wp.synchronize()

        pts_Nx = wp.zeros(shape=(n_pts, 8), dtype=vec10).to("cpu")
        pts_dNx = wp.zeros(shape=(n_pts, 8, 3), dtype=vec10).to("cpu")
        pts_ddNx = wp.zeros(shape=(n_pts, 8, 3, 3), dtype=vec10).to("cpu")

        wp.launch(
            kernel=cpu_utils.calc_weight,
            dim=(n_pts, ),
            inputs=[
                self.kdx,
                pos,
                topo,
                wp.from_torch(self.kernel_pos, dtype=vec3).to("cpu"),
                pts_Gp,
                pts_dGp,
                pts_ddGp,
                pts_Nx, pts_dNx, pts_ddNx
            ],
            device="cpu"
        )

        return wp.to_torch(pts_Nx).cuda(), wp.to_torch(pts_dNx).cuda(), wp.to_torch(pts_ddNx).cuda()


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
                wp.from_torch(self.mass),
                IP_mu, IP_lam, IP_rho
            ]
        )

        IP_mu = wp.to_torch(IP_mu)
        IP_lam = wp.to_torch(IP_lam)
        IP_rho = wp.to_torch(IP_rho)

        return IP_mu / IP_rho, IP_lam / IP_rho, IP_rho / (self.dx ** 3)


    def build_global(self):
        num_nonzero = (self.IP_pos.size(0) + self.is_pin.sum()) * 6400
        dimension = self.kernel_mask.sum() * 10
        mat = torch.zeros((dimension, dimension))
        wp_mat = wp.from_torch(mat)

        wp.launch(
            kernel=cuda_utils.build_IP_global,
            dim=(self.IP_pos.size(0) * 6400,),
            inputs=[
                wpfloat(self.dx),
                wpfloat(self.dt),
                wp.from_torch(self.IP_kernel),
                wp.from_torch(self.IP_mu),
                wp.from_torch(self.IP_lam),
                wp.from_torch(self.IP_rho),
                wp.from_torch(self.IP_Nx, dtype=vec10),
                wp.from_torch(self.IP_dNx, dtype=vec10),
                wp.from_torch(self.IP_ddNx, dtype=vec10),
                wp_mat
            ]
        )

        vid = torch.arange(0, self.pos.size(0), 1, dtype=torch.int32)
        vid = vid[self.is_pin]

        wp.launch(
            kernel=cuda_utils.build_pin_global,
            dim=(self.is_pin.sum() * 6400,),
            inputs=[
                wpfloat(self.stiff),
                wp.from_torch(vid),
                wp.from_torch(self.pts_kernel),
                wp.from_torch(self.pts_Nx, dtype=vec10),
                wp_mat
            ]
        )

        mat = wp.to_torch(wp_mat)
        global_matrix = torch.zeros((dimension * 3, dimension * 3))
        global_matrix[0::3, 0::3] = mat
        global_matrix[1::3, 1::3] = mat
        global_matrix[2::3, 2::3] = mat

        self.global_matrix = global_matrix
        global_matrix = global_matrix.cpu().numpy()

        print("eigen bg")

        # m_S, m_U = scipy.sparse.linalg.eigsh(
        #     global_matrix, k=self.subspace, which='SM'
        # )

        print("eigen fin")

        # self.S = torch.from_numpy(m_S).to(self.pos.device)
        # self.U = torch.from_numpy(m_U).to(self.pos.device)

        global_matrix = torch.from_numpy(global_matrix).cuda()
        idx = torch.arange(0, dimension * 3, 1, dtype=torch.int32)

        self.diag = global_matrix[idx, idx]

        self.global_non_diag = global_matrix
        self.global_non_diag[idx, idx] = 0.0

        wp_mat = wp.zeros(shape=(dimension, dimension), dtype=wpfloat)

        wp.launch(
            kernel=cuda_utils.build_IP_global,
            dim=(self.IP_pos.size(0) * 6400,),
            inputs=[
                wpfloat(self.dx),
                wpfloat(self.dt),
                wp.from_torch(self.IP_kernel),
                wp.zeros(shape=self.IP_pos.size(0), dtype=wpfloat),
                wp.zeros(shape=self.IP_pos.size(0), dtype=wpfloat),
                wp.from_torch(self.IP_rho),
                wp.from_torch(self.IP_Nx, dtype=vec10),
                wp.from_torch(self.IP_dNx, dtype=vec10),
                wp.from_torch(self.IP_ddNx, dtype=vec10),
                wp_mat
            ]
        )

        self.mass_matrix_invt2 = torch.zeros((dimension * 3, dimension * 3))

        mat = wp.to_torch(wp_mat)

        self.mass_matrix_invt2[0::3, 0::3] = mat
        self.mass_matrix_invt2[1::3, 1::3] = mat
        self.mass_matrix_invt2[2::3, 2::3] = mat


    def build_rhs(self):
        FF = wp.zeros(shape=self.IP_pos.size(0), dtype=mat3)
        RF = wp.zeros(shape=self.IP_pos.size(0), dtype=mat3)
        VF = wp.zeros(shape=self.IP_pos.size(0), dtype=mat3)
        wp.launch(
            kernel=cuda_utils.calc_elastic,
            dim=(self.IP_pos.size(0),),
            inputs=[
                wp.from_torch(self.IP_kernel),
                wp.from_torch(self.IP_dNx, dtype=vec10),
                wp.from_torch(self.dof.view(-1, 3), dtype=vec3),
                RF, VF, FF
            ]
        )

        # wp_rhs = wps.bsr_mv(self.mass_matrix_invt2, wp.from_torch(self.dof_tilde))
        # wp_rhs = wp.from_torch(wp.to_torch(wp_rhs).reshape(-1, 3), dtype=vec3)
        wp_rhs = wp.zeros(shape=self.dof.size(0) // 3, dtype=vec3)

        for dir in range(8):
            wp.launch(
                kernel=cuda_utils.collect_rhs_IP,
                dim=(self.IP_pos.size(0),),
                inputs=[
                    wpfloat(self.dx),
                    wp.int32(dir),
                    wp.from_torch(self.IP_kernel),
                    wp.from_torch(self.IP_mu),
                    wp.from_torch(self.IP_lam),
                    wp.from_torch(self.IP_dNx, dtype=vec10),
                    wp_rhs, RF, VF
                ]
            )

        return wp.to_torch(wp_rhs).reshape(-1)

    def jacobi(self, x, b):
        x = self.global_non_diag @ x
        x = b - x
        x /= self.diag
        return x


    def compute_momentum(self):
        self.dof_tilde = torch.zeros(self.dof.size(0))
        gravity = torch.zeros(self.dof.size(0)).reshape(-1, 3) + self.gravity.unsqueeze(0)
        self.dof_tilde += self.dof + self.dt * self.dof_vel + self.dt * self.dt * gravity.reshape(-1)
        return self.mass_matrix_invt2 @ self.dof_tilde + self.dof_f

    def update_force(self, vid, f):
        self.dof_f = torch.zeros(self.dof.size(0) // 3, 3)
        m = self.mass[vid]
        for i in range(8):
            kid = self.pts_kernel[vid, i]
            for j in range(10):
                self.dof_f[kid * 10 + j] += m * self.pts_Nx[vid, i][j] * f

        self.dof_f = self.dof_f.reshape(-1)


    def stepforward(self):
        momentum = self.compute_momentum()
        dof_last = self.dof.clone()
        for it in range(self.iters):
            rhs = momentum + self.build_rhs() - self.rhs_rest
            sub_rhs = self.U.permute(1, 0) @ rhs
            sub_x = sub_rhs / self.S
            x = self.U @ sub_x
            for i in range(5):
                x = self.jacobi(x, rhs)
            self.dof = self.dof_rest + x
        self.dof_vel = (self.dof - dof_last) / self.dt * 0.998

    def update_pos(self):
        self.pos = torch.zeros_like(self.pos)
        wp_pos = wp.zeros(shape=self.pos.size(0), dtype=vec3)
        wp.launch(
            kernel=cuda_utils.update_pos_kernel,
            dim=(self.pos.size(0),),
            inputs=[
                wp_pos,
                wp.from_torch(self.pts_kernel),
                wp.from_torch(self.dof.reshape(-1, 3), dtype=vec3),
                wp.from_torch(self.pts_Nx, dtype=vec10)
            ]
        )
        self.pos = wp.to_torch(wp_pos)





