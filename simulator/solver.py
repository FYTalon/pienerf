import scipy
import torch
import warp as wp
import numpy as np
from plyfile import PlyData, PlyElement
from kornia.utils.grid import create_meshgrid3d
from simulator import cuda_utils
from simulator import cpu_utils
from simulator.cuda_utils import torchfloat, npfloat, wpfloat, vec3, vec10, mat10, mat3


class Simulator:
    def __init__(
            self,
            dt=1e-2,
            iters=20,
            bbox=torch.tensor([2.0, 2.0, 2.0], dtype=torchfloat),
            kres=7,
            dx=1,
            gravity=torch.tensor([0.0, -9.8, 0.0], dtype=torchfloat),
            stiff=1e5,
            base=torch.tensor([-0.5, -0.5, -0.5], dtype=torchfloat)
    ):
        self.dt = dt
        self.iters = iters
        self.res = (bbox // dx).to(dtype=torch.int32).cuda()
        self.dx = dx
        self.base = base.cuda()

        self.kres = kres

        self.gravity = gravity.cuda()
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

        self.global_matrix = None

        self.mass_matrix_invt2 = None

        self.rhs_rest = None
        self.rhs_gravity = None

        self.dof = None
        self.dof_rest = None
        self.dof_tilde = None
        self.dof_vel = None
        self.dof_f = None

        self.kdx = None

        self.kernel_cnt = None
        self.kernel_bg = None
        self.tot = 0
        self.buffer = None


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

        self.grid_idx = (self.pos - self.base) // self.dx
        self.grid_idx = self.grid_idx.to(dtype=torch.int32)

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

        self.IP_pos = (self.IP_grid + 0.5) * self.dx + self.base

        self.kernel_mask = torch.zeros(
            (self.kres, self.kres, self.kres),
            dtype=torch.bool
        )

        self.kdx = ((self.res.max()) * self.dx) / (self.kres - 1)

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
            (self.kernel_mask.sum() * 30),
            dtype = torchfloat
        )

        k_idx = torch.arange(0, self.kernel_mask.sum(), 1, dtype=torch.int32)
        for x in range(3):
            self.dof[k_idx * 30 + x] = self.kernel_pos[:, x]
            self.dof[k_idx * 30 + 3 + x * 3 + x] = 1
        self.dof_tilde = self.dof.clone()
        self.dof_rest = self.dof.clone()

        self.dof_vel = torch.zeros(
            (self.kernel_mask.sum() * 30),
            dtype = torchfloat
        )
        self.dof_f = torch.zeros(
            (self.kernel_mask.sum() * 30),
            dtype = torchfloat
        )

        self.kernel_cnt = torch.zeros(
            (self.kernel_mask.sum()),
            dtype=torch.int32
        )

        wp_cnt = wp.zeros(shape=self.kernel_pos.size(0), dtype=wp.int32)

        wp.launch(
            kernel=cuda_utils.count_IP_kernel,
            dim=(self.IP_pos.size(0),),
            inputs=[
                wp.from_torch(self.IP_kernel),
                wp_cnt
            ]
        )

        self.kernel_cnt = wp.to_torch(wp_cnt)
        self.kernel_bg = torch.cumsum(self.kernel_cnt, dim=0, dtype=torch.int32) - self.kernel_cnt
        self.tot = self.kernel_cnt.sum()

        wp_buffer = wp.zeros(shape=self.kernel_cnt.sum().item(), dtype=wp.vec2i)
        wp_cnt = wp.zeros(shape=self.kernel_pos.size(0), dtype=wp.int32)

        wp.launch(
            kernel=cuda_utils.allocate_IP_kernel,
            dim=(self.IP_pos.size(0),),
            inputs=[
                wp.from_torch(self.IP_kernel),
                wp_cnt,
                wp.from_torch(self.kernel_bg),
                wp_buffer
            ]
        )

        self.buffer = wp.to_torch(wp_buffer)
        self.rhs_rest = self.build_rhs() + self.mass_matrix_invt2 @ self.dof

        wp_gravity = wp.zeros(shape=self.rhs_rest.size(0) // 3, dtype=vec3)

        wp.launch(
            kernel=cuda_utils.collect_gravity,
            dim=(self.IP_pos.size(0),),
            inputs=[
                self.dx,
                wp.from_torch(self.IP_kernel),
                wp.from_torch(self.IP_Nx, dtype=vec10),
                vec3(self.gravity[0], self.gravity[1], self.gravity[2]),
                wp.from_torch(self.IP_rho),
                wp_gravity
            ]
        )

        self.rhs_gravity = wp.to_torch(wp_gravity).view(-1)


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


    def get_IP_info(self):
        wp_pos = wp.zeros(shape=self.IP_pos.size(0), dtype=vec3)
        wp_F = wp.zeros(shape=self.IP_pos.size(0), dtype=mat3)
        wp_dF = wp.zeros(shape=(self.IP_pos.size(0), 3), dtype=mat3)

        wp.launch(
            kernel=cuda_utils.update_F_kernel,
            dim=(self.IP_pos.size(0),),
            inputs=[
                wp_pos,
                wp_F,
                wp_dF,
                wp.from_torch(self.IP_kernel),
                wp.from_torch(self.dof.view(-1, 3), dtype=vec3),
                wp.from_torch(self.IP_Nx, dtype=vec10),
                wp.from_torch(self.IP_dNx, dtype=vec10),
                wp.from_torch(self.IP_ddNx, dtype=vec10)
            ]
        )

        return wp.to_torch(wp_pos), wp.to_torch(wp_F).permute(0,2,1).contiguous().view(-1, 9), wp.to_torch(wp_dF).permute(0,3,2,1).contiguous().view(-1, 27)


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
        dimension = self.kernel_mask.sum() * 10
        mat = torch.zeros((dimension, dimension), dtype=torchfloat)
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
        global_matrix = torch.zeros((dimension * 3, dimension * 3), dtype=torchfloat)
        global_matrix[0::3, 0::3] = mat
        global_matrix[1::3, 1::3] = mat
        global_matrix[2::3, 2::3] = mat

        self.global_matrix = torch.zeros_like(global_matrix)

        lst = []
        for i in range(self.kernel_pos.size(0)):
            if global_matrix[i * 30, i * 30] > 0.0:
                for j in range(30):
                    lst.append(i * 30 + j)

        lst = torch.tensor(lst, dtype=torch.int32)
        mat = global_matrix[lst][:, lst]
        idx = torch.arange(0, mat.size(0), 1, dtype=torch.int32)
        mat[idx, idx] += 1e-3
        mat = mat.inverse()
        tmp = torch.zeros((dimension * 3, lst.size(0)), dtype=torchfloat).cuda()
        tmp[lst] = mat
        self.global_matrix[:, lst] = tmp

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

        self.mass_matrix_invt2 = torch.zeros((dimension * 3, dimension * 3), dtype=torchfloat)

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

        wp_rhs = wp.zeros(shape=self.dof.size(0) // 3, dtype=vec3)

        wp.launch(
            kernel=cuda_utils.collect_rhs_IP,
            dim=(self.IP_pos.size(0),),
            inputs=[
                wpfloat(self.dx),
                wp.from_torch(self.IP_kernel),
                wp.from_torch(self.IP_mu),
                wp.from_torch(self.IP_lam),
                wp.from_torch(self.IP_dNx, dtype=vec10),
                wp_rhs, RF, VF
            ]
        )

        return wp.to_torch(wp_rhs).reshape(-1)


    def compute_momentum(self):
        self.dof_tilde = self.dof + self.dt * self.dof_vel
        return self.mass_matrix_invt2 @ self.dof_tilde + self.dof_f + self.rhs_gravity

    def update_force(self, vid, f):
        self.dof_f = torch.zeros((self.dof.size(0) // 3, 3), dtype=torchfloat)
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
            x = self.global_matrix @ rhs
            self.dof = self.dof_rest + x
        self.dof_vel = (self.dof - dof_last) / self.dt * 0.998

    def update_pos(self):
        self.pos = torch.zeros_like(self.pos, dtype=torchfloat)
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





