import torch
import os
import sys
pienerf_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pienerf_dir)
from get_opts import *
from nerf.trainer import Trainer
from nerf.utils import *
from plyfile import PlyData, PlyElement
import numpy as np
import warp as wp
from nerf.utils import get_pnts_in_grids

def write_ply(filename, points, volumes, binary=True):
    # vertex = np.array([tuple(v) for v in points], dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])  # f8: float64
    vertex = np.array([tuple(v) + (vp,) for v, vp in zip(points, volumes)],
                      dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('vp', 'f8')])
    el = PlyElement.describe(vertex, 'vertex')
    if binary:
        PlyData([el]).write(filename)
    else:
        with open(filename, 'wb') as f:
            PlyData([el], text=True).write(f)

def read_ply(filename):
    plydata = PlyData.read(filename)
    vertex = plydata['vertex']
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    points = np.column_stack((x, y, z))
    return points

def distance(p0, p1):
    d = torch.norm(p0 - p1, p=2)
    # print(f"distance({p0},{p1})={d}")
    return d

@wp.func
def g2p(g: wp.vec3i, grid_size: wp.float32, bound: wp.float32):
    return wp.vec3f(wp.float32(g[0]) * grid_size - bound, wp.float32(g[1]) * grid_size - bound, wp.float32(g[2]) * grid_size - bound)

@wp.func
def p2g(p: wp.vec3f, grid_size: wp.float32, bound: wp.float32):
    return wp.vec3i(wp.int32(wp.floor((p[0] + bound) / grid_size)), wp.int32(wp.floor((p[1] + bound) / grid_size)), wp.int32(wp.floor((p[2] + bound) / grid_size)))

@wp.func
def hash_code(p: wp.vec3f, res: wp.float32, grid_size: wp.float32, bound: wp.float32):
    g = p2g(p, grid_size, bound)
    return int(wp.float32(g[2]) * res * res + wp.float32(g[1]) * res + wp.float32(g[0]))

@wp.func
def hash_code_g(g: wp.vec3i, res: wp.float32):
    return int(wp.float32(g[2]) * res * res + wp.float32(g[1]) * res + wp.float32(g[0]))

@wp.kernel
def get_grid_coords(
    res: wp.float32,
    grid_size: wp.float32,
    bound: wp.float32,
    pnts: wp.array(dtype=wp.vec3f),
    grid_coords: wp.array(dtype=wp.vec3i),
):
    gid = wp.tid()
    point = pnts[gid]
    g = p2g(point, grid_size, bound)
    # if g[0]==0 and g[1]==0 and g[2]==0:
    #     wp.printf("%f,%f,%f -> %i,%i,%i \n", point[0], point[1], point[2], g[0], g[1], g[2])
    gid = hash_code_g(g, res)
    grid_coords[gid] = g

@wp.kernel
def average_density(
        density_list: wp.array(dtype=wp.float32),
        count_list: wp.array(dtype=wp.int32),
):
    gid = wp.tid()
    density_list[gid] = density_list[gid] / wp.float32(count_list[gid])

@wp.kernel
def get_sub_bgn(
        tot: wp.array(dtype=wp.int32),
        sub_dims: wp.array(dtype=wp.int32),
        sub_bgn: wp.array(dtype=wp.int32),
):
    gid = wp.tid()
    sub_dim = sub_dims[gid]
    sub_bgn[gid] = wp.atomic_add(tot, wp.int32(0), sub_dim * sub_dim * sub_dim)

@wp.kernel
def get_pnts_add(
        points: wp.array(dtype=wp.vec3f),
        sub_mins: wp.array(dtype=wp.vec3f),
        sub_maxs: wp.array(dtype=wp.vec3f),
        sub_dims: wp.array(dtype=wp.int32),
        sub_bgn: wp.array(dtype=wp.int32),
        pnts_add: wp.array(dtype=wp.vec3f),
):
    gid = wp.tid()
    for i in range(sub_dims[gid] * sub_dims[gid] * sub_dims[gid]):
        scale = sub_maxs[gid] - sub_mins[gid]
        p = points[i]
        p = wp.vec3f(scale[0] * p[0], scale[1] * p[1], scale[2] * p[2])
        # p = wp.vec3f(scale[2] * p[2], scale[1] * p[1], scale[0] * p[0])
        p = p + sub_mins[gid]
        pnts_add[sub_bgn[gid] + i] = p

@wp.kernel
def get_sub_grid(
        bound: wp.float32,
        res: wp.float32,
        grid_size: wp.float32,
        sub_coeff: wp.float32,
        grid_density: wp.array(dtype=wp.float32),
        grid_coords: wp.array(dtype=wp.vec3i),
        sub_mins: wp.array(dtype=wp.vec3f),
        sub_maxs: wp.array(dtype=wp.vec3f),
        sub_dims: wp.array(dtype=wp.int32),
):
    gid = wp.tid()
    g0 = grid_coords[gid]
    g1 = g0 + wp.vec3i(0, 0, 1)
    g2 = g0 + wp.vec3i(0, 1, 0)
    g3 = g0 + wp.vec3i(0, 1, 1)
    g4 = g0 + wp.vec3i(1, 0, 0)
    g5 = g0 + wp.vec3i(1, 0, 1)
    g6 = g0 + wp.vec3i(1, 1, 0)
    g7 = g0 + wp.vec3i(1, 1, 1)
    d0 = grid_density[hash_code_g(g0, res)]
    d1 = grid_density[hash_code_g(g1, res)]
    d2 = grid_density[hash_code_g(g2, res)]
    d3 = grid_density[hash_code_g(g3, res)]
    d4 = grid_density[hash_code_g(g4, res)]
    d5 = grid_density[hash_code_g(g5, res)]
    d6 = grid_density[hash_code_g(g6, res)]
    d7 = grid_density[hash_code_g(g7, res)]
    grad_x = (d4 + d5 + d6 + d7 - (d0 + d1 + d2 + d3))
    grad_y = (d2 + d3 + d6 + d7 - (d0 + d1 + d4 + d5))
    grad_z = (d1 + d3 + d5 + d7 - (d0 + d2 + d4 + d6))
    grad_grid = wp.vec3f(grad_x, grad_y, grad_z)
    grad_norm = wp.length(grad_grid)
    if grad_norm == 0.0:
        sub_dims[gid] = wp.int32(0.0)
        sub_mins[gid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        sub_maxs[gid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    else:
        sub_mins[gid] = g2p(g0, grid_size, bound)
        sub_maxs[gid] = g2p(g7, grid_size, bound)
        sub_dims[gid] = wp.int32((sub_maxs[gid] - sub_mins[gid])[0] * sub_coeff * res * grad_norm)

class AdaptiveUniformSampling:
    def __init__(
            self,
            opt,
            model,
    ):
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.opt = opt
        self.bound = opt.bound
        self.threshold = opt.density_threshold
        self.res = opt.sub_res
        self.model = model.to(self.device)
        self.grid_size = 2 * self.bound / self.res

        if not os.path.exists(pienerf_dir + "/model"):
            os.mkdir(pienerf_dir + "/model")
        self.write_path = pienerf_dir + "/model/" + opt.workspace.split("/")[-1] + "/" + self.opt.exp_name

    def get_density(self, x):
        x = x.to(self.device)
        density = self.model.density(x.to(self.device))['sigma']
        density = 1 - torch.exp(-density / 128.0)
        return density

    def p2g(self, x):
        # x: [-bound, bound]
        return torch.floor((x + self.bound) / self.grid_size)

    def g2p(self, g):
        return g * self.grid_size - self.bound

    def hash_code(self, x):
        g = self.p2g(x)
        return int(g[2] * self.res * self.res + g[1] * self.res + g[0])

    def hash_code_g(self, g):
        return int(g[2] * self.res * self.res + g[1] * self.res + g[0])

    def get_point_volumes(self, pts):
        pts = pts.cuda()
        vols = torch.zeros(pts.shape[0], dtype=torch.float32).cuda()
        n_vtx = pts.shape[0]
        bmin = pts.min(axis=0).values
        bmax = pts.max(axis=0).values
        marg = 1e-3
        bbmin = bmin - marg * torch.ones(3, dtype=torch.float32).cuda()
        bbmax = bmax + marg * torch.ones(3, dtype=torch.float32).cuda()
        hgs = self.opt.hash_grid_size
        resolution = torch.ceil((bbmax - bbmin) / hgs).to(torch.int32)
        n_grid = resolution[2] * resolution[1] * resolution[0]
        pig_cnt, pig_bgn, pig_idx = get_pnts_in_grids(n_vtx, n_grid, pts, bbmin, bbmax, hgs, resolution)

        vol = hgs ** 3 / pig_cnt.float()
        for gid in range(n_grid):
            bgn = pig_bgn[gid]
            pid = pig_idx[bgn : bgn + pig_cnt[gid]]
            vols[pid] = vol[gid]
        return vols

    def sample(self):
        n_grid = self.res ** 3
        if self.opt.cut:
            if self.opt.cut_bounds[0] < -self.opt.bound: self.opt.cut_bounds[0] = -self.opt.bound
            if self.opt.cut_bounds[2] < -self.opt.bound: self.opt.cut_bounds[2] = -self.opt.bound
            if self.opt.cut_bounds[4] < -self.opt.bound: self.opt.cut_bounds[4] = -self.opt.bound
            if self.opt.cut_bounds[1] > self.opt.bound: self.opt.cut_bounds[1] = self.opt.bound
            if self.opt.cut_bounds[3] > self.opt.bound: self.opt.cut_bounds[3] = self.opt.bound
            if self.opt.cut_bounds[5] > self.opt.bound: self.opt.cut_bounds[5] = self.opt.bound
            assert self.opt.cut_bounds[0] < self.opt.cut_bounds[1]
            assert self.opt.cut_bounds[2] < self.opt.cut_bounds[3]
            assert self.opt.cut_bounds[4] < self.opt.cut_bounds[5]
            xs = torch.linspace(self.opt.cut_bounds[0], self.opt.cut_bounds[1], self.res)
            ys = torch.linspace(self.opt.cut_bounds[2], self.opt.cut_bounds[3], self.res)
            zs = torch.linspace(self.opt.cut_bounds[4], self.opt.cut_bounds[5], self.res)
            x_grid, y_grid, z_grid = custom_meshgrid(zs, ys, xs)
        else:
            xs = torch.linspace(-self.opt.bound, self.opt.bound, self.res)
            x_grid, y_grid, z_grid = custom_meshgrid(xs, xs, xs)

        coords = torch.stack([z_grid, y_grid, x_grid], dim=-1)
        grid_pts = coords.reshape(-1, 3).to(self.device)

        # ## debug #############################################
        # vols = self.get_point_volumes(grid_pts)
        # density = self.get_density(grid_pts)
        # write_ply(self.write_path + "_grid.ply", grid_pts[density > self.threshold].cpu().numpy(), vols.cpu().numpy())
        # print("writing to ", os.path.abspath(self.write_path + "_grid.ply"))
        # ######################################################

        print(f"{grid_pts.shape[0]} grid points sampled!")
        assert grid_pts.shape[0] > 0, "No grid points, check params!"
        grid_density = self.get_density(grid_pts).to(self.device)
        print(grid_density)
        print(f"grid density range: {grid_density.min()}~{grid_density.max()}")
        grid_coords = torch.zeros((n_grid,3), dtype=torch.int32, device=self.device)
        t = time.time()
        wp.launch(kernel=get_grid_coords, dim=(n_grid,),
                  inputs=[
                      self.res,
                      self.grid_size,
                      self.bound,
                      wp.from_torch(grid_pts, dtype=wp.vec3f),
                      wp.from_torch(grid_coords, dtype=wp.vec3i),
                  ],
                  device=self.device)

        sub_mins = torch.zeros((n_grid,3), dtype=torch.float32, device=self.device)
        sub_maxs = torch.zeros((n_grid,3), dtype=torch.float32, device=self.device)
        sub_dims = torch.zeros((n_grid,), dtype=torch.int32, device=self.device)
        sub_coeff = self.opt.sub_coeff # bigger, more points
        wp.launch(kernel=get_sub_grid, dim=(n_grid,),
                  inputs=[
                      self.bound, self.res, self.grid_size, sub_coeff,
                      wp.from_torch(grid_density, dtype=wp.float32),
                      wp.from_torch(grid_coords, dtype=wp.vec3i),
                      wp.from_torch(sub_mins, dtype=wp.vec3f),
                      wp.from_torch(sub_maxs, dtype=wp.vec3f),
                      wp.from_torch(sub_dims),
                  ],
                  device=self.device)

        sub_bgn = torch.zeros((n_grid,), dtype=torch.int32, device=self.device)
        sub_bgn.zero_()
        tot = torch.zeros((1,), dtype=torch.int32, device=self.device)
        wp.launch(kernel=get_sub_bgn, dim=(n_grid,),
                  inputs=[
                      wp.from_torch(tot),
                      wp.from_torch(sub_dims),
                      wp.from_torch(sub_bgn),
                  ],
                  device=self.device)

        max_dims = sub_dims.max()
        max_add = max_dims ** 3
        pnts_add = torch.zeros((tot[0], 3), dtype=torch.float32, device=self.device)
        points_tmp = torch.rand((max_add,3), dtype=torch.float32, device=self.device) # [0,0,0]-[1,1,1]
        wp.launch(kernel=get_pnts_add, dim=(n_grid,),
                  inputs=[
                      wp.from_torch(points_tmp, dtype=wp.vec3f),
                      wp.from_torch(sub_mins, dtype=wp.vec3f),
                      wp.from_torch(sub_maxs, dtype=wp.vec3f),
                      wp.from_torch(sub_dims),
                      wp.from_torch(sub_bgn),
                      wp.from_torch(pnts_add, dtype=wp.vec3f),
                  ],
                  device=self.device)
        assert pnts_add.shape[0] > 0, "No boundary points sampled, check params!"
        print(f"{pnts_add.shape[0]} boundary points sampled!")
        # ## debug #############################################
        # vols = self.get_point_volumes(pts)
        # density = self.get_density(pnts_add)
        # write_ply(self.write_path + "_add.ply", pnts_add[density > self.threshold], vols)
        # print("writing to ", os.path.abspath(self.write_path + "_add.ply"))
        # ######################################################

        pts = pnts_add.clone()
        pts = torch.cat((pts, grid_pts + 0.5 * 2 * self.opt.bound / float(self.res)), dim=0)
        density = self.get_density(pts)
        pts = pts[density > self.threshold]
        assert pts.shape[0] > 0, "No points sampled, check params!"

        vols = self.get_point_volumes(pts)

        print(f"{pts.shape[0]} points kept after thresholding!")
        write_ply(self.write_path + ".ply", pts.cpu().numpy(), vols.cpu().numpy())
        print("writing to ", os.path.abspath(self.write_path + ".ply"))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = get_shared_opts(parser)

    wp.init()
    # wp.config.mode = "debug"
    # wp.config.verify_cuda = True
    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    ckpt_path = os.path.join(opt.workspace, 'checkpoints')
    checkpoint_list = sorted(glob.glob(f'{ckpt_path}/ngp_ep*.pth'))
    if checkpoint_list:
        checkpoint = checkpoint_list[-1]
        print("reading ckpt: ", checkpoint)
        opt.ckpt_path = checkpoint
    else:
        print("no checkpoint found, ckpt_path:", ckpt_path)
        exit(-1)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )
    trainer = Trainer('ngp', opt, model, workspace=opt.workspace, use_checkpoint=opt.ckpt, eval_interval=50)

    AdaptiveUniformSampling(opt, model).sample()

