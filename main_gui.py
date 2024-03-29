import os
pienerf_dir = os.path.dirname(os.path.abspath(__file__))
from nerf.gui import NeRFSimGUI
from nerf.trainer import *
from get_opts import *
import warp as wp
wp.init()
from simulator.solver import Simulator

os.environ['KMP_DUPLICATE_LIB_OK'] = '1'

def get_args(opt):
    args = []
    for arg in vars(opt):
        if isinstance(getattr(opt, arg), str):
            args.append(f"--{arg}")
            args.append(getattr(opt, arg))
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    opt = get_shared_opts(parser)

    from nerf.network import NeRFNetwork

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer('ngp', opt, model,
                      device=device, workspace=opt.workspace, use_checkpoint=opt.ckpt)

    sim = Simulator(
        dt=opt.sim_dt,
        iters=opt.sim_iters,
        bbox=torch.tensor([2.0*opt.bound, 2.0*opt.bound, 2.0*opt.bound]),
        dx=opt.sim_dx,
        stiff=opt.sim_stiff,
        base=torch.tensor([-opt.bound, -opt.bound, -opt.bound])
    )

    sim.InitializeFromPly(pienerf_dir + "/assets/" + opt.exp_name + ".ply")

    IP_pos, IP_F, IP_dF = sim.get_IP_info()
    print("dof=", IP_pos.shape[0])
    model.p_ori = IP_pos
    model.p_def = IP_pos
    model.IP_F = IP_F
    model.IP_dF = IP_dF
    model.IP_dx = sim.dx * 1.05

    output_ply = opt.output_ply
    if output_ply:
        if not os.path.exists(pienerf_dir + "/outputs_gui/"):
            os.mkdir(pienerf_dir + "/outputs_gui/")
        sim.OutputToPly(pienerf_dir + "/outputs_gui/0.ply")

    with torch.no_grad():
        gui = NeRFSimGUI(opt, trainer, sim, pause_each_frame=False, output_ply=output_ply)
        # -> test_step -> test_gui -> test_step -> update_one_step
        gui.render()




