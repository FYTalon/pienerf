import os
from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.gui import NeRFSimGUI
from nerf.trainer import *
from get_opts import *
import warp as wp
wp.init()
# wp.config.mode = "debug"
# wp.config.verify_cuda = True
from solver import Simulator

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
    opt.current_folder = os.path.dirname(os.path.abspath(__file__))

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

    model = NeRFNetwork( # NeRFNetwork inherits NeRFRenderer
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
        dt=1e-2,
        iters=10,
        bbox=torch.tensor([2*opt.bound, 2*opt.bound, 2*opt.bound]),
        dx=0.05,
        stiff=1e7,
        base=torch.tensor([-opt.bound, -opt.bound, -opt.bound])
    )

    sim.InitializeFromPly("./assets/" + opt.exp_name + ".ply")

    IP_pos, IP_F, IP_dF = sim.get_IP_info()
    model.p_ori = IP_pos
    model.p_def = IP_pos
    model.IP_F = IP_F.view(-1, 9)
    model.IP_dF = IP_dF.view(-1, 27)

    with torch.no_grad():
        gui = NeRFSimGUI(opt, trainer, sim)
        # -> test_step -> test_gui -> test_step -> update_one_step

        gui.render()





