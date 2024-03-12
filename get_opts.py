import argparse

def get_shared_opts(parser):
    parser.add_argument('path', type=str)

    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1,
                        help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--T_thresh', type=float, default=1e-2, help="stop ray marching when transmittance < T_thresh, larger threshold, faster rendering")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2.0,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")

    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1 / 128,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1,
                        help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1,
                        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    # model parameters
    parser.add_argument('--exp_name', type=str, default='exp', help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None, help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--vres', type=int, default=96)
    parser.add_argument('--con', type=int, default=1) # num of connected components to keep

    parser.add_argument('--dataset_type', type=str, default="")# --scale 0.8 --bound 1.0 --dt_gamma 0.0 --W 800 --H 800

    # sampling settings
    parser.add_argument('--density_threshold', type=float, default=0.05)
    parser.add_argument('--sub_coeff', type=float, default=0.1, help="bigger, more boundary points")
    parser.add_argument('--sub_res', type=int, default=20, help="bigger, more grid points")
    parser.add_argument('--cut', action='store_true')
    parser.add_argument('--cut_bounds', nargs=6, type=float, default=[0.0,2.0,-2.0,1.0,-1.42,0.92])

    # rendering settings
    parser.add_argument('--num_seek_IP', type=int, default=1)
    parser.add_argument('--timing_on', action='store_true')
    parser.add_argument('--output_ply', action='store_true')
    parser.add_argument('--max_iter_num', type=int, default=100, help="maximum number of Newton iterations in quadratic bending")

    # simulator settings
    parser.add_argument('--sim_dt', type=float, default=1e-2)
    parser.add_argument('--sim_dx', type=float, default=0.05)
    parser.add_argument('--sim_iters', type=int, default=10)
    parser.add_argument('--sim_stiff', type=float, default=1e5)

    opt = parser.parse_args()
    opt.hash_grid_size = 1.2 * opt.sim_dx
    opt.num_seek_IP = max(min(3, opt.num_seek_IP), 1)
    print(f"num_seek_IP={opt.num_seek_IP}")

    if opt.dataset_type == "synthetic":
        opt.scale = 0.8
        opt.bound = 1.0
        opt.dt_gamma = 0.0
        opt.W = 800
        opt.H = 800
    # else:
    #     opt.scale = 0.33
    #     opt.bound = 2.0
    #     opt.W = 1920
    #     opt.H = 1080

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.patch_size > 1:
        opt.error_map = False  # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    return opt

