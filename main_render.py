import os
pienerf_dir = os.path.dirname(os.path.abspath(__file__))
from nerf.gui import NeRFSimGUI
from nerf.provider import NeRFDataset
from nerf.trainer import *
from nerf.provider import nerf_matrix_to_ngp
from get_opts import *
import warp as wp
wp.init()
import json

def save_image(image, path, opt):
    from PIL import Image

    image_data = image
    image_data = np.clip(image_data, 0, 1) * 255
    image_data = image_data.astype(np.uint8)

    W = opt.W
    H = opt.H
    image_data = image_data.reshape(H, W, 3)

    image_to_save = Image.fromarray(image_data, 'RGB')

    print("saving to ", os.path.abspath(path))
    image_to_save.save(path)

def get_pose(file_dir, frame_str):
    try:
        file_path = file_dir + "/transforms_train.json"
        with open(file_path) as file:
            data = json.load(file)
    except FileNotFoundError:
        try:
            file_path = file_dir + "/transforms.json"
            with open(file_path) as file:
                data = json.load(file)
        except FileNotFoundError:
            print("no transforms found in ", file_dir, "!!!")
            return None
    for frame in data["frames"]:
        if frame_str in frame["file_path"]:
            print("reading pose from ", file_path, frame_str, "...")
            return np.array(frame["transform_matrix"], dtype=np.float32)
    return None


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

    dataset = NeRFDataset(opt, device=device, type='test')
    intrinsics = dataset.intrinsics

    data_root = opt.path
    pose = None

    if opt.workspace.split("/")[-1] == "dinosaur":
        pose = get_pose(data_root, "0057")
    else:
        print("pose not specified!")
        exit(-2)

    pose = nerf_matrix_to_ngp(pose, scale=opt.scale, offset=opt.offset)

    out_img_path = './output_img/'
    out_img_dir = out_img_path + '/' + opt.exp_name
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    points = np.load(f"./debug/ip_pos_0.npy")
    model.p_ori = torch.from_numpy(points)
    model.IP_dx = opt.sim_dx * 1.05
    with torch.no_grad():
        gui = NeRFSimGUI(opt, trainer, show=False)
        for def_frame in range(10,11):
            points = np.load(f"./debug/ip_pos_{def_frame}.npy")
            F = np.load(f"./debug/ip_F_{def_frame}.npy")
            dF = np.load(f"./debug/ip_dF_{def_frame}.npy")
            model.p_def = torch.from_numpy(points)
            model.IP_F = torch.from_numpy(F)
            model.IP_dF = torch.from_numpy(dF)
            output_image_path = out_img_dir + "/img_" + str(def_frame) + ".png"
            image = gui.get_render_buffer(pose, intrinsics, opt.W, opt.H, render_def=True)
            save_image(image, output_image_path, opt)
