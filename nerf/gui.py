import math
import torch
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import json
from .utils import *
from .provider import nerf_matrix_to_ngp

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
    print("no pose named", frame_str, "in ", file_path)
    return None

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value-1))

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    def pose_to_params(self, pose):
        self.radius = - self.center[2] + pose[:3, 3][2] # assuming center not changed
        rotation_matrix = pose[:3, :3]
        self.rot = R.from_matrix(rotation_matrix)

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius # translate +z
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])


class NeRFGUI:
    def __init__(self, opt, trainer, train_loader=None, debug=True, show=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.show = show
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg
        self.training = False
        self.step = 0 # training step

        self.trainer = trainer
        self.train_loader = train_loader
        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth']

        self.dynamic_resolution = True
        self.downscale = 1
        self.train_steps = 16

        if self.show:
            dpg.create_context()
            self.register_dpg()
            self.test_step()


    def __del__(self):
        if self.show:
            dpg.destroy_context()


    def train_step(self):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        outputs = self.trainer.train_gui(self.train_loader, step=self.train_steps)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        self.need_update = True

        dpg.set_value("_log_train_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
        dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        else:
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)

    # added for single image rendering
    def get_render_buffer(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, render_def=False):
        outputs = self.trainer.test_gui(pose, intrinsics, W, H, bg_color, spp, downscale, render_def=render_def)
        render_buffer = self.prepare_buffer(outputs)
        return render_buffer

    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?

        if self.need_update or self.spp < self.opt.max_spp:

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            outputs = self.trainer.test_gui(
                self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, self.spp, self.downscale
            ) # NeRFGUI

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = self.prepare_buffer(outputs)
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
                self.spp += 1

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)


    def register_dpg(self):

        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=300, collapsed=True):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            if not self.opt.test:
                with dpg.group(horizontal=True):
                    dpg.add_text("Train time: ")
                    dpg.add_text("no data", tag="_log_train_time")

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            with dpg.group(horizontal=True):
                dpg.add_text("SPP: ")
                dpg.add_text("1", tag="_log_spp")

            # train button
            if not self.opt.test:
                with dpg.collapsing_header(label="Train", default_open=True):

                    # train / stop
                    with dpg.group(horizontal=True):
                        dpg.add_text("Train: ")

                        def callback_train(sender, app_data):
                            if self.training:
                                self.training = False
                                dpg.configure_item("_button_train", label="start")
                            else:
                                self.training = True
                                dpg.configure_item("_button_train", label="stop")

                        dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                        dpg.bind_item_theme("_button_train", theme_button)

                        def callback_reset(sender, app_data):
                            @torch.no_grad()
                            def weight_reset(m: nn.Module):
                                reset_parameters = getattr(m, "reset_parameters", None)
                                if callable(reset_parameters):
                                    m.reset_parameters()
                            self.trainer.model.apply(fn=weight_reset)
                            self.trainer.model.reset_extra_state() # for cuda_ray density_grid and step_counter
                            self.need_update = True

                        dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)
                        dpg.bind_item_theme("_button_reset", theme_button)

                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        def callback_save(sender, app_data):
                            self.trainer.save_checkpoint(full=True, best=False)
                            dpg.set_value("_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="save", tag="_button_save", callback=callback_save)
                        dpg.bind_item_theme("_button_save", theme_button)

                        dpg.add_text("", tag="_log_ckpt")

                    # save mesh
                    with dpg.group(horizontal=True):
                        dpg.add_text("Marching Cubes: ")

                        def callback_mesh(sender, app_data):
                            self.trainer.save_mesh(resolution=256, threshold=10)
                            dpg.set_value("_log_mesh", "saved " + f'{self.trainer.name}_{self.trainer.epoch}.ply')
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="mesh", tag="_button_mesh", callback=callback_mesh)
                        dpg.bind_item_theme("_button_mesh", theme_button)

                        dpg.add_text("", tag="_log_mesh")

                    with dpg.group(horizontal=True):
                        dpg.add_text("", tag="_log_train_log")


            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution, callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(('image', 'depth'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32) # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor", no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

                # dt_gamma slider
                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f", default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

                # max_steps slider
                def callback_set_max_steps(sender, app_data):
                    self.opt.max_steps = app_data
                    self.need_update = True

                dpg.add_slider_int(label="max steps", min_value=1, max_value=1024, format="%d", default_value=self.opt.max_steps, callback=callback_set_max_steps)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.trainer.model.aabb_infer[user_data] = app_data

                    # also change train aabb ? [better not...]
                    #self.trainer.model.aabb_train[user_data] = app_data

                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("Axis-aligned bounding box:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="x", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=0)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="y", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=1)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="z", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=2)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=5)


            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")

        ## register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)


        dpg.create_viewport(title='torch-ngp', width=self.W, height=self.H, resizable=False)

        # TODO: seems dearpygui doesn't support resizing texture...
        # def callback_resize(sender, app_data):
        #     self.W = app_data[0]
        #     self.H = app_data[1]
        #     # how to reload texture ???

        # dpg.set_viewport_resize_callback(callback_resize)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):

        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()


class NeRFSimGUI:
    def __init__(self, opt, trainer, sim, train_loader=None, debug=False, show=True, pause_each_frame=False, output_ply=False):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.show = show
        self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg

        self.trainer = trainer
        self.train_loader = train_loader
        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        self.spp = 1  # sample per pixel
        self.mode = 'image'  # choose from ['image', 'depth']

        self.dynamic_resolution = True
        self.downscale = 1
        self.train_steps = 16

        self.solver = sim ####
        self.frame = 0 ####
        self.paused = True
        self.pause_each_frame = pause_each_frame
        self.output_ply = output_ply
        # self.ctrl_pressed = False
        self.mouse_dragging = False
        self.depth = None
        self.depth_average = None
        self.rays_o = None
        self.rays_d = None
        self.sid = None
        self.pts, _, _ = self.solver.get_IP_info()
        self.pts = self.pts.cpu().numpy()

        self.render_IP = False
        self.log_FPS = False
        self.fps_values = []
        self.max_frames_to_capture = 30

        if self.show:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.show:
            dpg.destroy_context()

    def reset_camera(self): # TODO
        self.cam = OrbitCamera(self.W, self.H, r=self.opt.radius, fovy=self.opt.fovy)
        self.need_update = True

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        else:
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)

    def overlay_point_buffer(self, points):
        for pos in points.cpu().numpy():
            screen_x, screen_y, _ = self.world_to_screen(pos)
            screen_y, screen_x = int(screen_x), int(screen_y)
            # if 0 <= screen_x < self.W and 0 <= screen_y < self.H:
            if 0 <= screen_x < self.render_buffer.shape[0] and 0 <= screen_y < self.render_buffer.shape[1]:
                self.render_buffer[screen_x][screen_y] = np.array([0, 0, 1])
        return self.render_buffer

    def update_fps_plot(self):
        if dpg.does_item_exist("line_series_fps_log"):
            x_values = list(range(len(self.fps_values)))
            y_values = self.fps_values
            if len(self.fps_values) > 30:
                x_values = x_values[-30:]
                y_values = y_values[-30:]
            dpg.set_value("line_series_fps_log", [x_values, y_values])
            min_value, max_value = min(y_values), max(y_values)
            ticks = list(range(int(min_value), int(max_value) + 1, 1))
            dpg.set_axis_ticks("_fps_y_axis", tuple((str(tick), tick) for tick in ticks))
            dpg.set_axis_limits("_fps_x_axis", min(x_values), 30 if max(x_values) < 30 else max(x_values))
            dpg.set_axis_limits("_fps_y_axis", min_value, max_value)

    def test_step(self):
        self.pts, _, _ = self.solver.get_IP_info()
        self.pts = self.pts.cpu().numpy()

        if self.sid is not None:
            x1, y1 = dpg.get_mouse_pos()
            dpg.delete_item("c0")
            dpg.delete_item("c1")
            dpg.delete_item("l1")
            x0, y0, _ = self.world_to_screen(self.pts[self.sid])
            dpg.draw_circle((x0, y0), color=[255, 0, 0, 255], radius=10, parent="_primary_window", tag="c0", fill=(255, 0, 0))
            dpg.draw_circle((x1, y1), color=[0, 0, 255, 255], radius=10, parent="_primary_window", tag="c1", fill=(0, 0, 255))
            dpg.draw_line((x0, y0), (x1, y1), color=[0, 0, 0, 100], thickness=5, parent="_primary_window", tag="l1")
            p1, _ = self.screen_to_world(x1, y1)
            p0 = self.pts[self.sid]
            mouse_force_3d = 1e5 * (p1 - p0)
            # print(f"force: {np.linalg.norm(mouse_force_3d)}")
            force_norm = np.linalg.norm(mouse_force_3d)
            if force_norm > 5e5:
                mouse_force_3d *= 5e5 / force_norm
            self.solver.update_force(self.sid, torch.tensor([mouse_force_3d[0], mouse_force_3d[1], mouse_force_3d[2]]))

        if not self.paused:
            self.need_update = True

        t = 0
        if self.need_update or self.spp < self.opt.max_spp:

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics, self.W, self.H,
                                            self.bg_color, self.spp, self.downscale,
                                            render_def=True,####
                                            gui_sim=True,
                                            solver=self.solver,
                                            frame=self.frame,
                                            paused=self.paused,
                                            output_ply=self.output_ply
                                            ) # NeRFSimGUI

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 80 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1 / 4, math.sqrt(80 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = self.prepare_buffer(outputs)
                if self.render_IP:
                    IP_pos, _, _ = self.solver.get_IP_info()
                    self.render_buffer = self.overlay_point_buffer(IP_pos)
                self.spp = 1
                self.need_update = False
            # else:
            #     self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
            #     self.spp += 1

            self.depth = outputs['depth_0']
            self.average_depth = np.mean(self.depth[np.nonzero(self.depth)])
            self.rays_o = outputs['rays_o']
            self.rays_d = outputs['rays_d']
            # print(type(self.depth))
            # print("depth: ", np.nanmin(self.depth), "~", np.nanmax(self.depth))

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000 / t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            # dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_log_frame", self.frame)
            dpg.set_value("_texture", self.render_buffer)

        if not self.paused:
            if self.pause_each_frame:
                self.paused = True
            self.fps_values.append(int(1000 / t))
            self.frame = self.frame + 1
            if self.log_FPS:
                self.update_fps_plot()

    def screen_to_world(self, x, y):
        fx, fy, cx, cy = self.cam.intrinsics
        depth = self.depth.reshape(2 * int(cx), 2 * int(cy))[clamp(int(x), 0, 2 * int(cx)), clamp(int(y), 0, 2 * int(cy))]
        if depth == 0.0:
            depth = self.average_depth
        # depth = d
        xs = (x - cx) / fx * depth
        ys = (y - cy) / fy * depth
        zs = depth
        cam_coords = np.array([xs, ys, zs, 1.0])
        pos3d = self.cam.pose @ cam_coords
        return pos3d[:3], depth

    def world_to_screen(self, pos):
        fx, fy, cx, cy = self.cam.intrinsics
        pos_ = np.array([*pos, 1.0])
        cam_coords = np.linalg.inv(self.cam.pose) @ pos_
        xc, yc, zc = cam_coords[:3]
        xs = xc / zc * fx + cx
        ys = yc / zc * fy + cy
        return xs, ys, zc

    def register_dpg(self):

        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=20):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            with dpg.group(horizontal=True):
                dpg.add_text("frame: ")
                dpg.add_text("0", tag="_log_frame")

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_load_pose(sender, app_data, user_data):
                pose_id = dpg.get_value(sender)
                pose = get_pose(self.opt.path, pose_id)
                if pose is not None:
                    pose = nerf_matrix_to_ngp(pose, scale=self.opt.scale, offset=self.opt.offset)
                    self.cam.pose_to_params(pose)
                    self.need_update = True

            with dpg.group(horizontal=True):
                dpg.add_input_text(label="<- input pose ID & enter", default_value="330",
                               on_enter=True, callback=callback_load_pose, width=100)
                dpg.add_button(label="?", width=20, tag='pose_hint_button')
            with dpg.tooltip('pose_hint_button'):
                dpg.add_text('Pose ID is a substring of "file_path" attribute in transforms(_train).json, e.g., 400 for dataset trex')

            def callback_render_IP(sender, app_data):
                self.render_IP = not self.render_IP
                self.need_update = True
            dpg.add_checkbox(label="render IPs", default_value=False, callback=callback_render_IP)

            # FPS logging chart
            def callback_log_FPS(sender, app_data):
                self.log_FPS = not self.log_FPS
                dpg.configure_item("fps_over_time_plot", show=self.log_FPS)
            dpg.add_checkbox(label="log FPS", default_value=False, callback=callback_log_FPS)

            with dpg.plot(label="FPS Over Time", show=False, height=160, width=-1, no_title=True, no_mouse_pos=True, tag="fps_over_time_plot"):
                dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="_fps_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="FPS", tag="_fps_y_axis")
                dpg.add_line_series(list(range(len(self.fps_values))), self.fps_values, parent="_fps_y_axis", tag="line_series_fps_log")
                # dpg.set_axis_limits_auto("_fps_x_axis")
                # dpg.set_axis_limits_auto("_fps_y_axis")
                dpg.set_axis_limits("_fps_x_axis", 0, 30)
                dpg.set_axis_limits("_fps_y_axis", 10, 20)

            # with dpg.group(horizontal=True):
            #     dpg.add_text("SPP: ")
            #     dpg.add_text("1", tag="_log_spp")

            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution,
                                     callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(('image', 'depth'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor",
                                   no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg",
                                   default_value=self.cam.fovy, callback=callback_set_fovy)

                # dt_gamma slider
                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f",
                                     default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

                # max_steps slider
                def callback_set_max_steps(sender, app_data):
                    self.opt.max_steps = app_data
                    self.need_update = True

                dpg.add_slider_int(label="max steps", min_value=1, max_value=1024, format="%d",
                                   default_value=self.opt.max_steps, callback=callback_set_max_steps)


            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")

        def get_mouse_pos():
            x, y = dpg.get_mouse_pos()
            y = y + 20
            return x, y

        ### register keyboard & mouse handler
        def callback_key_press(sender, app_data):
            if dpg.is_key_pressed(32): # space
                self.paused = not self.paused
            if dpg.is_key_pressed(81): # Q
                self.sid = None
                dpg.delete_item("c0")
                dpg.delete_item("c1")
                dpg.delete_item("l1")
                self.mouse_dragging = False
            if dpg.is_key_pressed(99): # C
                self.reset_camera()

        def callback_key_release(sender, app_data):
            # if dpg.is_key_released(17): # ctrl
            #     self.ctrl_pressed = False
            pass

        def callback_mouse_click(sender, app_data):
            if dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and dpg.is_key_down(17): # ctrl
                x, y = get_mouse_pos()
                pos, d = self.screen_to_world(x, y)
                dp = self.pts - pos
                distances = np.sum(dp ** 2, axis=1)
                self.sid = np.argmin(distances)
                self.mouse_dragging = True
                # print(f"mouse clicked at ({x},{y}), depth={d}, pos={pos}, closest={self.pts[self.sid]}, distance={len(pos-self.pts[self.sid])}")

            if dpg.is_mouse_button_down(dpg.mvMouseButton_Right):
            # if dpg.is_mouse_button_pressed(dpg.mvMouseButton_Right):
                self.sid = None
                dpg.delete_item("c0")
                dpg.delete_item("c1")
                dpg.delete_item("l1")
                self.mouse_dragging = False

        def callback_mouse_move(sender, app_data):
            pass

        def callback_mouse_release(sender, app_data):
            pass

        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=callback_key_press)
            dpg.add_key_release_handler(callback=callback_key_release)
            dpg.add_mouse_click_handler(callback=callback_mouse_click)
            dpg.add_mouse_move_handler(callback=callback_mouse_move)
            dpg.add_mouse_release_handler(callback=callback_mouse_release)

        ### register camera handler
        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            if self.mouse_dragging:
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            if self.mouse_dragging:
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            if self.mouse_dragging:
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        dpg.create_viewport(title='pie-nerf', width=self.W, height=self.H, resizable=False)

        # TODO: seems dearpygui doesn't support resizing texture...
        # def callback_resize(sender, app_data):
        #     self.W = app_data[0]
        #     self.H = app_data[1]
        #     # how to reload texture ???

        # dpg.set_viewport_resize_callback(callback_resize)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):

        while dpg.is_dearpygui_running():
            self.test_step()
            dpg.render_dearpygui_frame()
