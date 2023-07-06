CONFIG = {
    "renderer": "PathTracing",
    "headless": False,
    "width": 512,
    "height": 512,
    "num_envs": 4,
    "num_samples": 10000,
    "sample_every_n_frames": 2,
    "dataset_name": "test",
    "num_lights": 3,
    "reset_every_n_frames": 50
    # "anti_aliasing": "FXAA",
}


# Open Simulation App
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(launch_config=CONFIG)

import json
import os

import carb
import numpy as np
import omni.replicator.core as rep
import omni.replicator.isaac as dr
import yaml
from franka_fk import FrankaFK
from omni.isaac.cloner import GridCloner
from omni.isaac.core import World, utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere, GroundPlane
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
from PIL import Image, ImageDraw
from spatialmath.base import q2r, r2q
from tqdm import tqdm

# create the world


class DataGenerator:
    def __init__(self, out_dir):
        # path setup
        self.out_dir = out_dir
        self.check_output_path()
        self.num_envs = CONFIG["num_envs"]
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.result = False
            return
        self.dome_texture_path = assets_root_path + "/NVIDIA/Assets/Skies/"
        texture_config = self.load_yaml("my_issac/texture_config.yaml")
        self.dome_texture_paths = [
            self.dome_texture_path + dome_texture + ".hdr"
            for dome_texture in texture_config["TEXTURES"]
        ]
        # add stuff
        self.world = self._create_world()
        self.distractor_view, self.franka_view = self.add_view()

        self.world.reset()
        self.camera, self.cam_render = self.create_cam_render()
        self.camera = Camera("/World/camera")
        self.camera.set_resolution((CONFIG["width"], CONFIG["height"]))
        # self.camera_node = rep.get.prim_at_path("/World/camera")
        self.config_dr()
        self.rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.seg_annot = rep.AnnotatorRegistry.get_annotator(
            "instance_id_segmentation", init_params={"colorize": True}
        )
        self.rgb_annot.attach(self.cam_render)
        self.seg_annot.attach(self.cam_render)
        self.franka_kf = FrankaFK()


    def load_yaml(self, file_path: str):
        with open(file_path, "r") as file:
            try:
                config = yaml.load(file, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)
        return config

    def randomize_camera_pose(self):
        # rand point on a cone   (uniform)
        d = np.random.uniform(low=6, high=8)  # distance to the camera
        z = np.random.uniform(low=0.4 * d, high=0.8 * d)
        theta = np.random.uniform(low=0, high=2 * np.pi)
        r = np.sqrt(d**2 - z**2)  # radius on slice
        x, y = r * np.cos(theta), r * np.sin(theta)

        direction_vec = -np.array([x, y, z])
        direction_vec /= np.linalg.norm(direction_vec)
        R = np.eye(3)
        # Camera class uses +x forward, z up convention for the camera frame
        for sign in [-1, 1]:
            y_cam = np.array(
                [sign * direction_vec[1], -sign * direction_vec[0], 0]
            )
            y_cam /= np.linalg.norm(y_cam)
            z_cam = np.cross(direction_vec, y_cam)
            if z_cam[2] > 0:
                break
        R[:, 0] = direction_vec
        R[:, 1] = y_cam
        R[:, 2] = z_cam
        self.camera.set_world_pose(position=[x, y, z], orientation=r2q(R))

    def _setup_randomizers(self):
        """Add domain randomization with Replicator Randomizers"""

        # Create and randomize sphere lights
        def randomize_sphere_lights():
            lights = rep.create.light(
                light_type="Sphere",
                color=rep.distribution.uniform(
                    (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
                ),
                intensity=rep.distribution.uniform(1000, 8000),
                position=rep.distribution.uniform((0, 0, 0), (50, 50, 100)),
                scale=rep.distribution.uniform(1, 20),
                count=CONFIG["num_lights"],
            )
            return lights.node

        def randomize_domelight(texture_paths):
            lights = rep.create.light(
                light_type="Dome",
                rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                texture=rep.distribution.choice(texture_paths),
            )
            return lights.node

        rep.randomizer.register(randomize_domelight, override=True)
        rep.randomizer.register(randomize_sphere_lights, override=True)

    def _create_world(self):
        # rep.settings.set_render_rtx_realtime()
        world = World(
            stage_units_in_meters=1.0,
            physics_prim_path="/physicsScene",
            backend="numpy",
        )
        assets_root_path = utils.nucleus.get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder, closing...")
            simulation_app.close()

        usd_path = (
            assets_root_path
            + "/Isaac/Environments/Grid/default_environment.usd"
        )
        GroundPlane("/World/ground", visible=False)
        cloner = GridCloner(spacing=1.5)
        cloner.define_base_env("/World/envs")
        utils.prims.define_prim("/World/envs/env_0")

        # set up the first environment
        utils.prims.create_prim(
            "/World/Light/WhiteSphere",
            "SphereLight",
            translation=(0, 0, 10.0),
            attributes={
                "radius": 2.5,
                "intensity": 10000.0,
                "color": (1.0, 1.0, 1.0),
            },
        )

        DynamicSphere(
            prim_path="/World/envs/env_0/object",
            radius=0.1,
            position=np.array([0.75, 0.0, 0.2]),
        )
        utils.stage.add_reference_to_stage(
            usd_path=assets_root_path + "/Isaac/Robots/Franka/franka.usd",
            prim_path="/World/envs/env_0/franka",
        )
        # clone environments
        prim_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        cloner.clone(
            source_prim_path="/World/envs/env_0", prim_paths=prim_paths
        )
        # assign semantic labels
        utils.semantics.add_update_semantics(
            utils.prims.get_prim_at_path("/World/envs/env_0/object"),
            semantic_label="sphere",
        )
        utils.semantics.add_update_semantics(
            utils.prims.get_prim_at_path("/World/envs/env_0/franka"),
            semantic_label="franka",
        )

        self.dome_lights = rep.create.light(
            light_type="Dome",
            rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
            texture=rep.distribution.choice(self.dome_texture_paths),
        )

        world.play()

        return world

    def add_view(self):
        # creates the views and set up world
        distractor_view = RigidPrimView(
            prim_paths_expr="/World/envs/*/object", name="distractor_view"
        )
        #
        franka_view = ArticulationView(
            prim_paths_expr="/World/envs/*/franka", name="franka_view"
        )

        self.world.scene.add(distractor_view)
        self.world.scene.add(franka_view)
        return distractor_view, franka_view

    def create_cam_render(self):
        # Create Camera and Register Writer
        cam_prim = utils.prims.create_prim(
            prim_path="/World/camera",
            prim_type="Camera",
            position=[4.8, 4.0, 2.5],
            orientation=[0.35377525, 0.24567726, 0.52083579, 0.73703178],
            attributes={
                "focusDistance": 400,
                "focalLength": 35,
                "clippingRange": (0.1, 10000),
            },
        )
        RESOLUTION = (CONFIG["width"], CONFIG["height"])
        cam_render = rep.create.render_product(
            str(cam_prim.GetPrimPath()), RESOLUTION
        )
        return cam_prim, cam_render

    def config_dr(self):
        num_dof = self.franka_view.num_dof
        # set up randomization with omni.replicator.isaac, imported as dr
        dr.physics_view.register_simulation_context(self.world)
        dr.physics_view.register_rigid_prim_view(self.distractor_view)
        dr.physics_view.register_articulation_view(self.franka_view)

        self._setup_randomizers()

        with dr.trigger.on_rl_frame(num_envs=self.num_envs):
            # with dr.gate.on_interval(interval=20):
            #     dr.physics_view.randomize_simulation_context(
            #         operation="scaling",
            #         gravity=rep.distribution.uniform((1, 1, 0.0), (1, 1, 2.0))
            #     )
            with dr.gate.on_interval(interval=50):
                # self._randomize_lights()

                dr.physics_view.randomize_rigid_prim_view(
                    view_name=self.distractor_view.name,
                    operation="direct",
                    force=rep.distribution.uniform((0, 0, 2.5), (0, 0, 5.0)),
                )
            with dr.gate.on_interval(interval=10):
                dr.physics_view.randomize_articulation_view(
                    view_name=self.franka_view.name,
                    operation="direct",
                    joint_velocities=rep.distribution.uniform(
                        tuple([-5] * num_dof), tuple([5] * num_dof)
                    ),
                )
            with dr.gate.on_env_reset():
                dr.physics_view.randomize_rigid_prim_view(
                    view_name=self.distractor_view.name,
                    operation="additive",
                    position=rep.distribution.normal(
                        (0.0, 0.0, 0.0), (0.2, 0.2, 0.0)
                    ),
                    velocity=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
                dr.physics_view.randomize_articulation_view(
                    view_name=self.franka_view.name,
                    operation="additive",
                    joint_positions=rep.distribution.uniform(
                        tuple([-0.8] * num_dof), tuple([0.8] * num_dof)
                    ),
                    position=rep.distribution.normal(
                        (0.0, 0.0, 0.0), (0.2, 0.2, 0.0)
                    ),
                )

    def write_rgb_data(self, rgb_data, file_path, debug_sync=False):
        # rgb = rgb_data[:, :, :3].astype(np.uint8)
        # rgb_img = Image.fromarray(rgb, "RGB")
        rgb_image_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(
            *rgb_data.shape, -1
        )
        rgb_img = Image.fromarray(rgb_image_data, "RGBA")
        rgb_img.convert("RGB")
        if debug_sync:
            joint_positions = self.get_joint_positions(group_by_envs=False)
            draw = ImageDraw.Draw(rgb_img)
            # base_points = self.franka_view.get_world_poses()
            # print(joint_positions)
            cam_coords = self.camera.get_image_coords_from_world_points(
                joint_positions
            )
            for im_coord in cam_coords.tolist():
                circ_size = 2
                circ_coord = (
                    int(im_coord[0]),
                    int(im_coord[1]),
                    int(im_coord[0]) + circ_size,
                    int(im_coord[1]) + circ_size,
                )
                # print(im_coord)
                draw.ellipse(circ_coord, fill="coral")
        rgb_img.save(file_path + ".png")

    # Util function to save semantic segmentation annotator data
    def write_seg_data(self, seg_data, file_path):
        id_to_labels = seg_data["info"]["idToLabels"]
        with open(file_path + ".json", "w") as f:
            json.dump(id_to_labels, f)
        seg_image_data = np.frombuffer(
            seg_data["data"], dtype=np.uint8
        ).reshape(*seg_data["data"].shape, -1)
        seg_img = Image.fromarray(seg_image_data, "RGBA")
        seg_img.save(file_path + ".png")

    # TODO: align sub dir structure, and move it to config file

    def get_joint_positions(self, group_by_envs=True) -> list:
        # joint position for n manipulators,  n = num_envs
        n_q = self.franka_view.get_joint_positions()
        base_t_quat = self.franka_view.get_world_poses()
        # print(n_q, base_t_quat)
        out = []
        for i in range(self.num_envs):
            # get manipulator base transformation w.r.t the world frame
            base_X = np.eye(4)
            base_X[:3, 3] = base_t_quat[0][i]
            base_X[:3, :3] = q2r(base_t_quat[1][i])
            q_i = np.array(n_q[i][0 : self.franka_kf.dof])
            X = base_X @ self.franka_kf.get_joint_poses(q_i)
            if group_by_envs:
                out.append(X[:, :3, 3].tolist())
            else:
                out += X[:, :3, 3].tolist()
        return out

    def get_joint_positions_in_cam(self) -> list:
        joint_positions = self.get_joint_positions(group_by_envs=False)
        return self.camera.get_image_coords_from_world_points(
            joint_positions
        ).tolist()

    def get_robot_base_poses(self) -> list:
        base_t_quat = self.franka_view.get_world_poses()
        out = []
        for i in range(self.num_envs):
            # get manipulator base transformation w.r.t the world frame,
            # [x, y, z, quat_w, quat_x, quat_y, quat_z]
            out.append(base_t_quat[0][i].tolist() + base_t_quat[1][i].tolist())
        return out

    def save_annot(self, frame_id, debug_sync=False):
        with open(
            f"{self.out_dir}/meta/{frame_id}_joint_angles" + ".json", "w"
        ) as f:
            json.dump(self.franka_view.get_joint_positions().tolist(), f)
        with open(
            f"{self.out_dir}/meta/{frame_id}_base_poses" + ".json", "w"
        ) as f:
            json.dump(self.get_robot_base_poses(), f)
        with open(
            f"{self.out_dir}/meta/{frame_id}_joint_cam_coords" + ".json", "w"
        ) as f:
            json.dump(self.get_joint_positions_in_cam(), f)

        self.write_rgb_data(
            self.rgb_annot.get_data(),
            f"{self.out_dir}/rgb/{frame_id}_rgb",
            debug_sync,
        )
        self.write_seg_data(
            self.seg_annot.get_data(),
            f"{self.out_dir}/segmentation/{frame_id}_seg",
        )
        # get manipulator proprioceptive readings
        # print(self.franka_view.body_names)

    def check_output_path(self):
        all_sub_dir = ["", "rgb", "segmentation", "meta"]
        for d in all_sub_dir:
            p = os.path.join(self.out_dir, d)
            if not os.path.exists(p):
                os.makedirs(p)


if __name__ == "__main__":
    dataset_path = os.path.join(
        os.path.expanduser("~"), "Dataset", CONFIG["dataset_name"]
    )
    print(
        f"Save dataset to {dataset_path}, dataset size: {CONFIG['num_samples']}"
    )
    data_generator = DataGenerator(dataset_path)
    # Run the application for several frames to allow the materials to load

    scene_idx = -1
    frame_idx = 0
    sample_idx = 0

    for _ in range(20):
        simulation_app.update()

    for _ in tqdm(
        range(CONFIG["num_samples"] * CONFIG["sample_every_n_frames"])
    ):
        reset_inds = list()
        if frame_idx % CONFIG["reset_every_n_frames"] == 0:
            # triggers reset every n steps
            with data_generator.dome_lights:
                rep.modify.attribute(
                    "texture:file",
                    rep.distribution.choice(data_generator.dome_texture_paths),
                )
            data_generator.randomize_camera_pose()
            reset_inds = np.arange(data_generator.num_envs)
            scene_idx += 1
            sample_idx = 0
        dr.physics_view.step_randomization(reset_inds)
        data_generator.world.step(render=False)
        # TODO: Check update -- Multiple render() calls for getting around the image buffer delay bug
        # https://forums.developer.nvidia.com/t/problem-with-images-i-get-from-cameras/252380/3
        if frame_idx % CONFIG["sample_every_n_frames"] == 0:
            data_generator.world.render()
            data_generator.world.render()
            data_generator.world.render()
            data_generator.world.render()
            data_generator.world.render()
            # data_generator.world.render()
            file_prefix = f"{scene_idx:04}_{sample_idx:03}"
            data_generator.save_annot(file_prefix)
            sample_idx += 1
        frame_idx += 1

simulation_app.close()
