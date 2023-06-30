CONFIG = {"renderer": "RayTracedLighting", "headless": True,
          "width": 1024, "height": 1024, "num_frames": 10,
          "num_envs": 4}

# Open Simulation App
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": CONFIG["headless"]})

import os
import json
import numpy as np
from PIL import Image

import carb
from omni.isaac.core import World, utils
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.cloner import GridCloner

import omni.replicator.isaac as dr
import omni.replicator.core as rep

# create the world


class DataGenerator:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self._check_output_path()
        self.num_envs = CONFIG["num_envs"]
        self.world = self._world()
        self.distractor_view, self.franka_view = self._add_view()
        self.world.reset()
        self._config_dr()
        self.cam_render = self._create_cam_render()
        self.rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.seg_annot = rep.AnnotatorRegistry.get_annotator(
            "instance_id_segmentation", init_params={"colorize": True})
        self.rgb_annot.attach(self.cam_render)
        self.seg_annot.attach(self.cam_render)
        # self.writer = self._writer(
        #     out_dir=os.path.join(os.getcwd(), "out"))

    def _world(self):
        world = World(stage_units_in_meters=1.0, 
                      physics_prim_path="/physicsScene",
                      backend="numpy")
        assets_root_path = utils.nucleus.get_assets_root_path()
        if assets_root_path is None:
            carb.log_error(
                "Could not find Isaac Sim assets folder, closing...")
            simulation_app.close()
        usd_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
        utils.stage.add_reference_to_stage(usd_path=usd_path,
                                           prim_path="/World/defaultGroundPlane")
        cloner = GridCloner(spacing=1.5)
        cloner.define_base_env("/World/envs")
        utils.prims.define_prim("/World/envs/env_0")

        # set up the first environment
        DynamicSphere(prim_path="/World/envs/env_0/object",
                      radius=0.1,
                      position=np.array([0.75, 0.0, 0.2]))
        utils.stage.add_reference_to_stage(
            usd_path=assets_root_path + "/Isaac/Robots/Franka/franka.usd",
            prim_path="/World/envs/env_0/franka"
        )
        # clone environments
        prim_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        cloner.clone(source_prim_path="/World/envs/env_0",
                     prim_paths=prim_paths)
        # assign semantic labels
        utils.semantics.add_update_semantics(
            utils.prims.get_prim_at_path("/World/envs/env_0/object"),
            semantic_label="sphere"
        )
        utils.semantics.add_update_semantics(
            utils.prims.get_prim_at_path("/World/envs/env_0/franka"),
            semantic_label="franka"
        )
        return world

    def _add_view(self):
        # creates the views and set up world
        distractor_view = RigidPrimView(prim_paths_expr="/World/envs/*/object",
                                        name="distractor_view")
        #
        franka_view = ArticulationView(prim_paths_expr="/World/envs/*/franka",
                                       name="franka_view")

        self.world.scene.add(distractor_view)
        self.world.scene.add(franka_view)
        return distractor_view, franka_view

    def _create_cam_render(self):
        # Create Camera and Register Writer
        cam_prim = utils.prims.create_prim(
            prim_path="/World/camera",
            prim_type="Camera",
            position=[4.8, 4.0, 2.5],
            orientation=[0.35377525, 0.24567726, 0.52083579, 0.73703178],
            attributes={"focusDistance": 400, "focalLength": 30, "clippingRange": (0.1, 10000000)},
        )
        RESOLUTION = (CONFIG["width"], CONFIG["height"])
        cam_render = rep.create.render_product(str(cam_prim.GetPrimPath()), 
                                               RESOLUTION)
        
        return cam_render
        
    def _config_dr(self):
        #TODO: sync base position shuffling and image saving. Manipulator base positions between images pairs should not differ.
        num_dof = self.franka_view.num_dof
        # set up randomization with omni.replicator.isaac, imported as dr
        dr.physics_view.register_simulation_context(self.world)
        dr.physics_view.register_rigid_prim_view(self.distractor_view)
        dr.physics_view.register_articulation_view(self.franka_view)
        with dr.trigger.on_rl_frame(num_envs=self.num_envs):
            with dr.gate.on_interval(interval=20):
                dr.physics_view.randomize_simulation_context(
                    operation="scaling", 
                    gravity=rep.distribution.uniform((1, 1, 0.0), (1, 1, 2.0))
                )
            with dr.gate.on_interval(interval=50):
                dr.physics_view.randomize_rigid_prim_view(
                    view_name=self.distractor_view.name, operation="direct", 
                    force=rep.distribution.uniform((0, 0, 2.5), (0, 0, 5.0))
                )
            with dr.gate.on_interval(interval=10):
                dr.physics_view.randomize_articulation_view(
                    view_name=self.franka_view.name,
                    operation="direct",
                    joint_velocities=rep.distribution.uniform(
                        tuple([-5] * num_dof), tuple([5] * num_dof)),
                )
            with dr.gate.on_env_reset():
                dr.physics_view.randomize_rigid_prim_view(
                    view_name=self.distractor_view.name,
                    operation="additive",
                    position=rep.distribution.normal((0.0, 0.0, 0.0), (0.2, 0.2, 0.0)),
                    velocity=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
                dr.physics_view.randomize_articulation_view(
                    view_name=self.franka_view.name,
                    operation="additive",
                    joint_positions=rep.distribution.uniform(
                        tuple([-0.8] * num_dof), tuple([0.8] * num_dof)),
                    position=rep.distribution.normal((0.0, 0.0, 0.0), (0.2, 0.2, 0.0)),
                )

    def _write_rgb_data(self, rgb_data, file_path):
        rgb_image_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape, -1)
        rgb_img = Image.fromarray(rgb_image_data, "RGBA")
        rgb_img.save(file_path + ".png")

    # Util function to save semantic segmentation annotator data
    def _write_seg_data(self, seg_data, file_path):
        id_to_labels = seg_data["info"]["idToLabels"]
        with open(file_path + ".json", "w") as f:
            json.dump(id_to_labels, f)
        seg_image_data = np.frombuffer(seg_data["data"], dtype=np.uint8).reshape(*seg_data["data"].shape, -1)
        seg_img = Image.fromarray(seg_image_data, "RGBA")
        seg_img.save(file_path + ".png")
    
    # TODO: align sub dir structure, and move it to config file
    
    def save_annot(self, frame_idx, out_dir="out"):
        self._write_rgb_data(self.rgb_annot.get_data(),
                             f"{out_dir}/rgb/{frame_idx:05}_rgb")
        self._write_seg_data(self.seg_annot.get_data(),
                             f"{out_dir}/segmentation/{frame_idx:05}_seg")
        with open(f"{out_dir}/joint/{frame_idx:05}_joint" + ".json", "w") as f:
            json.dump(self.franka_view.get_joint_positions().tolist(), f)
    
    # TODO: compute forward kinematics, and map joint pose to image
    
    def _check_output_path(self):
        all_sub_dir = ["", "rgb", "segmentation", "joint"]
        for d in all_sub_dir:
            p = os.path.join(self.out_dir, d)
            if not os.path.exists(p):
                os.makedirs(p)
            

if __name__ == "__main__":
    out_dir = os.path.join(os.getcwd(), "out")
    data_generator = DataGenerator(out_dir)
    # Run the application for several frames to allow the materials to load
    for i in range(20):
        simulation_app.update()
    
    frame_idx = 0
    sample_idx = 0
    while simulation_app.is_running():
        if data_generator.world.is_playing():
            reset_inds = list()
            if frame_idx % 200 == 0:
                # triggers reset every 200 steps
                reset_inds = np.arange(data_generator.num_envs)
            dr.physics_view.step_randomization(reset_inds)
            data_generator.world.step(render=False)
            if frame_idx % 3 == 0:
                data_generator.world.render()
                data_generator.world.render()
                data_generator.save_annot(sample_idx)
                sample_idx += 1
            frame_idx += 1

    # rep.orchestrator.run()

simulation_app.close()
