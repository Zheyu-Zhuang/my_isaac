import os
from omni.isaac.kit import SimulationApp

# Open Sim APP
config = {"headless": False}
simulation_app = SimulationApp(config)

# Any Omniverse level imports MUST occur after SimulationApp class is instantiated. 
from omni.isaac.core import World
from omni.isaac.core.utils import stage, prims
from omni.isaac.franka import Franka
from omni.isaac.sensor import Camera
import numpy as np

# Load Pre-made scene
# TODO: Move scene file path to config file
scene_path = os.path.join(os.getcwd(), "my_issac/asset/scene/panda_only.usd")
stage.open_stage(usd_path=scene_path)


sim_world = World()


camera = Camera("/World/camera")
bot = Franka("/World/Franka")

# reset after assigning objects
sim_world.reset()


#
# TODO: get image
# TODO: Scene and robot texture randomisation

sim_world.initialize_physics()

bot.initialize()
camera.initialize()


# TODO: Randomly add scene distractors
# TODO: Add a camera and assign random pose sampled from a hemisphere with centered around the manipulator base and optical axis jitter within a cone
# TODO: [Later] Add and randomly move camera based on config
# TODO: Move End-effector to random poses by solving ik, possibly load from saved valid joint poses.
# TODO: Collect RGB, Depth, Instance-mask, joint poses
import matplotlib.pyplot as plt

while True:
    bot.set_joint_positions(np.random.rand(9))
    cf = camera.get_current_frame()
    plt.imshow(cf["rgba"])
    plt.show()

    sim_world.step(render=True)
