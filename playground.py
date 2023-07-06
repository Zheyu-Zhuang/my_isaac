import os

from omni.isaac.kit import SimulationApp

# Open Sim APP
config = {"headless": False}
simulation_app = SimulationApp(config)

import numpy as np
# Any Omniverse level imports MUST occur after SimulationApp class is instantiated. 
from omni.isaac.core import World
from omni.isaac.core.utils import prims, stage
from omni.isaac.franka import Franka
from omni.isaac.sensor import Camera

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

sim_world.initialize_physics()

bot.initialize()
camera.initialize()


import matplotlib.pyplot as plt

while True:
    bot.set_joint_positions(np.random.rand(9))
    cf = camera.get_current_frame()
    plt.imshow(cf["rgba"])
    plt.show()

    sim_world.step(render=True)
