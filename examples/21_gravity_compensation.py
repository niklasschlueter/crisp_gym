import copy
import json
import sys
import time
import cv2
import numpy as np
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env, make_env_config
from crisp_gym.util.rl_utils import custom_reset, load_actions_safe

# config = make_env_config("my_env_v2")

max_step_velocity = 0.005
max_gripper_step = 0.05

# env = ManipulatorCartesianEnv(config=config, namespace="left")
# env = make_env("my_env_v3_grav_comp_xyz")
env = make_env("my_env_v3")

print("Env created")
env.wait_until_ready()
print("Env ready.")


obs, _ = env.reset()
# task.error_clip.z: 0.05 -> 0.003
# task.k_pos_z: 300.0 -> 5000.0
# task.d_pos_z: 20.0 -> 150.0

# env.robot.cartesian_controller_parameters_client.set_parameters([("task.error_clip.z", 0.003)])
# env.robot.cartesian_controller_parameters_client.set_parameters([("task.d_pos_z", 150.0)])
# env.robot.cartesian_controller_parameters_client.set_parameters([("task.k_pos_z", 5000.0)])

for _ in range(15 * 5):
    env.step(np.array([0.0, 0.0, 0.05 / (15 * 5), 0.0, 0.0, 0.0, 0.0]))

for _ in range(int(15 * 7.5)):
    env.step(np.array([0.0, 0.0, -0.05 / (15 * 5), 0.0, 0.0, 0.0, 0.0]))

input()


print("Going back home.")
env.home()
print("Homed.")
env.close()

# 3.0752587e-01,  1.4205652e-04,  4.8634282e-01
# 3.11061263e-01,  1.09783876e-04,  4.39315856e-01,
# 3.0972812e-01,  6.8447778e-05,  4.3932381e-01,
