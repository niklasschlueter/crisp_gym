import copy
import json
import sys
import time
import cv2
import numpy as np
from crisp_gym.envs.env_wrapper import (
    InsertionWrapper,
    NoRotationNoGripperActionWrapper,
    LastObservationWrapper,
)
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env, make_env_config
from crisp_gym.util.rl_utils import custom_reset, load_actions_safe
import gymnasium

# config = make_env_config("my_env_v2")

max_step_velocity = 0.005
max_gripper_step = 0.05

# env = ManipulatorCartesianEnv(config=config, namespace="left")
# env = make_env("my_env_v3_grav_comp_xyz")
# task.error_clip.z: 0.05 -> 0.003
# task.k_pos_z: 300.0 -> 5000.0
# task.d_pos_z: 20.0 -> 150.0
# env.robot.cartesian_controller_parameters_client.set_parameters([("task.k_pos_z", 300.0)])
# env.robot.cartesian_controller_parameters_client.set_parameters([("task.d_pos_z", 20.0)])
# env.robot.cartesian_controller_parameters_client.set_parameters([("task.error_clip.z", 0.05)])
env = make_env("my_env_v3")

print("Env created")
env.wait_until_ready()
print("Env ready.")

pickup_home = [
    -0.03049143,
    0.4469818,
    -0.02475526,
    -2.3372357,
    0.01456,
    2.7885091,
    0.71526223,
]  #  x y z


grasp_position_ground_truth = np.array([0.5106526, -0.03026352, 0.04147444])
goal_position_ground_truth = np.array([0.54262590, -0.030810941, 0.051836114])
env = NoRotationNoGripperActionWrapper(env)
env = LastObservationWrapper(env)
env = InsertionWrapper(
    env,
    home_config=pickup_home,
    grasp_position_ground_truth=grasp_position_ground_truth,
    goal_position_ground_truth=goal_position_ground_truth,
)
i = 0
try:
    while True:
        env.reset()
        time.sleep(1.0)
        print(i)
        i += 1
except KeyboardInterrupt:
    print("Interrupted, going home.")
    env.home(home_config=pickup_home)
    env.close()
