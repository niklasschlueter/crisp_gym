"""Environment wrappers for extending Gym environments with additional functionality.

This module provides wrappers that add features like observation stacking and receding
horizon control to Gym environments. These wrappers can be used to modify the behavior
of environments without changing their core implementation.

The module includes:
    - WindowWrapper: Stacks a fixed-size window of past observations along a new time dimension
    - RecedingHorizon: Applies a sequence of actions in a receding horizon manner
    - stack_gym_space: Helper function to repeat/stack Gym spaces
"""

import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from numpy.typing import NDArray

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv


def stack_gym_space(space: gym.Space, repeat: int) -> gym.Space:
    """Repeat a Gym space definition by stacking it multiple times.

    Args:
        space (gym.Space): The original Gym space to be repeated.
        repeat (int): Number of times to repeat/stack the space.

    Returns:
        gym.Space: A new Gym space with the original space stacked 'repeat' times.

    Raises:
        ValueError: If the input space type is not supported (currently supports Box and Dict spaces).
    """
    if isinstance(space, gym.spaces.Box):
        # Convert dtype to type to match Box constructor's type annotation
        dtype = np.dtype(space.dtype).type
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=dtype,
        )
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: stack_gym_space(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise ValueError(f"Space {space} is not supported.")


class WindowWrapper(gym.Wrapper):
    """A Gym wrapper that stacks a fixed-size window of past observations along a new time dimension.

    This allows agents to receive a temporal context of the environment by maintaining a history
    of the most recent `window_size` observations. The wrapper modifies the observation space
    to include the temporal dimension, making it compatible with policies that expect
    temporal information.

    Attributes:
        window_size (int): Number of consecutive observations to stack.
        window (list): List of most recent observations.
        observation_space (gym.Space): Modified observation space that includes temporal dimension.
    """

    def __init__(self, env: ManipulatorBaseEnv, window_size: int) -> None:
        """Initialize the WindowWrapper.

        Args:
            env (ManipulatorBaseEnv): The environment to wrap.
            window_size (int): Number of consecutive observations to stack.
        """
        super().__init__(env)
        self.window_size = window_size
        self.window = []
        self.observation_space = stack_gym_space(
            self.env.observation_space, self.window_size
        )

    def step(
        self, action: NDArray[np.float32], **kwargs: Any
    ) -> Tuple[Dict[str, NDArray[np.float32]], float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment.

        Args:
            action (np.ndarray): An action provided by the agent.
            **kwargs: Additional keyword arguments passed to the environment's step function.

        Returns:
            tuple:
                - dict: The current observation.
                - float: Amount of reward returned after previous action.
                - bool: Whether the episode has ended.
                - bool: Whether the episode was truncated.
                - dict: Contains auxiliary diagnostic information.
        """
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        self.window.append(obs)
        self.window = self.window[-self.window_size :]
        obs = {
            key: np.stack([frame[key] for frame in self.window])
            for key in self.window[0].keys()
        }
        return obs, float(reward), terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed (int, optional): The seed that is used to initialize the environment's PRNG.
            options (dict, optional): Additional information to specify how the environment is reset.

        Returns:
            tuple:
                - dict: The initial observation.
                - dict: Additional information.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.window = [obs] * self.window_size
        obs = {
            key: np.stack([frame[key] for frame in self.window])
            for key in self.window[0].keys()
        }
        return obs, info

    def close(self) -> None:
        """Clean up the environment's resources."""
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the wrapped environment.

        Args:
            name (str): Name of the attribute to get.

        Returns:
            Any: The value of the requested attribute.
        """
        return getattr(self.env, name)


class RecedingHorizon(gym.Wrapper):
    """A Gym wrapper that takes a sequence of actions and applies them in a receding horizon manner.

    This wrapper allows the agent to plan and execute a sequence of actions over a fixed time horizon.
    At each step, the agent provides a sequence of actions, and the wrapper executes them sequentially
    until either the horizon is reached or the episode terminates.
    """

    def __init__(self, env: ManipulatorBaseEnv, horizon_length: int) -> None:
        """Initialize the RecedingHorizon wrapper.

        Args:
            env (ManipulatorBaseEnv): The environment to wrap.
            horizon_length (int): The number of steps to look ahead and execute actions for.
        """
        super().__init__(env)
        self.horizon_length = horizon_length
        self.action_space = stack_gym_space(self.env.action_space, self.horizon_length)

    def step(
        self, action: NDArray[np.float32], **kwargs: Any
    ) -> Tuple[Dict[str, NDArray[np.float32]], float, bool, bool, Dict[str, Any]]:
        """Execute a sequence of actions over the horizon length.

        Args:
            action (np.ndarray): A sequence of actions to execute. Shape should be
                (horizon_length, action_dim) or (action_dim,) for horizon_length=1.
            **kwargs: Additional keyword arguments passed to the environment's step function.

        Returns:
            tuple:
                - dict: The final observation after executing all actions.
                - float: Sum of rewards received over the horizon.
                - bool: Whether the episode has ended.
                - bool: Whether the episode was truncated.
                - dict: Contains auxiliary diagnostic information.

        Raises:
            AssertionError: If the action sequence length is less than horizon_length.
        """
        obs = {}
        rewards = []
        terminated = False
        truncated = False
        info = {}

        if self.horizon_length == 1 and len(action.shape) == 1:
            action = action[None]
        assert action.shape[0] >= self.horizon_length

        for i in range(self.horizon_length):
            obs, reward, terminated, truncated, info = self.env.step(
                action[i], **kwargs
            )
            rewards.append(reward)
            if terminated or truncated:
                break

        return obs, float(np.sum(rewards)), terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed (int, optional): The seed that is used to initialize the environment's PRNG.
            options (dict, optional): Additional information to specify how the environment is reset.

        Returns:
            tuple:
                - dict: The initial observation.
                - dict: Additional information.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def close(self) -> None:
        """Clean up the environment's resources."""
        if rclpy.ok():  # pyright: ignore[reportPrivateImportUsage]
            rclpy.shutdown()
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the wrapped environment.

        Args:
            name (str): Name of the attribute to get.

        Returns:
            Any: The value of the requested attribute.
        """
        return getattr(self.env, name)


class ActionTimeStampWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        time_stamp = time.time()
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["action_t"] = time_stamp
        return observation, float(reward), terminated, truncated, info


class NoRotationNoGripperActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (3,))

    def action(self, action):
        return np.concatenate((action, np.zeros(4)))


class LastObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_cartesian_state = None
        self.last_angular_state = None
        self.last_gripper_state = None
        self.last_cartesian_error = None
        self.last_angular_error = None
        self.last_gripper_error = None
        self.last_t_obs = None

    @staticmethod
    def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
            dtype=np.float64,
        )

    @classmethod
    def _rotvec_to_matrix(cls, rotvec: NDArray[np.float64]) -> NDArray[np.float64]:
        theta = float(np.linalg.norm(rotvec))
        if theta < 1e-12:
            return np.eye(3, dtype=np.float64) + cls._skew(rotvec)

        axis = rotvec / theta
        k = cls._skew(axis)
        return (
            np.eye(3, dtype=np.float64)
            + np.sin(theta) * k
            + (1.0 - np.cos(theta)) * (k @ k)
        )

    @staticmethod
    def _matrix_to_rotvec(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
        cos_theta = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        theta = float(np.arccos(cos_theta))

        if theta < 1e-7:
            return 0.5 * np.array(
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ],
                dtype=np.float64,
            )

        sin_theta = float(np.sin(theta))
        if abs(sin_theta) > 1e-7:
            axis = np.array(
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ],
                dtype=np.float64,
            ) / (2.0 * sin_theta)
            return axis * theta

        # Near pi, infer axis from diagonal terms for numerical stability.
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
        axis[0] = np.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = np.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis[2] = np.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            return np.array([theta, 0.0, 0.0], dtype=np.float64)
        return (axis / axis_norm) * theta

    @classmethod
    def _relative_rotation_error(
        cls,
        target_rotvec: NDArray[np.float64],
        current_rotvec: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        rotation_target = cls._rotvec_to_matrix(target_rotvec)
        rotation_current = cls._rotvec_to_matrix(current_rotvec)
        rotation_error = rotation_target @ rotation_current.T
        return cls._matrix_to_rotvec(rotation_error)

    @classmethod
    def _relative_angular_velocity(
        cls,
        current_rotvec: NDArray[np.float64],
        previous_rotvec: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        if dt <= 0.0:
            return np.zeros_like(current_rotvec)

        rotation_current = cls._rotvec_to_matrix(current_rotvec)
        rotation_previous = cls._rotvec_to_matrix(previous_rotvec)
        delta_rotation = rotation_current @ rotation_previous.T
        return cls._matrix_to_rotvec(delta_rotation) / dt

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        current_cartesian_state = observation["observation.state.cartesian"][:3]
        current_angular_state = observation["observation.state.cartesian"][3:].astype(
            np.float64
        )
        target_angular_state = observation["observation.state.target"][3:].astype(
            np.float64
        )
        observation["observation.previous.action"] = np.zeros(
            self.env.action_space.shape or 7
        )
        observation["observation.previous.error.cartesian"] = np.zeros(3)
        observation["observation.previous.error.angular"] = np.zeros(3)

        observation["observation.velocity.cartesian"] = np.zeros_like(
            current_cartesian_state
        )
        observation["observation.velocity.angular"] = np.zeros_like(
            current_angular_state
        )
        observation["observation.error.cartesian"] = (
            observation["observation.state.target"][:3] - current_cartesian_state
        )
        observation["observation.error.angular"] = self._relative_rotation_error(
            target_angular_state, current_angular_state
        )

        self.last_cartesian_state = current_cartesian_state
        self.last_angular_state = current_angular_state
        self.last_cartesian_error = observation["observation.error.cartesian"]
        self.last_angular_error = observation["observation.error.angular"]
        self.last_t_obs = time.perf_counter()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        t_obs = time.perf_counter()
        current_cartesian_state = observation["observation.state.cartesian"][:3]
        current_angular_state = observation["observation.state.cartesian"][3:].astype(
            np.float64
        )
        target_angular_state = observation["observation.state.target"][3:].astype(
            np.float64
        )
        dt = (
            t_obs - self.last_t_obs
            if self.last_cartesian_state is not None and self.last_t_obs is not None
            else 0.0
        )

        observation["observation.previous.action"] = action
        observation["observation.previous.error.cartesian"] = self.last_cartesian_error
        observation["observation.previous.error.angular"] = self.last_angular_error

        observation["observation.velocity.cartesian"] = (
            (current_cartesian_state - self.last_cartesian_state) / dt
            if self.last_cartesian_state is not None and self.last_t_obs is not None
            else np.zeros_like(current_cartesian_state)
        )
        observation["observation.velocity.angular"] = (
            self._relative_angular_velocity(
                current_angular_state,
                self.last_angular_state,
                dt,
            )
            if self.last_angular_state is not None and self.last_t_obs is not None
            else np.zeros_like(current_angular_state)
        )
        observation["observation.error.cartesian"] = (
            observation["observation.state.target"][:3] - current_cartesian_state
        )
        observation["observation.error.angular"] = self._relative_rotation_error(
            target_angular_state, current_angular_state
        )

        self.last_cartesian_state = current_cartesian_state
        self.last_angular_state = current_angular_state
        self.last_cartesian_error = observation["observation.error.cartesian"]
        self.last_angular_error = observation["observation.error.angular"]
        self.last_t_obs = t_obs
        return observation, reward, terminated, truncated, info


# Foundationpose interface wrapper
# on reset:
#   reset with custom homing position, gripper open

#   A)
#   take observation,
#   PE:
#   pass to SAM3 for segmentation,
#   look at average pixel color to determine which block is which => error: set rollout unusable flag -> add to step info
#   2x:
#       set param in fp and fpt to correct mesh
#       get pose from fp,
#       set orientation to prior,
#       pass 4x to fpt
#   compute relative transform from observed pose to demo pose for grasped block in global frame

#   B) compute from stored demo

#   go to demo pose with other controller config
#   grasp block and move up

#   A)
#   PE of grasped block
#   compute relative transform from grasped block to placed block in global frame

#   B)
#   use known demo pose of grasped block for delta xy

#   switch controller back
#   make delta xy 0
#   move down until contact (e.g. force threshold)
# while stepping:
#   use estimated pose for safety box; step size as parameter
#   apply z-force


# crisp gym: no image cropping in ManipulatorEnv

# global safety box: make wrapper instead of modifying env directly

# env = ContainerWatcherWrapper(env, ctx=multiprocessing.get_context("spawn"))
# env = CLIWrapper(env, termination_fn=lambda _obs: False)
# env = StepLimitEnforcerWrapper(env, max_steps=150)

# env = ImageEncoderWrapper(env, n_cameras=1, image_size=(256, 256))  -> add custom cropping
# env = DictObservationToInfoMover(env)
# env = ObservationFormatterWrapper(
#     env,
#     "cuda",
#     keys_ranges_scales=[
#         ("observation.previous.action", (0, 2), 10.0),
#         ("observation.previous.error.cartesian", (0, 3), 10.0),
#         ("observation.velocity.cartesian", (0, 3), 100.0),
#         ("observation.error.cartesian", (0, 3), 10.0),
#         ("observation.images.wrist_camera", (0, 512), 1.0),
#         # ('observation.images.side_camera', (0, 512), 1.0)
#     ],
# )

# automatic termination? -> not yet
