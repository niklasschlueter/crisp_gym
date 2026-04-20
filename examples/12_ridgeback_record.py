#!/usr/bin/env python3
"""UR10e crisp_gym environment — data recording example.

Records robot demonstrations in LeRobot format. The robot is controlled
externally (spacemouse, mocap, manual jogging, etc.) — this script captures
observations and the current target pose as the action for each frame.

Usage:
    python examples/12_ridgeback_record.py --repo-id my_dataset
    python examples/12_ridgeback_record.py --repo-id my_dataset --resume
    python examples/12_ridgeback_record.py --repo-id my_dataset --num-episodes 50

Keyboard controls (episode management):
    r  →  start recording
    r  →  stop / pause recording
    s  →  save episode
    d  →  discard episode
    q  →  quit

Prerequisites:
    - Controllers loaded and running on the real robot
    - cartesian_controller activated
    - joint_state_broadcaster and pose_broadcaster active
    - Camera running (pixi run orbbec in clearpath_remote_ws)
"""

import argparse
import logging
import time

import numpy as np
import rclpy

from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ridgeback_recordings",
        help="Repository ID for the LeRobot dataset",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="perform task",
        help="Task description label for all episodes",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Recording frame rate (default: 20)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes to record, 0 = unlimited (default: 50)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume recording from an existing dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Push dataset to Hugging Face Hub when done",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="ur10e_ridgeback_env",
        help="Environment config name (default: ur10e_ridgeback_env)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--no-robot-control",
        action="store_true",
        help="Skip env.home() / env.reset() / on_start / on_end. Use when driving the env "
        "from a rosbag (no live controller_manager service).",
    )
    parser.add_argument(
        "--auto-duration",
        type=float,
        default=None,
        help="If set, programmatically start recording, stop after N seconds (discarding the "
        "episode), and exit. Lets us measure achieved FPS without keyboard interaction.",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)

    logger.info(f"Creating environment: {args.env_config}")
    env = make_env(env_type=args.env_config, control_type="cartesian", namespace="")

    if args.no_robot_control:
        # Bag has no /controller_manager service, and DDS discovery for the
        # gripper/camera subscribers can take longer than the env's hardcoded 3 s.
        env._wait_for_controllers = lambda timeout=10.0: None

        _orig_wait = env.wait_until_ready

        def _patient_wait():
            # Replicate the env's wait_until_ready but with 15 s timeouts.
            env.robot.wait_until_ready(timeout=15)
            if env.gripper is not None:
                env.gripper.wait_until_ready(timeout=15)
            for cam in env.cameras:
                cam.wait_until_ready(timeout=15)
            for sensor in env.sensors:
                sensor.wait_until_ready(timeout=15)

        env.wait_until_ready = _patient_wait

    logger.info("Waiting for robot to be ready...")
    env.wait_until_ready()
    logger.info("Robot ready.")

    features = get_features(env=env)
    logger.debug(f"Features: {list(features.keys())}")

    rec_config = RecordingManagerConfig(
        features=features,
        repo_id=args.repo_id,
        robot_type="ur10e",
        fps=args.fps,
        num_episodes=args.num_episodes,
        resume=args.resume,
        push_to_hub=args.push_to_hub,
    )
    recording_manager = make_recording_manager(
        recording_manager_type="keyboard",
        config=rec_config,
    )
    recording_manager.wait_until_ready()
    logger.info("Recording manager ready.")

    if args.no_robot_control:
        logger.info("Skipping env.home() / env.reset() (--no-robot-control).")
    else:
        logger.info("Homing robot...")
        env.home()
        env.reset()

    def data_fn():
        obs = env.get_obs()

        # Action = what a policy must reproduce:
        #   - Cartesian: commanded target_pose (from mocap tracker via /target_pose)
        #   - Gripper:   commanded target      (from mocap tracker via /target_gripper_state)
        # Fall back to current gripper reading only before the tracker has
        # issued its first grip command.
        target_pose = env.robot.target_pose.to_array(
            representation=env.config.orientation_representation
        ).astype(np.float32)

        if env.gripper is not None and env.gripper._target is not None:
            grip_action = float(env.gripper.target)
        elif env.gripper is not None:
            grip_action = float(env.gripper.value)
        else:
            grip_action = 0.0

        action = np.concatenate(
            [target_pose, np.array([grip_action], dtype=np.float32)]
        )

        return obs, action

    if args.no_robot_control:
        on_start = None
        on_end = None
    else:
        def on_start():
            env.reset()

        def on_end():
            env.robot.reset_targets()
            env.home(blocking=False)
            env.gripper.open()

    # Optional auto-driver: replaces keyboard interaction with a fixed-duration
    # measurement. Counts every data_fn() call so we can report achieved Hz at end.
    frame_count = [0]
    if args.auto_duration is not None:
        import threading

        original_data_fn = data_fn

        def data_fn():  # noqa: F811
            frame_count[0] += 1
            return original_data_fn()

        def _auto_drive():
            time.sleep(2.0)  # let env settle + first messages arrive
            logger.info("[auto] starting recording")
            recording_manager.state = "recording"
            time.sleep(args.auto_duration)
            logger.info("[auto] stopping recording")
            recording_manager.state = "to_be_deleted"
            time.sleep(0.5)
            recording_manager.state = "exit"

        threading.Thread(target=_auto_drive, daemon=True).start()
        _t_record_start = time.time()

    try:
        with recording_manager:
            while not recording_manager.done():
                ep_num = recording_manager.episode_count + 1
                num_ep_str = str(args.num_episodes) if args.num_episodes > 0 else "∞"
                logger.info(f"Episode {ep_num} / {num_ep_str}")
                recording_manager.record_episode(
                    data_fn=data_fn,
                    task=args.task,
                    on_start=on_start,
                    on_end=on_end,
                )

        if not args.no_robot_control:
            logger.info("Recording complete. Homing robot.")
            env.home()

        if args.auto_duration is not None:
            elapsed = time.time() - _t_record_start
            n = frame_count[0]
            print()
            print("=" * 56)
            print(" Auto-mode result")
            print("=" * 56)
            print(f" target rate:  {args.fps} Hz")
            print(f" actual rate:  {n / elapsed:.2f} Hz   ({n} frames / {elapsed:.2f}s)")
            print("=" * 56)

    except Exception:
        logger.exception("Error during recording.")
        raise
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
