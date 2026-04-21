#!/usr/bin/env python3
"""Replay a recorded LeRobot episode through the env at the recorded fps.

Feeds each frame's action vector [x, y, z, roll, pitch, yaw, gripper]
directly to env.step(). Verifies that dataset + env + Robot + controller
stack can reproduce the recorded motion.

The env YAML has publish_target_pose=False (so an external spacemouse
bridge owns /target_pose during recording). Replay overrides that in
memory so the Robot itself publishes — otherwise cartesian_controller
would never see our commanded pose. As a result, any running spacemouse
must be stopped before starting the replay, otherwise the cartesian
controller's two-publisher safety check will reject everything.

Usage (sim running, spacemouse STOPPED):
    pixi run -e jazzy-lerobot python examples/13_replay_episode.py \\
        --repo-id _sim_first_11 --episode-index 0
"""

import argparse
import logging
import os
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.envs.manipulator_env_config import make_env_config
from crisp_gym.util.setup_logger import setup_logging


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--repo-id", required=True, help="LeRobot dataset repo_id (folder under ~/.cache/huggingface/lerobot)")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--env-config", type=str, default="ur10e_ridgeback_env")
    p.add_argument(
        "--fps", type=float, default=None,
        help="override playback rate (default: dataset's recorded fps)",
    )
    p.add_argument("--no-home-before", action="store_true")
    p.add_argument("--no-home-after", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args()

    setup_logging(level=args.log_level)
    log = logging.getLogger(__name__)

    log.warning(
        "Stop any running spacemouse publisher before replay — this script "
        "makes Robot own /target_pose, so a running spacemouse creates a "
        "two-publisher conflict that cartesian_controller rejects."
    )

    # Replay-scoped override: Robot must own /target_pose so cartesian_controller
    # actually sees the commanded poses. Record-mode YAML keeps publish_target_pose=False
    # so the external spacemouse owns it; that's wrong for replay.
    env_config = make_env_config(args.env_config)
    env_config.robot_config.publish_target_pose = True

    log.info(f"Creating env: {args.env_config} (publish_target_pose=True override)")
    env = ManipulatorCartesianEnv(config=env_config, namespace="")
    env.wait_until_ready()

    ds_root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo_id}")
    log.info(f"Loading dataset from {ds_root}")
    ds = LeRobotDataset(args.repo_id, root=ds_root)

    if args.episode_index >= ds.meta.total_episodes:
        raise SystemExit(
            f"episode-index {args.episode_index} out of range (dataset has "
            f"{ds.meta.total_episodes} episode(s))"
        )

    ep = ds.meta.episodes[args.episode_index]
    lo = int(ep["dataset_from_index"])
    hi = int(ep["dataset_to_index"])
    fps = args.fps or ds.meta.fps
    period = 1.0 / fps
    n = hi - lo
    log.info(f"Replaying episode {args.episode_index}: {n} frames at {fps} Hz")

    if not args.no_home_before:
        log.info("Homing robot before replay...")
        env.home(blocking=True)
    env.reset()

    t0 = time.time()
    overruns = 0
    for i in range(lo, hi):
        t_frame = time.time()
        action = ds[i]["action"].numpy()  # [7] float32
        env.step(action, block=False)
        slack = period - (time.time() - t_frame)
        if slack > 0:
            time.sleep(slack)
        else:
            overruns += 1
            log.debug(f"frame {i - lo} overran by {-slack * 1000:.1f} ms")

    wall = time.time() - t0
    achieved_hz = n / wall if wall > 0 else 0.0
    log.info(
        f"Replay done: {n} frames in {wall:.2f}s → {achieved_hz:.2f} Hz "
        f"(target {fps} Hz, overruns={overruns})"
    )

    if not args.no_home_after:
        log.info("Homing robot after replay...")
        env.home(blocking=True)

    env.close()


if __name__ == "__main__":
    main()
