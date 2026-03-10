"""Filter out idle/zero-action frames from a LeRobot dataset and create a new one."""

import argparse
import contextlib
import os
import shutil
import sys

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Keys managed automatically by LeRobotDataset — must not appear in add_frame
AUTO_KEYS = {"index", "episode_index", "frame_index", "timestamp", "task_index"}


def is_idle_frame(
    action: np.ndarray,
    gripper_action_prev: float | None,
    gripper_state: float,
    gripper_state_prev: float | None,
    pos_thresh: float,
    gripper_thresh: float,
) -> bool:
    """Return True if the frame should be filtered out (position near-zero and gripper unchanged).

    Gripper is considered "changed" if either the gripper action or the gripper state
    differ by more than gripper_thresh from the previous timestep.
    """
    pos_idle = np.linalg.norm(action[:3]) <= pos_thresh

    if gripper_action_prev is None or gripper_state_prev is None:
        gripper_unchanged = True
    else:
        action_changed = abs(action[-1] - gripper_action_prev) > gripper_thresh
        state_changed = abs(gripper_state - gripper_state_prev) > gripper_thresh
        gripper_unchanged = not (action_changed or state_changed)

    return pos_idle and gripper_unchanged


def filter_dataset(
    source_repo_id: str,
    target_repo_id: str,
    pos_thresh: float = 1e-3,
    gripper_thresh: float = 5e-2,
    push: bool = False,
    private: bool = True,
):
    source = LeRobotDataset(repo_id=source_repo_id)

    # Remove old local cache for target if it exists so create() starts fresh
    from lerobot.constants import HF_LEROBOT_HOME
    target_root = HF_LEROBOT_HOME / target_repo_id
    if target_root.exists():
        shutil.rmtree(target_root)

    features = source.meta.features

    target = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source.meta.fps,
        robot_type=source.meta.robot_type,
        features=features,
        use_videos=True,
    )

    for ep_idx in range(source.meta.total_episodes):
        ep_start = source.episode_data_index["from"][ep_idx].item()
        ep_end = source.episode_data_index["to"][ep_idx].item()

        gripper_action_prev = None
        gripper_state_prev = None
        n_kept = 0

        for frame_idx in range(ep_start, ep_end):
            # Use __getitem__ to get decoded video frames + task string
            frame_data = source[frame_idx]
            action = frame_data["action"].numpy()
            gripper_state = float(frame_data["observation.state.gripper"].numpy())

            if is_idle_frame(action, gripper_action_prev, gripper_state, gripper_state_prev, pos_thresh, gripper_thresh):
                gripper_action_prev = float(action[-1])
                gripper_state_prev = gripper_state
                continue
            gripper_action_prev = float(action[-1])
            gripper_state_prev = gripper_state

            # Build frame dict for add_frame
            task = frame_data["task"]
            frame_dict = {}
            for key, value in frame_data.items():
                if key in AUTO_KEYS or key == "task":
                    continue
                if source.meta.features.get(key, {}).get("dtype") in ("image", "video"):
                    # CHW float32 [0,1] torch tensor -> HWC uint8 numpy
                    frame_dict[key] = (value.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    v = value.numpy() if hasattr(value, "numpy") else value
                    # Ensure arrays match expected shape (e.g. scalar -> (1,))
                    if isinstance(v, np.ndarray):
                        expected_shape = features.get(key, {}).get("shape")
                        if expected_shape is not None and v.shape != tuple(expected_shape):
                            v = v.reshape(expected_shape)
                    frame_dict[key] = v

            target.add_frame(frame_dict, task=task)
            n_kept += 1

        orig_len = ep_end - ep_start
        if n_kept > 0:
            # Suppress native SVT-AV1 encoder info spam on stderr
            devnull = os.open(os.devnull, os.O_WRONLY)
            stderr_fd = sys.stderr.fileno()
            saved_stderr = os.dup(stderr_fd)
            os.dup2(devnull, stderr_fd)
            os.close(devnull)
            try:
                target.save_episode()
            finally:
                os.dup2(saved_stderr, stderr_fd)
                os.close(saved_stderr)
            print(f"Episode {ep_idx}: {orig_len} -> {n_kept} frames")
        else:
            target.clear_episode_buffer()
            print(f"Episode {ep_idx}: all {orig_len} frames filtered out, skipping")

    print(f"\nFiltered dataset saved locally as '{target_repo_id}'")
    print(f"  Episodes: {source.meta.total_episodes} -> {target.meta.total_episodes}")
    print(f"  Frames:   {source.meta.total_frames} -> {target.meta.total_frames}")

    if push:
        target.push_to_hub(private=private)
        print(f"Pushed to hub: {target_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter idle frames from a LeRobot dataset")
    parser.add_argument("--source", type=str, required=True, help="Source dataset repo_id")
    parser.add_argument("--target", type=str, default=None, help="Target dataset repo_id (default: source + '_filtered')")
    parser.add_argument("--pos-thresh", type=float, default=1e-3, help="Position norm threshold for idle detection")
    parser.add_argument("--gripper-thresh", type=float, default=1e-2, help="Gripper change threshold for idle detection")
    parser.add_argument("--push", action="store_true", help="Push filtered dataset to HuggingFace Hub")
    args = parser.parse_args()

    target = args.target if args.target else args.source + "_filtered"
    filter_dataset(
        source_repo_id=args.source,
        target_repo_id=target,
        pos_thresh=args.pos_thresh,
        gripper_thresh=args.gripper_thresh,
        push=args.push,
        private=False,
    )
