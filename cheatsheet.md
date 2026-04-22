# cheatsheet

Personal quick-ref for the ur10e_ridgeback + spacemouse + orbbec +
lerobot v0.5.1 workflow. Not for publishing.

## prereqs (every session)

- sim or real robot up, `cartesian_controller` active, `joint_state_broadcaster` + `pose_broadcaster` active
- orbbec: `pixi run orbbec` (in `clearpath_remote_ws`)
- spacemouse bridge running → owns `/target_pose` (for recording only, STOP it for replay)
- kill stale sims/scripts first: `ps aux | grep -E 'ros2|gazebo|sim'` then check — `pkill` success ≠ dead

## record

```bash
cd ~/repos/crisp_gym
pixi run -e jazzy-lerobot python examples/12_ridgeback_record.py --repo-id my_dataset
```

keys: `r` start/stop, `s` save, `d` discard, `q` quit.
flags: `--resume`, `--num-episodes N`.

## replay

```bash
pixi run -e jazzy-lerobot python examples/13_replay_episode.py \
    --repo-id my_dataset --episode-index 0
```

**spacemouse bridge MUST be stopped** — replay flips `publish_target_pose=True` so the Robot publishes; two publishers on `/target_pose` trips the controller's safety check.

## visualize dataset

```bash
pixi run -e jazzy-lerobot lerobot-dataset-visualize --repo-id my_dataset
```

datasets live at `~/.cache/huggingface/lerobot/<repo-id>/`.

## deploy policy

```bash
pixi run -e jazzy-lerobot python crisp_gym/scripts/deploy_policy.py \
    --repo-id deploy_run --path outputs/train/<run>/pretrained_model
```

- uses spawn (CUDA can't fork); env must reach `wait_until_ready()` before `make_policy()` — deploy_policy.py already orders this
- AsyncLerobotPolicy requires a policy with `_queues` (diffusion, smolvla). ACT will raise — use sync `lerobot_policy` instead.

## dataset conversions

```bash
pixi run -e jazzy-lerobot python crisp_gym/scripts/dataset_conversions/<script>.py
```

## gotchas hit today

- `CODEBASE_VERSION` is `v3.0` on v0.5.1 — feature-specs still work (verified)
- `add_frame(frame, task=...)` removed in v0.5 → put `task` inside the frame dict
- `push_to_hub(repo_id=..., private=True)` → drop `repo_id`, just `private=True`
- DDS cameras need warm-up before any CUDA child process spawns
- if pixi sticks on an old resolver state: `rm pixi.lock` and re-solve
