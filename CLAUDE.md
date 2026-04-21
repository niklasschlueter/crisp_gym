For the first question I ask always:
- Prepare a plan of potential changes
- DO NOT CHANGE ANYTHING EXCEPT SPECIFICALLY SPECIFIED
- Ask questions on how to proceed always

---

# Investigation: crisp_gym recording rate bottleneck (April 2026)

## Summary

The user records robot demonstrations at 20–30 Hz using `crisp_gym` + `crisp_py`
on a UR10e/Ridgeback with an Orbbec camera. On the recording laptop the rate
hit 20 Hz. On a workstation replaying a rosbag, the rate dropped to ~4 Hz.
**Fully fixed as of 2026-04-20** — the full recorder now hits the 30 Hz target
on FastRTPS (no middleware change needed).

## Setup

- **Robot**: UR10e on Clearpath Ridgeback, Robotiq 2F-85 gripper, Orbbec Femto Bolt camera
- **Recording script**: `examples/12_ridgeback_record.py` using `make_env("ur10e_ridgeback_env")`
- **Env config**: `crisp_gym/config/envs/ur10e_ridgeback_env.yaml` (640x480, 20 Hz, euler orientation)
- **Rosbag**: `rosbags/calib_20260416_022352/` — 147 s, 9 topics, camera at 30 Hz
- **Fork**: https://github.com/niklasschlueter/crisp_gym (forked from utiasDSL/crisp_gym)
- **Pixi env**: `jazzy-lerobot` (ROS2 jazzy + lerobot + crisp_py via editable path to `~/repos/crisp_py`)

## Current status: FIXED

| target | achieved | how |
|---|---|---|
| 30 Hz (recording window) | **~30 Hz** | single-exec crisp_py + `start_image_writer` |

Measured `examples/12_ridgeback_record.py --fps 30 --auto-duration 15 --no-robot-control`:

```
target rate: 30 Hz
actual rate: 25.46 Hz   (447 frames / 17.55s)   # averaged incl. 2s settle
```

During the 15 s recording window itself: 447 / 15 ≈ 29.8 Hz. `fresh_rate ==
actual_rate` (verified via instrumentation) — no silent duplicate frames.

## Root cause (two stacked bottlenecks)

### 1. rclpy `MultiThreadedExecutor` + `ReentrantCallbackGroup` on the same node

This is the primary cliff. `crisp_py.Camera` subscribes to both the image
topic and the camera_info topic on the same Node with `ReentrantCallbackGroup`
and uses `MultiThreadedExecutor(num_threads=2)`. Two callbacks on one node
trade the Python GIL + per-node wait-set locks instead of actually running
in parallel — net slower than one thread dispatching serially.

Proven via 10-level bisection (`scripts/bench/bench_min_subscriber.py`):
L0–L6 of a minimal subscriber hit 30 Hz; L7 (adds the second RELIABLE sub
on the same node) drops to ~6 Hz. Variants pinned the mechanism (320×240,
FastRTPS):

| L7 variant | Hz | verdict |
|---|---|---|
| default (reliable info sub, same node, MultiExec(2)) | 1.96 | cliff |
| `--info-qos besteffort` | 4.21 | reliability is not the cause |
| `--info-node separate` | 20.61 | same-node partially contributes |
| **`--executor single`** | **30.07** | **fixes it fully** |

**Fix**: set `THREADS_REQUIRED = 1` on Camera, Robot, Gripper, Sensor; make
`_spin_node` use `SingleThreadedExecutor` when `THREADS_REQUIRED <= 1`; also
replace `while rclpy.ok(): executor.spin_once(0.1)` with `executor.spin()`
(the old pattern had up to 100 ms idle between callbacks).

### 2. Synchronous PIL PNG encoding in the writer process

Once the executor fix is in place, the recorder climbs to ~22 Hz and
plateaus. Per-frame timing attribution (instrumented `_writer_proc` and
monkey-patched `dataset._save_image`):

```
add_frame = 39.22 ms/frame
  of which save_image = 39.17 ms/frame (99.9%)
unpack    = 0.001 ms/frame
```

99% of the writer's per-frame cost is PIL-encoding a 640×480×3 PNG (`zlib
DEFLATE` on a 921 KB buffer, ≈30–40 ms per image on one CPU core). At the
30 Hz publisher period of 33 ms, a single-core PNG encoder can never keep
up, the size-16 mp queue fills, and `queue.put` in the recording loop blocks
for the producer–writer rate difference.

**Fix**: call `dataset.start_image_writer(num_processes=4, num_threads=2)`
in `_writer_proc` right after dataset creation. This installs an
`AsyncImageWriter` that spawns 4 daemon worker processes; `_save_image`
then drops into `queue.put((image, fpath))` and returns in microseconds.
Encoding runs on 4 cores in parallel, amortizing the per-frame cost to
~10 ms. Must pair with `dataset.stop_image_writer()` on SHUTDOWN — otherwise
the daemon pool workers block on `queue.get` and `writer.join()` in
`RecordingManager.__exit__` hangs.

## Applied fixes

### In `~/repos/crisp_py/` (editable install via `crisp_gym/pixi.toml`)

- `crisp_py/camera/camera.py` — `THREADS_REQUIRED: 2 → 1`; `_spin_node` uses
  conditional `SingleThreadedExecutor` (falls back to `MultiThreadedExecutor`
  if `THREADS_REQUIRED > 1`); `while: spin_once(0.1)` → `executor.spin()`
- `crisp_py/robot/robot.py` — `THREADS_REQUIRED: 4 → 1`; same executor pattern
- `crisp_py/gripper/gripper.py` — `THREADS_REQUIRED: 2 → 1`; added
  `SingleThreadedExecutor` import; same executor pattern
- `crisp_py/sensors/sensor.py` — `THREADS_REQUIRED: 2 → 1`; `spin_once(0.1)`
  loop → `executor.spin()` (conditional already in place)

### In this repo

- `pixi.toml` — uncommented
  `crisp_python = { path = "../crisp_py", editable = true }` so the above
  changes survive `pixi install`
- `crisp_gym/record/recording_manager.py:_writer_proc` — call
  `dataset.start_image_writer(num_processes=4, num_threads=2)` on startup
  and `dataset.stop_image_writer()` in the SHUTDOWN handler
- `scripts/bench/bench_min_subscriber.py` — new bisection harness that
  proved the mechanism (L0–L9 with `--info-qos`, `--info-node`, `--executor`
  flags)

## What was investigated and ruled out

### 1. `queue.put` pickle cost — NOT the bottleneck

**Hypothesis**: pickling large image arrays through `mp.JoinableQueue` is slow.
**Result**: SHM ring buffer microbench is 14× faster than pickle at 720p,
but at 640×480 pickle is already <1 ms. Recording rate was identical with
and without the SHM patch (~4 Hz both ways). Pickle is not the bottleneck
for this workload. (Post-fix reconfirmed: `queue.put = 0.02 ms`.) SHM patch
reverted; still in git history as commit `8e1f5a5 shared memory patch`.

### 2. FastDDS XML profiles — NOT the bottleneck for the recorder

Multiple variants tested (SHM-only, no-multicast, BEST_EFFORT reader
overrides, LARGE_DATA builtin transport, Pleune's SHM+data_sharing config
from ros2/ros2#1289). At 640×480, all variants gave 4–6 Hz under FastRTPS —
no meaningful difference. Pleune's XML does help the **camera path** at
large image sizes (fixes UDP fragmentation), but **hurts** the full
recorder's mixed-rate topic set (500 Hz joint_states + 360 Hz tf + others):
21.7 → 8.4 Hz under Pleune's XML. Don't enable XML for the recorder.
XML files remain in `scripts/` for reference.

### 3. Kernel UDP buffer limits — NOT the bottleneck

Raised `net.core.rmem_max` to 8 MB via `sysctl`. No change.

### 4. QoS mismatch (RELIABLE bag vs BEST_EFFORT subscriber) — NOT the bottleneck

Overrode bag playback QoS via `scripts/bag_qos_override.yaml`. No change.
L7 variant with `info_qos=besteffort` barely helps either.

### 5. FastRTPS middleware itself — NOT the cause

Earlier framing ("FastRTPS's internal threading interacts badly with the
crisp_py executor model") was misleading. With `SingleThreadedExecutor`,
FastRTPS delivers camera callbacks at 30+ Hz on this workstation. The
previous narrative conflated the symptom (slow recorder) with the mechanism
(rclpy multi-exec contention).

### 6. CycloneDDS workaround — NOT available for production

CycloneDDS hits 18.4 Hz with zero code changes but **cannot be used in
production**: the robot uses FastRTPS Discovery Server, a vendor-specific
feature. A CycloneDDS subscriber cannot discover topics published by
participants registered with the Discovery Server. This forced us to find a
FastRTPS-compatible fix.

## Key numbers for reference

| scenario | RMW | rate |
|---|---|---|
| Pre-fix baseline | FastRTPS | 5.55 Hz |
| + single-exec crisp_py (no XML) | FastRTPS | 21.7 Hz (writer-bound) |
| + `start_image_writer(num_processes=4, num_threads=2)` | FastRTPS | **~30 Hz ✓** |
| bench L6 (crisp_py.Camera analog without 2nd sub) | FastRTPS | 30 Hz |
| bench L7 (full Camera analog) with MultiExec | FastRTPS | 1.96 Hz |
| bench L7 with SingleThreadedExecutor | FastRTPS | 30.07 Hz |
| Full recorder, pre-fix, CycloneDDS | CycloneDDS | 18.4 Hz |
| Writer `_save_image` per 640×480 image | — | 39 ms |
| Writer `_save_image` with 4-process pool (amortized) | — | ~10 ms |
| pickle vs SHM queue (640×480 micro) | — | both <1 ms (neither is the cap) |

## Files added/modified in this repo

| file | status | description |
|---|---|---|
| `examples/12_ridgeback_record.py` | added | User's recording script (from ChazzKemal/ur10_clearpath), with `--no-robot-control` and `--auto-duration` flags for bag testing |
| `crisp_gym/config/envs/ur10e_ridgeback_env.yaml` | added | UR10e env config |
| `crisp_gym/record/recording_manager.py` | modified | `_writer_proc` now calls `start_image_writer(4, 2)` and `stop_image_writer()` on SHUTDOWN. SHM patch was applied then reverted (not related to current fix). |
| `scripts/bench/bench_publisher.py` | added | Synthetic ROS2 publisher for testing |
| `scripts/bench/bench_min_subscriber.py` | added | 10-level bisection harness that proved the L7 mechanism (`--info-qos`, `--info-node`, `--executor` flags) |
| `scripts/bench/bench_recorder.py` | added | Recording benchmark (stubs LeRobot writer) |
| `scripts/bench/bench_microqueue.py` | added | Pure pickle-vs-SHM microbenchmark |
| `scripts/bench/run_all.py` | added | Automated 4-case A/B test runner |
| `scripts/fastdds_*.xml` | added | FastDDS XML variants, kept for reference; none enabled in production |
| `scripts/fastdds_L7_fix.xml` | added | Pleune's SHM+data_sharing XML; helps camera path at 1280×720 but hurts full recorder; not used in production |
| `scripts/bag_qos_override.yaml` | added | BEST_EFFORT QoS override for ros2 bag play (unused — QoS isn't the cause) |
| `pixi.toml` | modified | `crisp_python = { path = "../crisp_py", editable = true }` points at the local crisp_py clone (was previously installed via pinned git rev) |

## Follow-ups

- Upstream a PR to `utiasDSL/crisp_py` with the `THREADS_REQUIRED=1` +
  conditional `SingleThreadedExecutor` + `executor.spin()` changes.
- Upstream a PR to `utiasDSL/crisp_gym` with the `start_image_writer` /
  `stop_image_writer` lifecycle in `RecordingManager._writer_proc`.
- If either of the above isn't upstreamed, users of this repo must keep the
  editable path to `../crisp_py` in `pixi.toml` — otherwise `pixi install`
  pulls the unpatched crisp_py and the bottleneck returns silently.
- Bump lerobot from the current pin (git rev `dacd1d7f...`, pyproject says
  `0.2.0` but no such release exists — it's a July 2025 dev snapshot) to a
  published release. Easiest: `v0.3.2` — drop-in, same API (`start_image_writer`
  / `stop_image_writer` on the dataset), no source changes. For `v0.5.1` the
  writer API moved to `dataset.writer.start_image_writer(...)` and the
  idiomatic path is `LeRobotDataset.create(..., image_writer_processes=4,
  image_writer_threads=2)` — requires a small refactor in
  `recording_manager.py` + verifying other crisp_gym imports still work.
