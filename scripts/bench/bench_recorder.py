"""Benchmark crisp_gym's recording loop against synthetic publisher topics.

Uses the real RecordingManager (so the patched _swap_images_for_handles +
SharedMemory ring + _materialize is actually exercised) but stubs the
LeRobot writer so the test isolates Python loop / queue throughput from
disk I/O and ffmpeg.

Pair with bench_publisher.py.

Run:
  python scripts/bench/bench_recorder.py --duration 30
  python scripts/bench/bench_recorder.py --duration 30 --no-shm   # A/B
"""

import argparse
import logging
import multiprocessing as mp
import threading
import time

import numpy as np
import rclpy

from crisp_gym.record.recording_manager import RecordingManager

try:
    from crisp_gym.record.recording_manager import _materialize_shm_frames
except ImportError:
    def _materialize_shm_frames(obs, _blocks):
        return obs
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_py.camera import Camera
from crisp_py.camera.camera_config import CameraConfig


class _StubDataset:
    """No-op stand-in for LeRobotDataset; we never call any of its methods."""


class BenchRecorder(RecordingManager):
    """Drains the queue, runs _materialize, counts frames; never writes to disk."""

    def __init__(self, *args, **kwargs):
        # mp.Value MUST be created before super().__init__ — that's where the
        # writer process forks and inherits this object via shared memory.
        self._frame_count = mp.Value("i", 0)
        super().__init__(*args, **kwargs)

    def get_instructions(self) -> str:
        return ""

    def _create_dataset(self):
        return _StubDataset()

    def _writer_proc(self):
        _ = self._create_dataset()
        self.dataset_ready.set()

        shm_blocks: dict = {}

        while True:
            msg = self.queue.get()
            mtype = msg["type"]
            if mtype == "FRAME":
                # Exercise the same SHM read path as the real writer; discard result.
                _materialize_shm_frames(msg["data"][0], shm_blocks)
                with self._frame_count.get_lock():
                    self._frame_count.value += 1
            elif mtype == "SHUTDOWN":
                for blk in shm_blocks.values():
                    blk.close()
                break
            # Other message types (SAVE_EPISODE, DELETE_EPISODE, PUSH_TO_HUB) are no-ops here.


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds")
    p.add_argument("--target-rate", type=int, default=30, help="Target recording FPS")
    p.add_argument(
        "--target-height",
        type=int,
        default=720,
        help="CameraConfig.resolution height (matches publisher source by default)",
    )
    p.add_argument("--target-width", type=int, default=1280, help="CameraConfig.resolution width")
    p.add_argument(
        "--image-topic",
        default="bench/camera/color/image_raw",
        help="Base image topic (Camera appends '/compressed'). "
        "For a real rosbag set e.g. 'camera/color/image_raw'.",
    )
    p.add_argument(
        "--info-topic",
        default="bench/camera/color/camera_info",
        help="Camera info topic.",
    )
    p.add_argument(
        "--no-shm",
        action="store_true",
        help="Bypass _swap_images_for_handles to A/B against the patch",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    rclpy.init()

    cam = Camera(
        namespace="",
        config=CameraConfig(
            camera_name="bench",
            camera_color_image_topic=args.image_topic,
            camera_color_info_topic=args.info_topic,
            resolution=(args.target_height, args.target_width),
        ),
    )
    print("Waiting for publisher...", flush=True)
    cam.wait_until_ready(timeout=15.0)
    print(
        f"Camera ready. CameraConfig.resolution={cam.config.resolution}  "
        f"actual cam.current_image.shape={cam.current_image.shape}",
        flush=True,
    )

    config = RecordingManagerConfig(
        features={
            "action": {"dtype": "float32", "shape": (7,), "names": None},
            "observation.images.bench": {
                "dtype": "uint8",
                "shape": (args.target_height, args.target_width, 3),
                "names": ["height", "width", "channels"],
            },
        },
        repo_id="bench/test",
        fps=args.target_rate,
        num_episodes=1,
        use_sound=False,
        queue_size=16,
    )

    manager = BenchRecorder(config=config)
    manager.wait_until_ready()

    if args.no_shm:
        manager._swap_images_for_handles = lambda obs: obs
        print("[A/B] SharedMemory swap DISABLED.", flush=True)

    def data_fn():
        return (
            {"observation.images.bench": cam.current_image},
            np.zeros(7, dtype=np.float32),
        )

    warn_count = [0]

    class _CountWarnings(logging.Handler):
        def emit(self, record):
            if "Frame processing took too long" in record.getMessage():
                warn_count[0] += 1

    logging.getLogger("crisp_gym.record.recording_manager").addHandler(_CountWarnings())

    def _stop_after(seconds: float):
        time.sleep(seconds)
        manager.state = "to_be_deleted"

    threading.Thread(target=_stop_after, args=(args.duration,), daemon=True).start()

    print(
        f"Recording for {args.duration:.1f}s @ target {args.target_rate} Hz "
        f"(resolution {args.target_height}x{args.target_width})...",
        flush=True,
    )
    manager.state = "recording"
    t0 = time.time()
    manager.record_episode(data_fn=data_fn, task="bench")
    elapsed = time.time() - t0

    manager.queue.put({"type": "SHUTDOWN"})
    manager.writer.join(timeout=10.0)
    if hasattr(manager, "_image_ring") and manager._image_ring is not None:
        manager._image_ring.cleanup()

    n = manager._frame_count.value
    rate = n / elapsed if elapsed > 0 else 0.0

    print()
    print("=" * 56)
    print(" Benchmark results")
    print("=" * 56)
    print(f" target rate:           {args.target_rate} Hz")
    print(f" actual rate:           {rate:.2f} Hz  ({n} frames / {elapsed:.2f}s)")
    print(f" overrun warnings:      {warn_count[0]}")
    print(f" target resolution:     {args.target_height}x{args.target_width}x3")
    print(f" SharedMemory swap:     {'disabled' if args.no_shm else 'enabled'}")
    print("=" * 56)

    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
