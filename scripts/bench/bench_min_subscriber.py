"""Minimal subscriber bisection harness.

Pair with scripts/bench/bench_publisher.py. Measures callback fire rate of a
subscriber that grows feature-by-feature — starting at L0 (known-good raw
spin) and ending at L9 (full crisp_py.Camera behaviour).

Usage:
    # shell 1: publisher
    python scripts/bench/bench_publisher.py --rate 30

    # shell 2: subscriber
    python scripts/bench/bench_min_subscriber.py --level 0 --duration 15
    python scripts/bench/bench_min_subscriber.py --level 1 --duration 15
    ...
"""

import argparse
import threading
import time
from collections import deque
from functools import wraps

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from sensor_msgs.msg import CameraInfo, CompressedImage

IMG_TOPIC = "bench/camera/color/image_raw/compressed"
INFO_TOPIC = "bench/camera/color/camera_info"
TARGET_RES = (480, 640)  # (H, W) — matches env config


def _resize_with_aspect_ratio(image: np.ndarray, target_res: tuple) -> np.ndarray:
    """Copied verbatim from crisp_py.camera.camera._resize_with_aspect_ratio."""
    h, w = image.shape[:2]
    target_h, target_w = target_res
    if h == target_h and w == target_w:
        return image
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    return resized[start_y : start_y + target_h, start_x : start_x + target_w]


class State:
    """Mutable shared state for the subscriber."""

    def __init__(self):
        self.count = 0
        self.last_image = None
        self.cv_bridge = CvBridge()
        # L8-monitor state
        self.monitor_timestamps = deque(maxlen=50)
        self.monitor_callback_count = 0
        self.monitor_last_time = None


def run_level(level: int, duration: float,
              info_qos: str = "reliable",
              info_node: str = "same",
              executor_kind: str = "default") -> float:
    """Build and run the subscriber for `level`; return callback Hz.

    L7 variant flags (only affect level>=7):
      info_qos: "reliable" (default, qos_profile_system_default) or "besteffort"
      info_node: "same" (default, share main node) or "separate" (new Node)
      executor_kind: "default" (ladder), "single", or "multi"
    """
    rclpy.init()
    node = rclpy.create_node(f"bench_min_subscriber_l{level}")
    info_node_obj = None  # created only if info_node == "separate"
    state = State()

    # -------- Level-determined choices --------

    # L1+: sensor_data QoS; L0: default reliable depth=10
    qos = qos_profile_sensor_data if level >= 1 else 10

    # L6+: ReentrantCallbackGroup on image sub; else None (default MutEx)
    img_cb_group = ReentrantCallbackGroup() if level >= 6 else None

    # -------- Callback body (L2 adds decode, L3 adds resize) --------

    def image_cb(msg: CompressedImage):
        state.count += 1
        if level >= 2:
            img = np.asarray(
                state.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
            )
            if level >= 3:
                img = _resize_with_aspect_ratio(img, TARGET_RES)
            state.last_image = img

    # L8: wrap with monitor decorator
    if level >= 8:
        raw_cb = image_cb

        @wraps(raw_cb)
        def monitored(msg):
            now = time.time()
            state.monitor_timestamps.append(now)
            state.monitor_last_time = now
            state.monitor_callback_count += 1
            return raw_cb(msg)

        final_cb = monitored
    else:
        final_cb = image_cb

    # -------- Create image subscription --------
    node.create_subscription(
        CompressedImage,
        IMG_TOPIC,
        final_cb,
        qos,
        callback_group=img_cb_group,
    )

    # -------- L7+: second subscription (camera_info) --------
    if level >= 7:
        def info_cb(msg: CameraInfo):
            _ = (msg.height, msg.width)  # no-op, matches crisp_py

        info_qos_profile = (
            qos_profile_sensor_data if info_qos == "besteffort"
            else qos_profile_system_default
        )
        info_cb_group = ReentrantCallbackGroup()

        if info_node == "separate":
            info_node_obj = rclpy.create_node(f"bench_min_subscriber_info_l{level}")
            info_node_obj.create_subscription(
                CameraInfo, INFO_TOPIC, info_cb,
                info_qos_profile, callback_group=info_cb_group,
            )
        else:
            node.create_subscription(
                CameraInfo, INFO_TOPIC, info_cb,
                info_qos_profile, callback_group=info_cb_group,
            )

    # -------- L9: 1 Hz /diagnostics timer --------
    if level >= 9:
        diag_pub = node.create_publisher(DiagnosticArray, "/diagnostics", 10)

        def publish_diagnostics():
            if not state.monitor_timestamps:
                return
            ts = list(state.monitor_timestamps)
            time_span = ts[-1] - ts[0]
            freq = (len(ts) - 1) / time_span if time_span > 0 else 0.0
            mean_iv = 1.0 / freq if freq > 0 else 0.0
            intervals = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
            stddev = float(np.std(intervals)) if intervals else 0.0

            is_stale = (
                state.monitor_last_time is None
                or (time.time() - state.monitor_last_time) > 2.0
            )
            status = DiagnosticStatus(
                level=DiagnosticStatus.STALE if is_stale else DiagnosticStatus.OK,
                name="Bench Image Monitor",
                message="stale" if is_stale else "active",
                values=[
                    KeyValue(key="frequency_hz", value=f"{freq:.4f}"),
                    KeyValue(key="mean_interval_ms", value=f"{mean_iv * 1000:.1f}"),
                    KeyValue(key="interval_stddev_ms", value=f"{stddev * 1000:.1f}"),
                ],
            )
            arr = DiagnosticArray()
            arr.header.stamp = node.get_clock().now().to_msg()
            arr.status = [status]
            diag_pub.publish(arr)

        node.create_timer(1.0, publish_diagnostics)

    # -------- Spin strategy --------
    # L0-L3: main-thread spin_once loop (matches CLAUDE.md raw-spin baseline)
    # L4:    daemon-thread SingleThreadedExecutor.spin()
    # L5+:   daemon-thread MultiThreadedExecutor(num_threads=2).spin()

    t0 = time.perf_counter()
    deadline = t0 + duration

    if level <= 3 and executor_kind == "default":
        while time.perf_counter() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    else:
        if executor_kind == "single" or (executor_kind == "default" and level == 4):
            executor = SingleThreadedExecutor()
        else:
            executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        if info_node_obj is not None:
            executor.add_node(info_node_obj)
        threading.Thread(target=executor.spin, daemon=True).start()
        # Main thread sleeps; daemon does the work.
        while time.perf_counter() < deadline:
            time.sleep(0.1)

    elapsed = time.perf_counter() - t0
    hz = state.count / elapsed if elapsed > 0 else 0.0

    # Cleanup
    try:
        node.destroy_node()
    except Exception:
        pass
    if info_node_obj is not None:
        try:
            info_node_obj.destroy_node()
        except Exception:
            pass
    rclpy.shutdown()

    return hz, elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", type=int, required=True, help="0..9")
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--info-qos", choices=["reliable", "besteffort"], default="reliable")
    p.add_argument("--info-node", choices=["same", "separate"], default="same")
    p.add_argument("--executor", choices=["default", "single", "multi"], default="default")
    args = p.parse_args()

    hz, elapsed = run_level(args.level, args.duration,
                            info_qos=args.info_qos,
                            info_node=args.info_node,
                            executor_kind=args.executor)
    tag = f"L{args.level}"
    if args.info_qos != "reliable":
        tag += f"[iqos={args.info_qos}]"
    if args.info_node != "same":
        tag += f"[inode={args.info_node}]"
    if args.executor != "default":
        tag += f"[exec={args.executor}]"
    print(f"{tag}: {int(round(hz * elapsed))} cbs / {elapsed:.2f}s = {hz:.2f} Hz", flush=True)


if __name__ == "__main__":
    main()
