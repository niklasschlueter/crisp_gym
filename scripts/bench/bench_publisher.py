"""Synthetic ROS2 publisher mimicking the topics crisp_py expects, at a fixed rate.

Pair with bench_recorder.py to measure end-to-end recording throughput
without real hardware.

Topics (all under the bench/ namespace):
  bench/camera/color/image_raw/compressed  (sensor_msgs/CompressedImage)
  bench/camera/color/camera_info           (sensor_msgs/CameraInfo)
  bench/joint_states                       (sensor_msgs/JointState)
  bench/current_pose                       (geometry_msgs/PoseStamped)
  bench/gripper/joint_states               (sensor_msgs/JointState)

Run:
  python scripts/bench/bench_publisher.py --rate 30 --width 1280 --height 720
"""

import argparse

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState

UR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class BenchPublisher(Node):
    def __init__(self, rate_hz: float, width: int, height: int, jpeg_quality: int):
        super().__init__("bench_crisp_publisher")
        self._rate_hz = rate_hz
        self._width = width
        self._height = height
        self._jpeg_quality = int(jpeg_quality)
        self._frame_idx = 0

        self._image_pub = self.create_publisher(
            CompressedImage,
            "bench/camera/color/image_raw/compressed",
            qos_profile_sensor_data,
        )
        self._info_pub = self.create_publisher(
            CameraInfo,
            "bench/camera/color/camera_info",
            qos_profile_system_default,
        )
        self._joint_pub = self.create_publisher(
            JointState,
            "bench/joint_states",
            qos_profile_system_default,
        )
        self._pose_pub = self.create_publisher(
            PoseStamped,
            "bench/current_pose",
            qos_profile_system_default,
        )
        self._gripper_pub = self.create_publisher(
            JointState,
            "bench/gripper/joint_states",
            qos_profile_system_default,
        )

        self._cam_info = CameraInfo()
        self._cam_info.width = width
        self._cam_info.height = height

        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"Publishing at {rate_hz} Hz, image {height}x{width} JPEG q={jpeg_quality}"
        )

    def _tick(self):
        now = self.get_clock().now().to_msg()

        # Fresh random pixels each tick so the JPEG payload size is realistic
        # (uniform images compress to tiny streams that don't stress the decoder).
        img = np.random.randint(0, 256, (self._height, self._width, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            self.get_logger().error("JPEG encode failed")
            return
        img_msg = CompressedImage()
        img_msg.header.stamp = now
        img_msg.header.frame_id = "bench_camera"
        img_msg.format = "jpeg"
        img_msg.data = buf.tobytes()
        self._image_pub.publish(img_msg)

        self._cam_info.header.stamp = now
        self._info_pub.publish(self._cam_info)

        t = self._frame_idx / self._rate_hz

        joint = JointState()
        joint.header.stamp = now
        joint.name = UR_JOINT_NAMES
        joint.position = [0.1 * float(np.sin(t + i)) for i in range(6)]
        joint.velocity = [0.0] * 6
        joint.effort = [0.0] * 6
        self._joint_pub.publish(joint)

        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = "base_link"
        pose.pose.position.x = 0.4 + 0.05 * float(np.sin(t))
        pose.pose.position.z = 0.4
        pose.pose.orientation.w = 1.0
        self._pose_pub.publish(pose)

        gripper = JointState()
        gripper.header.stamp = now
        gripper.name = ["robotiq_85_left_knuckle_joint"]
        gripper.position = [0.4 + 0.4 * float(np.sin(t))]
        self._gripper_pub.publish(gripper)

        self._frame_idx += 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rate", type=float, default=30.0, help="Publish rate Hz (default 30)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--jpeg-quality", type=int, default=80)
    args = p.parse_args()

    rclpy.init()
    node = BenchPublisher(args.rate, args.width, args.height, args.jpeg_quality)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
