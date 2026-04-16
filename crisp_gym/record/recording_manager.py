"""Keyboard event listener for controlling episode recording."""

import logging
import multiprocessing as mp
import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import signature
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import rclpy

# TODO: make this optional, we do not want to depend on lerobot
try:
    from lerobot.utils.constants import HF_LEROBOT_HOME
except ImportError:
    from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pynput import keyboard
from rclpy.executors import SingleThreadedExecutor
from rich import print
from rich.panel import Panel
from std_msgs.msg import String
from typing_extensions import override

from crisp_gym.config.path import find_config
from crisp_gym.policy.policy import Action, Observation
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_gym.util.lerobot_features import concatenate_state_features

logger = logging.getLogger(__name__)

_ADD_FRAME_HAS_TASK = "task" in signature(LeRobotDataset.add_frame).parameters


@dataclass
class _SharedFrame:
    """Pickle-cheap handle pointing at an image stored in a SharedMemory block."""

    block_name: str
    shape: tuple
    dtype: str


class _SharedImageRing:
    """Ring buffer of SharedMemory blocks for zero-copy image transfer.

    Producer copies pixels into the next slot and emits a `_SharedFrame`.
    Consumer reconstructs a numpy view from the named block and copies it out.
    Ring size must exceed the queue depth so slots are never overwritten in flight.
    """

    def __init__(self, ring_size: int, max_image_bytes: int, name_prefix: str):
        self.blocks = [
            shared_memory.SharedMemory(
                create=True, size=max_image_bytes, name=f"{name_prefix}_{i}"
            )
            for i in range(ring_size)
        ]
        self._cursor = 0
        self._ring_size = ring_size

    def claim_slot(self, arr: np.ndarray) -> _SharedFrame:
        block = self.blocks[self._cursor]
        assert arr.nbytes <= block.size, (
            f"Image of {arr.nbytes} bytes exceeds slot size {block.size}; "
            "increase max_image_bytes or shrink the image."
        )
        np.ndarray(arr.shape, dtype=arr.dtype, buffer=block.buf)[:] = arr
        handle = _SharedFrame(
            block_name=block.name, shape=tuple(arr.shape), dtype=str(arr.dtype)
        )
        self._cursor = (self._cursor + 1) % self._ring_size
        return handle

    def cleanup(self) -> None:
        for block in self.blocks:
            try:
                block.close()
                block.unlink()
            except FileNotFoundError:
                pass


def _materialize_shm_frames(
    obs: dict, shm_blocks: dict[str, shared_memory.SharedMemory]
) -> dict:
    """Replace _SharedFrame handles in `obs` with copied numpy arrays.

    The caller owns `shm_blocks` so each block is opened at most once across
    many calls. Returned arrays are copies so the producer is free to
    overwrite the slot on its next claim.
    """
    out = dict(obs)
    for k, v in obs.items():
        if isinstance(v, _SharedFrame):
            if v.block_name not in shm_blocks:
                shm_blocks[v.block_name] = shared_memory.SharedMemory(name=v.block_name)
            out[k] = np.ndarray(
                v.shape, dtype=np.dtype(v.dtype), buffer=shm_blocks[v.block_name].buf
            ).copy()
    return out


class RecordingManager(ABC):
    """Base class for event listener to control episode recording."""

    def __init__(
        self,
        config: RecordingManagerConfig | None = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the recording manager.

        Args:
            config: RecordingManagerConfig instance. If provided, other parameters are ignored.
            **kwargs: Individual parameters for backwards compatibility.
        """
        # Handle config vs individual parameters
        self.config = (
            config
            if config is not None
            else RecordingManagerConfig.from_yaml(
                find_config("recording/default_recording.yaml"), **kwargs
            )
        )

        self.state: Literal[
            "is_waiting",
            "recording",
            "paused",
            "to_be_saved",
            "to_be_deleted",
            "exit",
        ] = "is_waiting"

        self.episode_count = 0

        self.queue = mp.JoinableQueue(self.config.queue_size)
        self.episode_count_queue = mp.Queue(1)
        self.dataset_ready = mp.Event()

        self._image_ring: _SharedImageRing | None = None

        # Start the writer process
        self.writer = mp.Process(
            target=self._writer_proc,
            args=(),
            name="dataset_writer",
            daemon=False,
        )
        self.writer.start()

    @property
    def dataset_directory(self) -> Path:
        """Return the path to the dataset directory."""
        return Path(HF_LEROBOT_HOME / self.config.repo_id)

    @property
    def num_episodes(self) -> int:
        """Return the number of episodes to record."""
        return self.config.num_episodes

    def wait_until_ready(self, timeout: float | None = None) -> None:
        """Wait until the dataset writer is ready."""
        if timeout is None:
            timeout = self.config.writer_timeout

        original_timeout = timeout
        while not self.dataset_ready.is_set():
            logger.debug("Waiting for dataset to be ready...")
            time.sleep(1.0)
            timeout -= 1.0
            if timeout <= 0.0:
                raise TimeoutError(
                    f"Timeout waiting for dataset to be ready after {original_timeout} seconds."
                )

        self.update_episode_count()

    def update_episode_count(self) -> None:
        """Update the episode count from the queue.

        This is useful when resuming from an existing dataset.
        If the queue is empty, it will not change the episode count.
        """
        if not self.episode_count_queue.empty():
            self.episode_count = self.episode_count_queue.get()

    def done(self) -> bool:
        """Return true if we are done recording."""
        return self.state == "exit"

    @abstractmethod
    def get_instructions(self) -> str:
        """Return the instructions to use the recording manager."""
        raise NotImplementedError()

    def _create_dataset(self) -> LeRobotDataset:
        """Factory function to create a dataset object."""
        logger.debug("Creating dataset object.")
        if self.config.resume:
            logger.info(f"Resuming recording from existing dataset: {self.config.repo_id}")
            dataset = LeRobotDataset(repo_id=self.config.repo_id)
            if self.config.num_episodes <= dataset.num_episodes:
                logger.error(
                    f"The dataset already has {dataset.num_episodes} recorded. Please select a larger number."
                )
                exit()
            logger.info(
                f"Resuming from episode {dataset.num_episodes} with {self.config.num_episodes} episodes to record."
            )
            self.episode_count_queue.put(dataset.num_episodes - 1)
        else:
            logger.info(
                f"[green]Creating new dataset: {self.config.repo_id}", extra={"markup": True}
            )
            # Clean up existing dataset if it exists
            if Path(HF_LEROBOT_HOME / self.config.repo_id).exists():
                logger.error(
                    f"The repo_id already exists. If you intended to resume the collection of data, then execute this script with the --resume flag. Otherwise remove it:\n'rm -r {str(Path(HF_LEROBOT_HOME / self.config.repo_id))}'."
                )
                raise FileExistsError(
                    f"The repo_id already exists. If you intended to resume the collection of data, then execute this script with the --resume flag. Otherwise remove it:\n'rm -r {str(Path(HF_LEROBOT_HOME / self.config.repo_id))}'."
                )
            dataset = LeRobotDataset.create(
                repo_id=self.config.repo_id,
                fps=self.config.fps,
                robot_type=self.config.robot_type,
                features=self.config.features,
                use_videos=True,
            )
            logger.debug(f"Dataset created with meta: {dataset.meta}")
        return dataset

    def _writer_proc(self):
        """Process to write data to the dataset."""
        logger.info("Starting dataset writer process.")
        dataset = self._create_dataset()
        self.dataset_ready.set()
        logger.debug(f"Dataset features: {list(self.config.features.keys())}")

        shm_blocks: dict[str, shared_memory.SharedMemory] = {}

        while True:
            msg = self.queue.get()
            logger.debug(f"Received message: {msg['type']}")
            try:
                mtype = msg["type"]

                if mtype == "FRAME":
                    obs, action, task = msg["data"]
                    obs = _materialize_shm_frames(obs, shm_blocks)

                    logger.debug(f"Received frame with action: {action} and obs: {obs.keys()}")

                    # Build frame directly from observation using feature-based approach
                    frame = {"action": action.astype(np.float32)}

                    # Add all observation features that match our dataset features
                    for feature_name in self.config.features:
                        if feature_name == "action":
                            continue  # Already added above
                        if feature_name in obs:
                            value = obs[feature_name]
                            if isinstance(value, np.ndarray) and feature_name.startswith(
                                "observation.state"
                            ):
                                frame[feature_name] = value.astype(np.float32)
                            else:
                                frame[feature_name] = value

                    # Concatenate state vector
                    frame["observation.state"] = concatenate_state_features(
                        obs, self.config.features
                    )

                    logger.debug(f"Constructed frame with keys: {frame.keys()}")
                    if _ADD_FRAME_HAS_TASK:  # For lerobot versions with explicit `task` parameter (>= v3.0)
                        dataset.add_frame(frame, task=task)
                    else:  # For older lerobot versions without `task` parameter (< v3.0)
                        frame["task"] = task
                        dataset.add_frame(frame)

                elif mtype == "SAVE_EPISODE":
                    if self.config.use_sound:
                        try:
                            subprocess.Popen(
                                [
                                    "paplay",
                                    "/usr/share/sounds/freedesktop/stereo/complete.oga ",
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to play sound for episode completion: {e}",
                            )

                    logger.info("Saving current episode to dataset.")

                    dataset.save_episode()

                elif mtype == "DELETE_EPISODE":
                    if self.config.use_sound:
                        try:
                            subprocess.Popen(
                                [
                                    "paplay",
                                    "/usr/share/sounds/freedesktop/stereo/suspend-error.oga",
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to play sound for episode deletion: {e}",
                            )

                    dataset.clear_episode_buffer()

                elif mtype == "PUSH_TO_HUB":
                    logger.info(
                        "Pushing dataset to Hugging Face Hub...",
                    )
                    try:
                        dataset.push_to_hub(repo_id=self.config.repo_id, private=True)
                        logger.info("Dataset pushed to Hugging Face Hub successfully.")
                    except Exception as e:
                        logger.error(
                            f"Failed to push dataset to Hugging Face Hub: {e}",
                            exc_info=True,
                        )
                elif mtype == "SHUTDOWN":
                    for blk in shm_blocks.values():
                        blk.close()
                    logger.info("Shutting down writer process.")
                    break
            except Exception as e:
                logger.exception("Error occured: ", e)
            finally:
                pass

        self.queue.task_done()
        logger.info("Writter process finished.")

    def _swap_images_for_handles(self, obs: dict) -> dict:
        """Replace numpy image arrays in `obs` with cheap _SharedFrame handles.

        Pixels are copied once into a pre-allocated SharedMemory ring slot, so
        the queued payload becomes ~1 KB instead of multiple MB and pickling
        cost across the multiprocessing queue drops to microseconds.
        """
        image_keys = [
            k for k, v in obs.items()
            if k.startswith("observation.images.") and isinstance(v, np.ndarray)
        ]
        if not image_keys:
            return obs

        if self._image_ring is None:
            max_bytes = max(obs[k].nbytes for k in image_keys)
            ring_size = self.config.queue_size + 4
            self._image_ring = _SharedImageRing(
                ring_size=ring_size,
                max_image_bytes=max_bytes,
                name_prefix=f"crisp_record_{os.getpid()}",
            )

        swapped = dict(obs)
        for k in image_keys:
            swapped[k] = self._image_ring.claim_slot(obs[k])
        return swapped

    def record_episode(
        self,
        data_fn: Callable[[], tuple[Observation, Action]],
        task: str,
        on_start: Callable[[], None] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> None:
        """Record a single episode from user-provided data function.

        Args:
            data_fn: A function that returns (obs, action) at each step.
            task: The task label for the episode.
            on_start: Optional hook called at the start of the episode.
            on_end: Optional hook called at the end (before save/delete).
        """
        try:
            self._wait_for_start_signal()
        except StopIteration:
            logger.info("Recording manager is shutting down.")
            return

        if on_start:
            logger.info("Resetting Environment.")
            on_start()

        logger.info("Started recording episode.")

        while self.state == "recording":
            frame_start = time.time()

            obs, action = data_fn()

            if obs is None or action is None:
                logger.debug("Data function returned None, skipping frame.")
                # If the data function returns None, skip this frame
                sleep_time = 1 / self.config.fps - (time.time() - frame_start)
                time.sleep(sleep_time)
                continue

            obs = self._swap_images_for_handles(obs)
            self.queue.put({"type": "FRAME", "data": (obs, action, task)})

            sleep_time = 1 / self.config.fps - (time.time() - frame_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(
                    f"Frame processing took too long: {time.time() - frame_start - 1.0 / self.config.fps:.3f} seconds too long i.e. {1.0 / (time.time() - frame_start):.2f} FPS. "
                    "Consider decreasing the FPS or optimizing the data function."
                )
            logger.debug(f"Finished sleeping for {sleep_time:.3f} seconds.")

        logger.debug("Finished recording...")

        if on_end:
            on_end()

        self._handle_post_episode()

    def _wait_for_start_signal(self) -> None:
        """Wait until the recording state is set to 'recording'."""
        logger.info("Waiting to start recording...")
        while self.state != "recording":
            if self.state == "exit":
                raise StopIteration
            time.sleep(0.05)

    def _handle_post_episode(self) -> None:
        """Handle the state after recording an episode."""
        if self.state == "paused":
            logger.info("Paused. Awaiting user decision to save/delete...")
            while self.state == "paused":
                time.sleep(0.5)

        if self.state == "to_be_saved":
            logger.info("Saving current episode.")
            self.queue.put({"type": "SAVE_EPISODE"})
            self.episode_count += 1
            self._set_to_wait()
        elif self.state == "to_be_deleted":
            logger.info("Deleting current episode.")
            self.queue.put({"type": "DELETE_EPISODE"})
            self._set_to_wait()
        elif self.state == "exit":
            pass
        else:
            logger.warning(f"Unexpected state after recording: {self.state}")

    def __enter__(self) -> "RecordingManager":  # noqa: D105
        """Enter the recording manager context."""
        print(Panel(self.get_instructions()))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001, D105
        """Exit the recording manager."""
        if exc_type is not None:
            logger.error(
                "An error occurred during recording. Shutting down the recording manager.",
                exc_info=(exc_type, exc_value, traceback),
            )

        if not self.config.push_to_hub:
            logger.info("Not pushing dataset to Hugging Face Hub.")
        else:
            self.queue.put({"type": "PUSH_TO_HUB"})
        logger.info("Shutting down the record process...")
        self.queue.put({"type": "SHUTDOWN"})

        self.writer.join()

        if self._image_ring is not None:
            self._image_ring.cleanup()
            self._image_ring = None

    def _set_to_wait(self) -> None:
        """Set to wait if possible."""
        if self.state not in ["to_be_saved", "to_be_deleted"]:
            raise ValueError("Can not go to wait state if the state is not to be saved or deleted!")
        if self.episode_count >= self.config.num_episodes:
            self.state = "exit"
        else:
            self.state = "is_waiting"


class ROSRecordingManager(RecordingManager):
    """ROS-based recording manager for controlling episode recording."""

    def __init__(self, config: RecordingManagerConfig | None = None, **kwargs) -> None:  # noqa: ANN003
        """Initialize ROS recording manager.

        Args:
            config: RecordingManagerConfig instance. If provided, **kwargs are ignored except for backwards compatibility.
            **kwargs: Individual parameters for backwards compatibility.
        """
        super().__init__(config=config, **kwargs)
        if not rclpy.ok():
            raise RuntimeError(
                "ROS2 is not initialized. Please initialize ROS2 before using the RecordingManager."
            )
        self.allowed_actions = ["record", "save", "delete", "exit"]
        self.node = rclpy.create_node("recording_manager")
        self._subscriber = self.node.create_subscription(
            String, "record_transition", self._callback_recording_trigger, 10
        )
        logger.debug("ROS2 node created and subscriber initialized.")

        threading.Thread(target=self._spin_node, daemon=True).start()

    def _spin_node(self):
        """Spin the ROS2 node in a separate thread."""
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    @override
    def get_instructions(self) -> str:
        """Returns the instructions to use the recording manager."""
        return (
            "[b]Published messages for recording state:[/b]\n"
            "<record> to start/stop recording.\n"
            "<save> to save the current recorded episode.\n"
            "<delete> to delete the current episode.\n"
            "<exit> to exit the recording manager."
        )

    def _callback_recording_trigger(self, msg: String) -> None:
        """Callback for recording state trigger.

        Args:
            msg: The message containing the recording state
        """
        if msg.data not in self.allowed_actions:
            print(f"[red]Invalid action received: {msg.data}[/red]")
            print("[yellow]Allowed actions are: record, save, delete, exit[/yellow]")
            return

        logger.debug(f"Received message: {msg.data}")
        logger.debug(f"Current state: {self.state}")

        if self.state == "is_waiting":
            if msg.data == "record":
                logger.debug("Transitioning to recording state.")
                self.state = "recording"
            if msg.data == "exit":
                logger.debug("Transitioning to exit state.")
                self.state = "exit"
        elif self.state == "recording":
            if msg.data == "record":
                logger.debug("Transitioning to paused state.")
                self.state = "paused"
        elif self.state == "paused":
            if msg.data == "exit":
                logger.debug("Transitioning to exit state.")
                self.state = "exit"
            if msg.data == "save":
                logger.debug("Transitioning to to_be_saved state.")
                self.state = "to_be_saved"
            if msg.data == "delete":
                logger.debug("Transitioning to to_be_deleted state.")
                self.state = "to_be_deleted"


class KeyboardRecordingManager(RecordingManager):
    """Keyboard-based recording manager for controlling episode recording."""

    def __init__(self, config: RecordingManagerConfig | None = None, **kwargs) -> None:  # noqa: ANN003
        """Initialize keyboard recording manager.

        Args:
            config: RecordingManagerConfig instance. If provided, **kwargs are ignored except for backwards compatibility.
            **kwargs: Individual parameters for backwards compatibility.
        """
        super().__init__(config=config, **kwargs)
        self.listener = keyboard.Listener(on_press=self._on_press)

    @override
    def get_instructions(self) -> str:
        """Returns the instructions to use the recording manager."""
        return "[b]Keys for recording:[/b]\n<r> To start/stop [b]R[/b]ecording.\n<s> To [b]S[/b]ave the current recorded episode.\n<d> to [b]D[/b]elete the current episode.\n<q> To [b]Q[/b]uit the recording."

    def _on_press(self, key: keyboard.KeyCode | keyboard.Key | None) -> None:
        """Handle keyboard press events.

        Args:
            key: The keyboard key that was pressed
        """
        if key is None:
            return

        if isinstance(key, keyboard.Key):
            return

        try:
            if self.state == "is_waiting":
                if key.char == "r":
                    self.state = "recording"
                if key.char == "q":
                    self.state = "exit"
            elif self.state == "recording":
                if key.char == "r":
                    self.state = "paused"
            elif self.state == "paused":
                if key.char == "q":
                    self.state = "exit"
                if key.char == "s":
                    self.state = "to_be_saved"
                if key.char == "d":
                    self.state = "to_be_deleted"
        except AttributeError:
            pass

    def stop(self) -> None:
        """Stop the keyboard listener."""
        self.listener.stop()

    def __enter__(self) -> "RecordingManager":  # noqa: D105
        self.listener.start()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001, D105
        self.listener.stop()
        super().__exit__(exc_type, exc_value, traceback)


def make_recording_manager(
    recording_manager_type: Literal["keyboard", "ros"],
    config: RecordingManagerConfig | None = None,
    config_path: Path | str | None = None,
    **kwargs: dict,
) -> RecordingManager:
    """Factory function to create a recording manager.

    Args:
        recording_manager_type: Type of recording manager to create.
        config: RecordingManagerConfig instance. Takes precedence over config_path.
        config_path: Path to YAML config file to load.
        **kwargs: Additional arguments to override config values or for backwards compatibility.

    Returns:
        A RecordingManager instance of the specified type.
    """
    if config is not None:
        if kwargs:
            config_dict = config.__dict__.copy()
            config_dict.update(kwargs)
            final_config = RecordingManagerConfig(**config_dict)
        else:
            final_config = config
    elif config_path is not None:
        final_config = RecordingManagerConfig.from_yaml(config_path, **kwargs)
    else:
        final_config = None

    if recording_manager_type == "keyboard":
        return KeyboardRecordingManager(config=final_config, **kwargs)
    elif recording_manager_type == "ros":
        return ROSRecordingManager(config=final_config, **kwargs)
    else:
        raise ValueError(f"Unknown recording manager type: {recording_manager_type}")
