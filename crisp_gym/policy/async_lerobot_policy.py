"""Asynchronous Lerobot Policy Module."""

import logging
import multiprocessing as mp
from collections import deque
from multiprocessing.connection import Connection
from typing import Callable, Tuple

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.utils import populate_queues

try:
    from lerobot.policies.factory import make_pre_post_processors
    USE_LEROBOT_PROCESSORS = True
except ImportError:
    USE_LEROBOT_PROCESSORS = False

try:
    from lerobot.utils.constants import ACTION, OBS_IMAGES
except ImportError:
    from lerobot.constants import ACTION, OBS_IMAGES
from typing_extensions import override

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.util.lerobot_features import concatenate_state_features, numpy_obs_to_torch


@register_policy("async_lerobot_policy")
class AsyncLerobotPolicy(Policy):
    """Asynchronous Lerobot Policy."""

    def __init__(self, pretrained_path: str, env: ManipulatorBaseEnv):
        """Initialize the policy."""
        # Derive n_obs from the policy's own config so the parent's obs-buffer
        # pre-fill matches what the worker actually needs (the worker reads
        # cfg.n_obs_steps at load time too — keep them in sync).
        # Loading PreTrainedConfig is cheap (config only, no weights).
        policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
        self.n_obs = int(policy_config.n_obs_steps)

        # Spawn context: fork + CUDA is broken (torch refuses re-init after
        # parent has touched it). See lerobot_policy.py for the same fix.
        ctx = mp.get_context("spawn")
        self.parent_conn, self.child_conn = ctx.Pipe()
        self.env = env
        # ToDo: expose n_act / replan_time as constructor args.
        self.n_act = 5
        self.replan_time = 3

        self.inf_proc = ctx.Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "steps": self.n_act,
                "replan_time": self.replan_time,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:  # noqa: ANN002, ANN003
        """Return a function that returns (obs, action) each frame by talking to the worker.

        Behaviour:
         - On first call: collect n_obs observations into a rolling buffer.
         - Request chunks according to the replan_time parameter.
         - Each call executes one action from the current chunk and returns (obs, action) for sorting/recording.
        """
        # Before starting, fill the observation buffer
        obs_buf: deque = deque(maxlen=self.n_obs)
        for _ in range(self.n_obs):
            obs_buf.append(self.env._get_obs())

        # Prepare first chunk in the case that n_act != replan_time
        if self.n_act != self.replan_time:
            self.parent_conn.send({"type": "OBS_SEQ", "obs_seq": list(obs_buf)})
            print("Starting new inference")

        i = 0
        next_chunk = None
        current_chunk = None

        def _fn() -> tuple:
            nonlocal i, next_chunk, current_chunk  # Required to mutate across calls
            if i == 0:
                if (
                    self.n_act == self.replan_time
                ):  # Edge case when we want to make a new prediction after all action chunks have been used up
                    obs_buf.append(self.env._get_obs())
                    self.parent_conn.send({"type": "OBS_SEQ", "obs_seq": list(obs_buf)})
                    print("Starting new inference")
                next_chunk = self.parent_conn.recv()
                current_chunk = next_chunk[self.n_act - self.replan_time :]
                print("Length ot the new current chunk:", len(current_chunk))

            # execute action
            action = current_chunk[i]
            print("Process element:", i)
            obs, *_ = self.env.step(action, block=False)
            obs_buf.append(obs)

            # Start prediction
            if i == (2 * self.replan_time - self.n_act):
                self.parent_conn.send({"type": "OBS_SEQ", "obs_seq": list(obs_buf)})
                print("Starting new inference")

            # step done
            i += 1

            # when done with one episode reset the counter
            if i >= (len(current_chunk)):
                i = 0

            return obs, action

        return _fn

    @override
    def reset(self):
        """Reset the policy state."""
        self.parent_conn.send("reset")

    @override
    def shutdown(self):
        """Shutdown the policy and release resources."""
        self.parent_conn.send(None)
        _drain_conn(self.parent_conn)
        self.inf_proc.join()


def inference_worker(  # noqa: D417
    conn: Connection,
    pretrained_path: str,
    steps: int | None,
    replan_time: int,
):  # noqa: ANN001
    """Policy inference process: loads policy on GPU, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
        steps (int): How many actions are executed from the prediction
        replan_time (int): After how many steps to start predicting a new action chunk
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config = TrainPipelineConfig.from_pretrained(pretrained_path)
    if train_config.policy is None:
        raise ValueError(
            f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
            "Please ensure the policy is correctly configured."
        )
    policy_cls = get_policy_class(train_config.policy.type)

    policy_config = PreTrainedConfig.from_pretrained(pretrained_path)

    if steps is not None:
        # Check if the number of steps make sense
        horizon = policy_config.horizon
        if steps >= horizon:
            raise ValueError(
                f"The policy steps={steps} must be smaller than the horizon={horizon}."
                "Please modify your cli."
            )
        policy_config.n_action_steps = int(steps)

    policy = policy_cls.from_pretrained(pretrained_path, config=policy_config)

    # The queue-priming + OBS_IMAGES-stacking in the inference loop below
    # assumes a policy whose predict_action_chunk reads from self._queues
    # with per-feature deques (DiffusionPolicy, SmolVLAPolicy). ACTPolicy
    # uses a different layout (_action_queue + list-of-tensors for images)
    # and would crash here. Fail fast with a clear message.
    if not hasattr(policy, "_queues"):
        raise RuntimeError(
            f"AsyncLerobotPolicy only supports policies whose predict_action_chunk "
            f"reads from self._queues (diffusion, smolvla). Got {policy.name}. "
            f"Use LerobotPolicy (sync) instead, which dispatches via select_action."
        )

    logging.info(
        f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
    )

    policy.reset()
    policy.to(device).eval()

    if USE_LEROBOT_PROCESSORS:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=pretrained_path
        )
    else:
        preprocessor = postprocessor = None
        logging.warning(
            "[AsyncInference] No processor chain — predictions will be un-normalised."
        )

    # Read policy config to know obs/action window sizes
    cfg = policy.config
    n_obs = int(cfg.n_obs_steps)
    print("Ready to recive information")

    while True:
        # Check if messages are recieved correctly
        msg = conn.recv()
        if msg is None:
            break
        if msg == "reset":
            logging.info("[Inference] Resetting policy")
            policy.reset()
            if preprocessor is not None:
                preprocessor.reset()
                postprocessor.reset()
            continue
        if not (isinstance(msg, dict) and msg.get("type") == "OBS_SEQ"):
            logging.warning(f"[Inference] Unknown message: {type(msg)}")
            continue

        # We are recieving a list of dictonaries with the last observations
        obs_seq = msg["obs_seq"]

        # Make the policy predict an action chunk for the current observation.
        # v0.5 flow:
        #   - preprocessor handles per-key normalization but does NOT stack
        #     image_features into the combined OBS_IMAGES key — we do that
        #     manually, same as the v0.3 pattern.
        #   - preprocessor adds placeholder entries for `action` /
        #     `next.reward` / etc., which we must NOT populate into
        #     policy._queues[ACTION] (mirrors lerobot's own select_action,
        #     which passes exclude_keys=[ACTION]).
        with torch.inference_mode():
            for i in range(n_obs):
                last = obs_seq[i]
                last["observation.state"] = concatenate_state_features(last)
                batch = numpy_obs_to_torch(last)
                if preprocessor is not None:
                    batch = preprocessor(batch)
                # Stack per-camera image features into the combined OBS_IMAGES
                # key expected by the policy's internal queue.
                if policy.config.image_features:
                    batch = dict(batch)  # shallow copy before mutating
                    batch[OBS_IMAGES] = torch.stack(
                        [batch[k] for k in policy.config.image_features], dim=-4
                    )
                policy._queues = populate_queues(
                    policy._queues, batch, exclude_keys=[ACTION]
                )

            # Now get a fresh chunk using the last preprocessed batch.
            # Drop non-observation placeholders the preprocessor tacks on
            # (action / next.reward / next.done / next.truncated / info) —
            # predict_action_chunk iterates batch keys against policy._queues
            # and would hit the empty action queue otherwise.
            batch_for_chunk = {
                k: v for k, v in batch.items()
                if k.startswith("observation.")
            }
            chunk = policy.predict_action_chunk(batch_for_chunk)
            if postprocessor is not None:
                chunk = postprocessor(chunk)
            chunk = chunk.squeeze(0).to(device="cpu").numpy()

        logging.debug(f"[Inference] Computed chunk with shape {tuple(chunk.shape)}")
        conn.send(chunk)

    conn.close()
    logging.info("[Inference] Worker shutting down")


# To be implemented later to avoid stale messages in the pipe
def _drain_conn(conn):  # noqa: ANN001
    """Non-blocking: remove any pending messages so we don't reuse stale chunks."""
    try:
        while conn.poll(0):
            _ = conn.recv()
    except (EOFError, OSError):
        pass
