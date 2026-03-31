"""Interface for a Policy interacting in CRISP."""

import csv
import datetime
import json
import logging
import multiprocessing
import statistics
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import LeRobotDatasetMetadata, get_policy_class
from typing_extensions import override

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.util.lerobot_features import concatenate_state_features, numpy_obs_to_torch
from crisp_gym.util.setup_logger import setup_logging

try:
    from lerobot.policies.factory import make_pre_post_processors
    USE_LEROBOT_PROCESSORS = True
    logging.info("Found lerobot pre/post processor support.")
except ImportError:
    USE_LEROBOT_PROCESSORS = False
    logging.warning("No lerobot pre/post processor support found.")


logger = logging.getLogger(__name__)


@register_policy("lerobot_policy")
class LerobotPolicy(Policy):
    """A policy implementation that wraps a LeRobot policy for use in CRISP environments.

    This class runs LeRobot policy inference in a separate process and communicates with the
    environment to generate actions based on observations. It is intended for direct use in
    CRISP-based manipulation environments.
    """

    def __init__(
        self,
        pretrained_path: str,
        env: ManipulatorBaseEnv,
        overrides: dict | None = None,
        task: str | None = None,
        peft_path: str | None = None,
        profile: bool = False,
        repo_id: str | None = None,
    ):
        """Initialize the policy.

        Args:
            pretrained_path (str): Path to the pretrained policy model.
            env (ManipulatorBaseEnv): The environment in which the policy will be applied.
            overrides (dict | None): Optional overrides for the policy configuration.
            task (str | None): Task description for language-conditioned policies.
        """
        self.env = env
        self.overrides = overrides if overrides is not None else {}
        self.task = task

        ctx = multiprocessing.get_context("spawn")
        self.parent_conn, self.child_conn = ctx.Pipe()

        # Extract env data before spawning (env may not be picklable — ROS2 handles)
        observation_space = env.observation_space
        env_metadata = env.get_metadata()

        self.inf_proc = ctx.Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "observation_space": observation_space,
                "env_metadata": env_metadata,
                "overrides": self.overrides,
                "task": self.task,
                "peft_path": peft_path,
                "profile": profile,
                "repo_id": repo_id,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:  # noqa: ANN002, ANN003
        """Generate observation and action by communicating with the inference worker."""

        def _fn() -> tuple:
            """Function to apply the policy in the environment.

            This function observes the current state of the environment, sends the observation
            to the inference worker, receives the action, and steps the environment.

            Returns:
                tuple: A tuple containing the observation from the environment and the action taken.
            """
            logger.debug("Requesting action from policy...")
            obs_raw: Observation = self.env.get_obs()

            obs_raw["observation.state"] = concatenate_state_features(obs_raw)

            if self.task is not None:
                obs_raw["task"] = self.task

            self.parent_conn.send(obs_raw)
            action: Action = self.parent_conn.recv().squeeze(0).to("cpu").numpy()
            logger.debug(f"Action: {action}")

            try:
                self.env.step(action, block=False)
                # pass
            except Exception as e:
                logger.exception(f"Error during environment step: {e}")

            return obs_raw, action

        return _fn

    def set_task(self, task: str):
        """Update the language instruction for subsequent inference steps."""
        logger.info(f"[Policy] Task changed to: '{task}'")
        self.task = task

    @override
    def reset(self):
        """Reset the policy state."""
        self.parent_conn.send("reset")

    @override
    def shutdown(self):
        """Shutdown the policy and release resources."""
        self.parent_conn.send(None)
        self.inf_proc.join()


def _save_profile_data(logger, step_profiles, device, adapter_vram=None, disc_vram=None,
                       backbone_vram_mb=0.0, total_param_vram_mb=0.0, repo_id=None):
    """Save profiling data to JSON, CSV, and Markdown in ./outputs/clare_profile/."""
    if not step_profiles:
        return
    if adapter_vram is None:
        adapter_vram = {}
    if disc_vram is None:
        disc_vram = {}

    profile_dir = Path("outputs/clare_profile")
    profile_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_prefix = repo_id.replace("/", "_") if repo_id else "profile"

    n = len(step_profiles)

    # Build summary
    summary = {}
    timing_keys = set()
    for s in step_profiles:
        for k in s:
            if k.endswith("_ms"):
                timing_keys.add(k)
    for key in sorted(timing_keys):
        vals = [s[key] for s in step_profiles if key in s and s[key] is not None]
        if vals:
            summary[key] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "p50": statistics.median(vals),
                "p95": sorted(vals)[int(len(vals) * 0.95)] if len(vals) > 1 else vals[0],
                "min": min(vals),
                "max": max(vals),
                "count": len(vals),
            }

    # VRAM info (static, from first step that has it)
    vram_summary = {}
    vram_keys = set()
    for s in step_profiles:
        for k in s:
            if k.endswith("_vram_mb"):
                vram_keys.add(k)
    for key in sorted(vram_keys):
        for s in step_profiles:
            if key in s and s[key] is not None:
                vram_summary[key] = s[key]
                break

    # Process-level VRAM
    process_vram = {}
    if device.type == "cuda":
        process_vram = {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        }

    # Sanitize step_profiles for JSON (convert non-serializable types)
    sanitized_steps = []
    for s in step_profiles:
        sanitized = {}
        for k, v in s.items():
            if isinstance(v, (list, tuple)):
                sanitized[k] = [int(x) if isinstance(x, (int, np.integer)) else x for x in v]
            elif isinstance(v, (np.integer,)):
                sanitized[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sanitized[k] = float(v)
            else:
                sanitized[k] = v
        sanitized_steps.append(sanitized)

    # Write JSON
    json_path = profile_dir / f"{file_prefix}_{timestamp}.json"
    json_data = {
        "num_steps": n,
        "steps": sanitized_steps,
        "summary": summary,
        "vram_per_component": vram_summary,
        "vram_adapters": adapter_vram,
        "vram_discriminators": disc_vram,
        "vram_backbone_mb": backbone_vram_mb,
        "vram_total_model_mb": total_param_vram_mb,
        "vram_process": process_vram,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"[Profile] JSON saved to {json_path}")

    # Write CSV
    csv_path = profile_dir / f"{file_prefix}_{timestamp}.csv"
    all_keys = set()
    for s in sanitized_steps:
        all_keys.update(k for k in s.keys() if not isinstance(s[k], (list, dict)))
    fieldnames = sorted(all_keys, key=lambda k: (k != "step", k != "select_action_ms", k))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in sanitized_steps:
            row = {k: v for k, v in s.items() if not isinstance(v, (list, dict))}
            writer.writerow(row)
    logger.info(f"[Profile] CSV saved to {csv_path}")

    # Write Markdown report
    md_path = profile_dir / f"{file_prefix}_{timestamp}.md"
    aggregate_keys = ["select_action_ms", "backbone_ms",
                      "all_discriminators_total_ms", "all_discriminators_mean_ms",
                      "all_adapters_total_ms", "all_adapters_mean_ms"]
    md_lines = [
        f"# CLARE Inference Profile Report",
        f"**Repo ID:** {repo_id or 'N/A'} | **Date:** {date_str} | **Steps:** {n}",
        "",
        "## Timing Summary",
        "| Metric | Mean (ms) | Std | P50 | P95 | Min | Max |",
        "|--------|-----------|-----|-----|-----|-----|-----|",
    ]
    for key in aggregate_keys:
        if key in summary:
            s = summary[key]
            md_lines.append(
                f"| {key} | {s['mean']:.2f} | {s['std']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} | {s['min']:.2f} | {s['max']:.2f} |"
            )

    per_layer = {k: v for k, v in summary.items() if k not in aggregate_keys}
    if per_layer:
        md_lines += [
            "",
            "## Per-Layer Detail",
            "| Metric | Mean (ms) | Std | Steps |",
            "|--------|-----------|-----|-------|",
        ]
        for key, s in per_layer.items():
            md_lines.append(f"| {key} | {s['mean']:.2f} | {s['std']:.2f} | {s['count']} |")

    md_lines += ["", "## VRAM (Model Params)", "| Component | MB |", "|-----------|---:|"]
    if total_param_vram_mb > 0:
        md_lines.append(f"| **Total model** | {total_param_vram_mb:.2f} |")
        md_lines.append(f"| **Backbone** | {backbone_vram_mb:.2f} |")
        md_lines.append(f"| **All adapters** | {sum(adapter_vram.values()):.2f} |")
        md_lines.append(f"| **All discriminators** | {sum(disc_vram.values()):.2f} |")
    for key, val in adapter_vram.items():
        md_lines.append(f"| {key} | {val:.2f} |")
    for key, val in disc_vram.items():
        md_lines.append(f"| {key} | {val:.2f} |")
    for key, val in vram_summary.items():
        md_lines.append(f"| {key} | {val:.2f} |")

    if process_vram:
        md_lines += [
            "",
            "## Process VRAM",
            "| Metric | MB |",
            "|-----------|---:|",
            f"| Allocated | {process_vram['allocated_mb']:.1f} |",
            f"| Max allocated | {process_vram['max_allocated_mb']:.1f} |",
        ]

    md_lines += [
        "",
        "---",
        f"*Files: [{file_prefix}_{timestamp}.json]({file_prefix}_{timestamp}.json) | [{file_prefix}_{timestamp}.csv]({file_prefix}_{timestamp}.csv)*",
    ]
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    logger.info(f"[Profile] Markdown report saved to {md_path}")

    # Log summary — highlight aggregate metrics first
    logger.info(f"[Profile] === Timing Summary ({n} steps) ===")
    for key in aggregate_keys:
        if key in summary:
            stats = summary[key]
            logger.info(
                f"[Profile] {key}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                f"p50={stats['p50']:.2f}, p95={stats['p95']:.2f}, "
                f"min={stats['min']:.2f}, max={stats['max']:.2f}"
            )

    logger.info("[Profile] === Per-Layer Detail ===")
    for key, stats in summary.items():
        if key not in aggregate_keys:
            logger.info(
                f"[Profile] {key}: mean={stats['mean']:.2f}, std={stats['std']:.2f} "
                f"(over {stats['count']} steps)"
            )

    logger.info("[Profile] === VRAM (model params) ===")
    if total_param_vram_mb > 0:
        logger.info(f"[Profile] Total model: {total_param_vram_mb:.2f}MB")
        logger.info(f"[Profile] Backbone: {backbone_vram_mb:.2f}MB")
        logger.info(f"[Profile] All adapters: {sum(adapter_vram.values()):.2f}MB")
        logger.info(f"[Profile] All discriminators: {sum(disc_vram.values()):.2f}MB")
    for key, val in adapter_vram.items():
        logger.info(f"[Profile]   {key}: {val:.2f}MB")
    for key, val in disc_vram.items():
        logger.info(f"[Profile]   {key}: {val:.2f}MB")
    for key, val in vram_summary.items():
        logger.info(f"[Profile]   {key}: {val:.2f}MB")
    if process_vram:
        logger.info(
            f"[Profile] Process VRAM: allocated={process_vram['allocated_mb']:.1f}MB, "
            f"max_allocated={process_vram['max_allocated_mb']:.1f}MB"
        )


def inference_worker(
    conn: Connection,
    pretrained_path: str,
    observation_space,
    env_metadata: dict,
    overrides: dict | None = None,
    task: str | None = None,
    peft_path: str | None = None,
    profile: bool = False,
    repo_id: str | None = None,
):  # noqa: ANN001
    """Policy inference process: loads policy on GPU, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
        observation_space: The environment's observation space (pre-extracted for spawn compatibility).
        env_metadata (dict): The environment metadata (pre-extracted for spawn compatibility).
        overrides (dict | None): Optional overrides for the policy configuration.
        task (str | None): Task description for language-conditioned policies.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
    except ImportError:
        logger.warning(
            "[Inference] Could not import third-party plugins for LeRobot. Continuing without them."
        )
    logger.info("[Inference] Starting inference worker...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Inference] Using device: {device}")

        logger.info(f"[Inference] Loading training config from {pretrained_path}...")

        train_config = TrainPipelineConfig.from_pretrained(pretrained_path)

        _check_dataset_metadata(train_config, env_metadata, logger)

        logger.info("[Inference] Loaded training config.")

        logger.debug(f"[Inference] Train config: {train_config}")

        if train_config.policy is None:
            raise ValueError(
                f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
                "Please ensure the policy is correctly configured."
            )

        logger.info("[Inference] Loading policy...")
        policy_cls = get_policy_class(train_config.policy.type)
        policy = policy_cls.from_pretrained(pretrained_path)
        policy_name = policy.name

        if peft_path is not None:
            from peft import PeftConfig, PeftModel

            logger.info(f"[Inference] Loading PEFT adapter from {peft_path}...")
            peft_config = PeftConfig.from_pretrained(peft_path)
            policy = PeftModel.from_pretrained(policy, peft_path, config=peft_config)
            logger.info("[Inference] PEFT adapter applied successfully.")

        # After PEFT wrapping, policy.config is PeftConfig, not the model config
        model_config = policy.get_base_model().config if peft_path is not None else policy.config
        for override_key, override_value in (overrides or {}).items():
            logger.warning(
                f"[Inference] Overriding policy config: {override_key} = {getattr(model_config, override_key)} -> {override_value}"
            )
            setattr(model_config, override_key, override_value)

        logger.info(f"[Inference] num_inference_steps = {model_config.num_inference_steps}")

        # logger.info(
        #     f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
        # )
        policy.reset()
        policy.to(device).eval()

        if USE_LEROBOT_PROCESSORS:
            norm_path = peft_path if peft_path is not None else pretrained_path
            preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=norm_path)


            # logger.info(f"[Inference] Normalization stats loaded from: {norm_path}")
            # for name, pipeline in [("preprocessor", preprocessor), ("postprocessor", postprocessor)]:
            #     for step in pipeline.steps:
            #         if hasattr(step, "stats") and step.stats:
            #             logger.info(f"[Inference] {name} step {step.__class__.__name__} stats: {step.stats}")

        warmup_obs_raw = observation_space.sample()
        warmup_obs_raw["observation.state"] = concatenate_state_features(warmup_obs_raw)
        if task is not None:
            warmup_obs_raw["task"] = task
        warmup_obs = numpy_obs_to_torch(warmup_obs_raw)
        if USE_LEROBOT_PROCESSORS:
            warmup_obs = preprocessor(warmup_obs)
            warmup_obs.pop("action", None)

        logger.info("[Inference] Warming up policy...")
        elapsed_list = []
        with torch.inference_mode():
            for _ in range(100):
                start = time.time()
                _ = policy.select_action(warmup_obs)
                end = time.time()
                elapsed = end - start
                elapsed_list.append(elapsed)

            torch.cuda.synchronize()

        avg_elapsed = sum(elapsed_list) / len(elapsed_list)
        std_elapsed = np.std(elapsed_list)
        max_elapsed = max(elapsed_list)
        min_elapsed = min(elapsed_list)
        logger.info(
            f"[Inference] Warm-up timing over 100 runs: "
            f"avg={avg_elapsed * 1000:.2f}ms, std={std_elapsed * 1000:.2f}ms, max={max_elapsed * 1000:.2f}ms, min={min_elapsed * 1000:.2f}ms"
        )

        logger.info("[Inference] Warm-up complete")

        # Enable CLARE profiling after warmup
        clare_layers = []
        adapter_vram = {}  # {layer_i/adapter_j: vram_mb}
        disc_vram = {}
        backbone_vram_mb = 0.0
        total_param_vram_mb = 0.0
        if profile and peft_path is not None:
            # Step 1: Discover CLARE layers
            try:
                from peft.tuners.clare.layer import CLARELayer
                clare_layers = [m for m in policy.modules() if isinstance(m, CLARELayer)]
            except ImportError:
                logger.warning("[Profile] Could not import CLARELayer")
            logger.info(f"[Profile] Found {len(clare_layers)} CLARELayers via module scan")

            # Step 2: Enable profiling on each layer
            for layer in clare_layers:
                layer._profile_inference = True

            # Step 3: Compute VRAM breakdown
            if clare_layers:
                for i, layer in enumerate(clare_layers):
                    adapter_name = layer.adapter_name
                    for j, adapter in enumerate(layer.clare_func_adapters[adapter_name]):
                        param_bytes = sum(p.nelement() * p.element_size() for p in adapter.parameters())
                        vram_mb = param_bytes / (1024 ** 2)
                        adapter_vram[f"layer_{i}/adapter_{j}_vram_mb"] = vram_mb
                        logger.info(f"[Profile] layer_{i}/adapter_{j} VRAM: {vram_mb:.2f}MB")

                    for j, disc in enumerate(layer.clare_discriminators[adapter_name]):
                        param_bytes = sum(p.nelement() * p.element_size() for p in disc.parameters())
                        vram_mb = param_bytes / (1024 ** 2)
                        disc_vram[f"layer_{i}/discriminator_{j}_vram_mb"] = vram_mb
                        logger.info(f"[Profile] layer_{i}/discriminator_{j} VRAM: {vram_mb:.2f}MB")

                total_param_bytes = sum(p.nelement() * p.element_size() for p in policy.parameters())
                total_param_vram_mb = total_param_bytes / (1024 ** 2)
                total_adapter_vram_mb = sum(adapter_vram.values())
                total_disc_vram_mb = sum(disc_vram.values())
                backbone_vram_mb = total_param_vram_mb - total_adapter_vram_mb - total_disc_vram_mb

                logger.info(f"[Profile] Total model VRAM: {total_param_vram_mb:.2f}MB")
                logger.info(f"[Profile] Backbone VRAM: {backbone_vram_mb:.2f}MB")
                logger.info(f"[Profile] Total adapter VRAM: {total_adapter_vram_mb:.2f}MB")
                logger.info(f"[Profile] Total discriminator VRAM: {total_disc_vram_mb:.2f}MB")

        if profile and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            logger.info(
                f"[Profile] VRAM after warmup: "
                f"allocated={torch.cuda.memory_allocated(device) / 1024**2:.1f}MB"
            )

        step_profiles = []
        step_count = 0

        while True:
            obs_raw = conn.recv()
            if obs_raw is None:
                break
            if obs_raw == "reset":
                logger.info("[Inference] Resetting policy")
                policy.reset()
                if USE_LEROBOT_PROCESSORS:
                    preprocessor.reset()
                    postprocessor.reset()
                continue

            with torch.inference_mode():
                obs = numpy_obs_to_torch(obs_raw)
                if USE_LEROBOT_PROCESSORS:
                    obs = preprocessor(obs)
                    obs.pop("action", None)

                if profile:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t0 = time.perf_counter()

                action = policy.select_action(obs)

                if profile:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    step_data = {"step": step_count, "select_action_ms": elapsed_ms}
                    all_disc_times = []
                    all_adapter_times = []
                    for i, layer in enumerate(clare_layers):
                        info = layer.info_dicts
                        if not info:
                            continue
                        prefix = f"layer_{i}"
                        if "top_1_idx_list" in info:
                            step_data[f"{prefix}/routing"] = info["top_1_idx_list"]
                        if "adapter_timing" in info:
                            for aid, atinfo in info["adapter_timing"].items():
                                t = atinfo.get("time_ms")
                                step_data[f"{prefix}/adapter_{aid}_ms"] = t
                                if t is not None:
                                    all_adapter_times.append(t)
                        for key in info:
                            if key.startswith("discriminator_"):
                                d_info = info[key]
                                if "time_ms" in d_info:
                                    t = d_info["time_ms"]
                                    step_data[f"{prefix}/{key}_ms"] = t
                                    all_disc_times.append(t)
                                if "loaded_vram_mb" in d_info:
                                    step_data[f"{prefix}/{key}_vram_mb"] = d_info["loaded_vram_mb"]
                    # Aggregate timing across all layers
                    disc_total = sum(all_disc_times) if all_disc_times else 0.0
                    adapter_total = sum(all_adapter_times) if all_adapter_times else 0.0
                    if all_disc_times:
                        step_data["all_discriminators_total_ms"] = disc_total
                        step_data["all_discriminators_mean_ms"] = disc_total / len(all_disc_times)
                    if all_adapter_times:
                        step_data["all_adapters_total_ms"] = adapter_total
                        step_data["all_adapters_mean_ms"] = adapter_total / len(all_adapter_times)
                    # Backbone time = select_action - discriminators - adapters
                    step_data["backbone_ms"] = elapsed_ms - disc_total - adapter_total
                    step_profiles.append(step_data)
                    step_count += 1

                if USE_LEROBOT_PROCESSORS:
                    action = postprocessor(action)

            logger.debug(f"[Inference] Computed action: {action}")
            conn.send(action)

        # Save profiling data on shutdown
        if profile and step_profiles:
            _save_profile_data(
                logger, step_profiles, device,
                adapter_vram=adapter_vram,
                disc_vram=disc_vram,
                backbone_vram_mb=backbone_vram_mb,
                total_param_vram_mb=total_param_vram_mb,
                repo_id=repo_id,
            )
    except Exception as e:
        logger.exception(f"[Inference] Exception in inference worker: {e}")

    conn.close()
    logger.info("[Inference] Worker shutting down")


def _check_dataset_metadata(
    train_config: TrainPipelineConfig,
    env_metadata: dict,
    logger: logging.Logger,
    keys_to_skip: list[str] | None = None,
):
    """Check if the dataset metadata matches the environment configuration.

    Args:
        train_config (TrainPipelineConfig): The training pipeline configuration.
        env_metadata (dict): The environment metadata dict to compare against.
        logger (logging.Logger): Logger for logging information.
        keys_to_skip (list[str] | None): List of metadata keys to skip during comparison.
    """
    if keys_to_skip is None:
        keys_to_skip = []

    def _warn_if_not_equal(key: str, env_val: Any, policy_val: Any):
        if env_val != policy_val:
            logger.warning(
                f"[Inference] Mismatch in metadata for key '{key}': "
                f"env has '{env_val}', policy has '{policy_val}'."
            )

    def _warn_if_missing(key: str):
        logger.warning(f"[Inference] Key '{key}' not found in environment metadata.")

    try:
        metadata = LeRobotDatasetMetadata(repo_id=train_config.dataset.repo_id)
        logger.debug(f"[Inference] Loaded dataset metadata: {metadata}")

        path_to_metadata = Path(metadata.root / "meta" / "crisp_meta.json")
        if path_to_metadata.exists():
            logger.info(
                "[Inference] Found crisp_meta.json in dataset, comparing environment and policy configs..."
            )
            with open(path_to_metadata, "r") as f:
                dataset_metadata = json.load(f)
            for key, value in dataset_metadata.items():
                if key in keys_to_skip:
                    continue
                if isinstance(value, dict):
                    if key not in env_metadata:
                        _warn_if_missing(key)
                        continue
                    for subkey, subvalue in value.items():
                        if subkey not in env_metadata[key]:
                            _warn_if_missing(f"{key}.{subkey}")
                            continue
                        _warn_if_not_equal(
                            f"{key}.{subkey}",
                            env_metadata[key].get(subkey),
                            subvalue,
                        )
                else:
                    if key not in env_metadata:
                        _warn_if_missing(key)
                    _warn_if_not_equal(key, env_metadata.get(key), value)

    except Exception as e:
        logger.warning(f"[Inference] Could not load dataset metadata: {e}")
        logger.info("[Inference] Skipping metadata comparison.")
