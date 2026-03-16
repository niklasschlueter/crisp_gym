"""Standalone test: load any LeRobot policy from a checkpoint path and run inference.

Usage:
    python crisp_gym/scripts/test_lerobot_policy.py --policy-path /path/to/checkpoint
    python crisp_gym/scripts/test_lerobot_policy.py --policy-path continuallearning/groot_fft_10000steps_ga4_real_0_put_bowl
"""

import argparse
import json
import logging
import multiprocessing
import os
import time

import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

logger = logging.getLogger(__name__)


def _spawn_worker(conn, policy_path: str, device: str):
    """Top-level function for spawn test: loads policy on the given device and reports success/failure.

    Must be a top-level function (not a lambda/closure) to be picklable for spawn.
    """
    try:
        import json as _json
        from lerobot.policies.factory import get_policy_class

        if os.path.isdir(policy_path):
            config_path = os.path.join(policy_path, "config.json")
            with open(config_path) as f:
                cfg = _json.load(f)
        else:
            from huggingface_hub import hf_hub_download
            with open(hf_hub_download(policy_path, "config.json")) as f:
                cfg = _json.load(f)

        policy_type = cfg.get("type") or cfg.get("policy_type")
        policy_cls = get_policy_class(policy_type)
        policy = policy_cls.from_pretrained(policy_path)
        policy.to(device).eval()
        conn.send({"ok": True, "device": device})
    except Exception as e:
        conn.send({"ok": False, "error": str(e)})
    finally:
        conn.close()


def make_dummy_obs(policy, device: str) -> dict[str, torch.Tensor]:
    """Build a random observation batch (size 1) matching policy.config.input_features.

    Images: float32 in [0, 1] (FeatureType.VISUAL)
    State/env/etc: standard normal float32

    Note: tensors have explicit batch dim (1, ...) so AddBatchDimensionProcessorStep
    (which only adds dim to 1D/3D tensors) leaves them unchanged — no double-batching.
    """
    obs = {}
    for key, feature in policy.config.input_features.items():
        shape = (1, *feature.shape)
        if feature.type is FeatureType.VISUAL:
            obs[key] = torch.rand(shape, dtype=torch.float32, device=device)
        else:
            obs[key] = torch.randn(shape, dtype=torch.float32, device=device)
    return obs


def load_processors(policy, policy_path: str):
    """Load pre/post processors: try from checkpoint first, then fresh, then None.

    GR00T REQUIRES processors — its select_action filters batch to only eagle_*/state/*
    keys; without preprocessing, observation.* keys are silently dropped → crash.
    """
    # Attempt 1: load saved processors from checkpoint (includes dataset stats for normalization)
    try:
        pre, post = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy_path,
        )
        print("  status: loaded from checkpoint (with saved stats)")
        return pre, post
    except Exception as e1:
        print(f"  WARN: Could not load from checkpoint: {e1}")

    # Attempt 2: create fresh processors (no normalization stats — values may be out of range)
    try:
        pre, post = make_pre_post_processors(policy_cfg=policy.config)
        print("  status: created fresh (no dataset stats — normalization skipped)")
        return pre, post
    except Exception as e2:
        print(f"  WARN: Could not create fresh processors: {e2}")
        print("  status: unavailable — inference will proceed without preprocessing")
        print("  NOTE: GR00T policies WILL fail without processors!")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Test LeRobot policy loading and inference.")
    parser.add_argument("--policy-path", required=True, help="Path or HF Hub repo ID of LeRobot checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--n-bench", type=int, default=10, help="Benchmark iterations.")
    parser.add_argument("--test-spawn", action=argparse.BooleanOptionalAction, default=True,
                        help="After inference test, spawn a subprocess to verify no CUDA fork error.")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("LeRobot Policy Integration Test")
    print(f"{'='*60}")
    print(f"  policy_path : {args.policy_path}")
    print(f"  device      : {args.device}")

    # ── Step 1: Load train config ──────────────────────────────────
    print("\n[1/4] Loading TrainPipelineConfig...")
    try:
        train_cfg = TrainPipelineConfig.from_pretrained(args.policy_path)
        policy_type = train_cfg.policy.type
        print(f"  policy type : {policy_type}")
        print(f"  dataset     : {train_cfg.dataset.repo_id}")
    except Exception as e:
        print(f"  WARN: Could not load TrainPipelineConfig: {e}")
        print("  Falling back: reading policy type from config.json...")
        if os.path.isdir(args.policy_path):
            config_path = os.path.join(args.policy_path, "config.json")
            with open(config_path) as f:
                cfg_json = json.load(f)
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(args.policy_path, "config.json")
            with open(config_file) as f:
                cfg_json = json.load(f)
        policy_type = cfg_json.get("type") or cfg_json.get("policy_type")
        if not policy_type:
            raise RuntimeError("Cannot determine policy type from checkpoint.")
        print(f"  policy type : {policy_type}")

    # ── Step 2: Load policy ────────────────────────────────────────
    print("\n[2/4] Loading policy weights...")
    policy_cls = get_policy_class(policy_type)
    policy = policy_cls.from_pretrained(args.policy_path)
    policy.to(args.device).eval()
    policy.reset()

    print(f"  class       : {policy.__class__.__name__}")
    for key, ft in policy.config.input_features.items():
        print(f"    {key:40s} type={ft.type.value:8s} shape={ft.shape}")

    # ── Step 3: Pre/post processors ───────────────────────────────
    print("\n[3/4] Loading processors...")
    pre, post = load_processors(policy, args.policy_path)
    use_processors = pre is not None

    # ── Step 4: Dummy obs + inference ─────────────────────────────
    print(f"\n[4/4] Running inference ({args.n_warmup} warmup + {args.n_bench} bench)...")
    dummy_obs = make_dummy_obs(policy, args.device)
    if use_processors:
        dummy_obs = pre(dummy_obs)

    with torch.inference_mode():
        for _ in range(args.n_warmup):
            policy.reset()
            _ = policy.select_action(dummy_obs)
        if args.device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(args.n_bench):
            policy.reset()
            t0 = time.perf_counter()
            action = policy.select_action(dummy_obs)
            if args.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    if use_processors:
        action = post(action)

    avg_ms = sum(times) / len(times) * 1000
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  action shape : {tuple(action.shape)}")
    print(f"  avg latency  : {avg_ms:.2f} ms")
    print(f"  min / max    : {min(times)*1000:.2f} / {max(times)*1000:.2f} ms")
    print("\nPASS: policy loaded and inference completed successfully.")

    # ── Step 5: Spawn subprocess test (CUDA fork safety) ──────────
    if args.test_spawn:
        print(f"\n[5/5] Spawn subprocess test (CUDA already initialized in main process)...")
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_spawn_worker,
            args=(child_conn, args.policy_path, args.device),
            daemon=True,
        )
        proc.start()
        child_conn.close()  # close child end in parent
        result = parent_conn.recv()
        proc.join(timeout=300)
        if result.get("ok"):
            print(f"PASS: spawn worker loaded policy on {result['device']} without fork error")
        else:
            print(f"FAIL: spawn worker reported error: {result.get('error')}")
            raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
