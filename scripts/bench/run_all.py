"""Automated benchmark runner for the crisp_gym recording-rate fix.

Spawns bench_publisher.py + bench_recorder.py for a small matrix of
configurations (SHM on/off × camera resolution), parses the recorder's
stdout, and prints a comparison table plus a pass/fail verdict.

Run (inside the jazzy-lerobot pixi env):
  pixi run -e jazzy-lerobot python scripts/bench/run_all.py
  pixi run -e jazzy-lerobot python scripts/bench/run_all.py --duration 30

Exit code: 0 if every case meets >= 95% of target rate with zero overrun
warnings, 1 otherwise.
"""

import argparse
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
PUBLISHER = HERE / "bench_publisher.py"
RECORDER = HERE / "bench_recorder.py"

_RATE_RE = re.compile(r"actual rate:\s+([0-9.]+)\s*Hz\s+\(([0-9]+) frames / ([0-9.]+)s\)")
_WARN_RE = re.compile(r"overrun warnings:\s+([0-9]+)")


@dataclass
class TestCase:
    name: str
    source_height: int
    source_width: int
    target_height: int
    target_width: int
    shm: bool


@dataclass
class Result:
    case: TestCase
    target_rate: int
    actual_rate: float
    frames: int
    elapsed: float
    warnings: int

    @property
    def passed(self) -> bool:
        return self.actual_rate >= 0.95 * self.target_rate and self.warnings == 0


def _cleanup_shm() -> None:
    """Remove leftover /dev/shm/crisp_record_* blocks from prior runs."""
    for p in Path("/dev/shm").glob("crisp_record_*"):
        try:
            p.unlink()
        except OSError:
            pass


def _check_prereqs() -> None:
    for path in (PUBLISHER, RECORDER):
        if not path.exists():
            print(f"FATAL: missing required script {path}", file=sys.stderr)
            sys.exit(2)


def run_case(case: TestCase, duration: float, target_rate: int) -> Optional[Result]:
    _cleanup_shm()

    pub_cmd = [
        sys.executable,
        str(PUBLISHER),
        "--rate", str(target_rate),
        "--width", str(case.source_width),
        "--height", str(case.source_height),
    ]
    rec_cmd = [
        sys.executable,
        str(RECORDER),
        "--duration", str(duration),
        "--target-rate", str(target_rate),
        "--target-height", str(case.target_height),
        "--target-width", str(case.target_width),
    ]
    if not case.shm:
        rec_cmd.append("--no-shm")

    print(f"\n>>> {case.name}")
    print(f"    publisher  src={case.source_height}x{case.source_width} @ {target_rate} Hz")
    print(f"    recorder   target={case.target_height}x{case.target_width}  shm={'on' if case.shm else 'off'}")
    sys.stdout.flush()

    pub = subprocess.Popen(pub_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2.5)  # let publisher come up

    if pub.poll() is not None:
        _, err = pub.communicate(timeout=2)
        print("    !! publisher exited prematurely:")
        print(err.decode(errors="replace"))
        return None

    t0 = time.time()
    try:
        rec = subprocess.run(rec_cmd, capture_output=True, text=True, timeout=duration + 60)
    except subprocess.TimeoutExpired:
        print(f"    !! recorder timed out after {duration + 60:.0f}s")
        rec = None
    finally:
        pub.send_signal(signal.SIGTERM)
        try:
            pub.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pub.kill()

    if rec is None:
        return None

    elapsed_wall = time.time() - t0
    print(f"    finished in {elapsed_wall:.1f}s")

    rate_m = _RATE_RE.search(rec.stdout)
    warn_m = _WARN_RE.search(rec.stdout)
    if not rate_m or not warn_m:
        print("    !! could not parse recorder output:")
        print("    --- recorder stdout ---")
        print(rec.stdout)
        print("    --- recorder stderr ---")
        print(rec.stderr)
        return None

    return Result(
        case=case,
        target_rate=target_rate,
        actual_rate=float(rate_m.group(1)),
        frames=int(rate_m.group(2)),
        elapsed=float(rate_m.group(3)),
        warnings=int(warn_m.group(1)),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--duration", type=float, default=15.0, help="Seconds per case (default 15)")
    p.add_argument("--target-rate", type=int, default=30, help="Target FPS")
    args = p.parse_args()

    _check_prereqs()

    cases = [
        TestCase(
            "720p source, full-res obs (worst case for queue.put)",
            source_height=720, source_width=1280,
            target_height=720, target_width=1280,
            shm=True,
        ),
        TestCase(
            "720p source, full-res obs (baseline)",
            source_height=720, source_width=1280,
            target_height=720, target_width=1280,
            shm=False,
        ),
        TestCase(
            "720p source, 256x256 resize",
            source_height=720, source_width=1280,
            target_height=256, target_width=256,
            shm=True,
        ),
        TestCase(
            "720p source, 256x256 resize (baseline)",
            source_height=720, source_width=1280,
            target_height=256, target_width=256,
            shm=False,
        ),
    ]

    results = []
    for c in cases:
        r = run_case(c, args.duration, args.target_rate)
        if r is not None:
            results.append(r)

    _cleanup_shm()

    if not results:
        print("\nNo successful runs.")
        return 1

    print()
    print("=" * 84)
    print(" SUMMARY")
    print("=" * 84)
    print(f" {'case':<54} {'rate':>9} {'warns':>6} {'verdict':>8}")
    print(" " + "-" * 82)
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        print(f" {r.case.name:<54} {r.actual_rate:>7.2f}Hz {r.warnings:>6} {verdict:>8}")
    print("=" * 84)

    on720 = next(
        (r for r in results
         if r.case.target_height == 720 and r.case.shm), None
    )
    off720 = next(
        (r for r in results
         if r.case.target_height == 720 and not r.case.shm), None
    )
    if on720 and off720 and off720.actual_rate > 0:
        speedup = on720.actual_rate / off720.actual_rate
        print(
            f" 720p SHM speedup: {speedup:.2f}x  "
            f"(off={off720.actual_rate:.2f}Hz, on={on720.actual_rate:.2f}Hz)"
        )

    n_pass = sum(1 for r in results if r.passed)
    print(f" overall: {n_pass}/{len(results)} cases pass")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
