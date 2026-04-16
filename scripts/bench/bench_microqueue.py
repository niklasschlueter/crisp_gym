"""Microbenchmark: pickle-based vs SharedMemory-based mp.Queue throughput.

Isolates the SHM fix from ROS, JPEG decode, cameras, and DDS jitter.
Only measures the producer-side cost of getting one image-shaped numpy
array into a multiprocessing queue and out the other side.

Run:
  pixi run -e jazzy-lerobot python scripts/bench/bench_microqueue.py
"""

import argparse
import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np


SHAPES = {
    "720p (720x1280x3 = 2.76 MB)": (720, 1280, 3),
    "256x256 (196 KB)": (256, 256, 3),
    "224x224 (147 KB)": (224, 224, 3),
}


# ---------------- baseline: pickle-based queue --------------------------------

def _pickle_writer(q: mp.JoinableQueue, count_value):
    while True:
        msg = q.get()
        if msg is None:
            break
        # Touch the array to force any lazy unpickling
        _ = msg["img"].shape
        with count_value.get_lock():
            count_value.value += 1


def measure_pickle(shape: tuple, n_frames: int, queue_size: int = 16):
    q = mp.JoinableQueue(queue_size)
    count = mp.Value("i", 0)
    p = mp.Process(target=_pickle_writer, args=(q, count))
    p.start()
    img = np.random.randint(0, 256, shape, dtype=np.uint8)

    t0 = time.perf_counter()
    times = []
    for _ in range(n_frames):
        ti = time.perf_counter()
        q.put({"img": img.copy()})  # copy so each frame is fresh in producer memory
        times.append(time.perf_counter() - ti)
    elapsed = time.perf_counter() - t0

    q.put(None)
    p.join(timeout=10)
    return elapsed, n_frames, times, count.value


# ---------------- patched: SharedMemory ring ----------------------------------

@dataclass
class _Handle:
    name: str
    shape: tuple
    dtype: str


class _Ring:
    def __init__(self, ring_size: int, max_bytes: int, prefix: str):
        self.blocks = [
            shared_memory.SharedMemory(create=True, size=max_bytes, name=f"{prefix}_{i}")
            for i in range(ring_size)
        ]
        self.cursor = 0
        self.size = ring_size

    def claim(self, arr: np.ndarray) -> _Handle:
        b = self.blocks[self.cursor]
        np.ndarray(arr.shape, dtype=arr.dtype, buffer=b.buf)[:] = arr
        h = _Handle(b.name, tuple(arr.shape), str(arr.dtype))
        self.cursor = (self.cursor + 1) % self.size
        return h

    def cleanup(self):
        for b in self.blocks:
            try:
                b.close()
                b.unlink()
            except FileNotFoundError:
                pass


def _shm_writer(q: mp.JoinableQueue, count_value):
    blocks: dict = {}
    while True:
        msg = q.get()
        if msg is None:
            break
        h: _Handle = msg["img"]
        if h.name not in blocks:
            blocks[h.name] = shared_memory.SharedMemory(name=h.name)
        view = np.ndarray(h.shape, dtype=np.dtype(h.dtype), buffer=blocks[h.name].buf)
        out = view.copy()  # copy out so we can reuse the slot
        _ = out.shape
        with count_value.get_lock():
            count_value.value += 1
    for b in blocks.values():
        b.close()


def measure_shm(shape: tuple, n_frames: int, queue_size: int = 16):
    nbytes = int(np.prod(shape))
    ring = _Ring(ring_size=queue_size + 4, max_bytes=nbytes, prefix=f"microbench_{mp.current_process().pid}")
    q = mp.JoinableQueue(queue_size)
    count = mp.Value("i", 0)
    p = mp.Process(target=_shm_writer, args=(q, count))
    p.start()
    img = np.random.randint(0, 256, shape, dtype=np.uint8)

    t0 = time.perf_counter()
    times = []
    for _ in range(n_frames):
        ti = time.perf_counter()
        h = ring.claim(img)
        q.put({"img": h})
        times.append(time.perf_counter() - ti)
    elapsed = time.perf_counter() - t0

    q.put(None)
    p.join(timeout=10)
    ring.cleanup()
    return elapsed, n_frames, times, count.value


# ---------------- driver ------------------------------------------------------

def percentiles(xs, ps=(0.50, 0.95, 0.99)):
    a = np.array(xs) * 1e3  # ms
    return tuple(float(np.quantile(a, p)) for p in ps)


def run(shape: tuple, n_frames: int):
    print(f"\n--- shape {shape} ({np.prod(shape) / 1024:.0f} KB) ---")

    # Warmup
    measure_pickle(shape, 30)
    measure_shm(shape, 30)

    e_p, n_p, t_p, c_p = measure_pickle(shape, n_frames)
    e_s, n_s, t_s, c_s = measure_shm(shape, n_frames)

    p50_p, p95_p, p99_p = percentiles(t_p)
    p50_s, p95_s, p99_s = percentiles(t_s)

    rate_p = n_p / e_p
    rate_s = n_s / e_s

    print(f"  pickle: total={e_p:.2f}s  put p50/p95/p99 = {p50_p:.3f}/{p95_p:.3f}/{p99_p:.3f} ms  "
          f"sustained {rate_p:7.0f} put/s  (writer recv {c_p}/{n_p})")
    print(f"  shm:    total={e_s:.2f}s  put p50/p95/p99 = {p50_s:.3f}/{p95_s:.3f}/{p99_s:.3f} ms  "
          f"sustained {rate_s:7.0f} put/s  (writer recv {c_s}/{n_s})")
    print(f"  speedup (sustained put rate): {rate_s / rate_p:.2f}x")
    print(f"  speedup (p50 put latency):    {p50_p / max(p50_s, 1e-9):.2f}x")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=600, help="Frames per measurement")
    args = p.parse_args()

    print("=" * 70)
    print(" Microbenchmark: pickle vs SharedMemory mp.Queue throughput")
    print("=" * 70)

    for label, shape in SHAPES.items():
        print(f"\n[{label}]", end="")
        run(shape, args.frames)

    print()
    print("=" * 70)
    print(" Higher 'sustained put/s' = better. Speedup > 1 means SHM is faster.")
    print("=" * 70)


if __name__ == "__main__":
    main()
