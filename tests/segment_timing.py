"""Read times across segment caps: the default cap's evidence (DESIGN §3.15).

python tests/segment_timing.py [--purge] >> "$GITHUB_STEP_SUMMARY"

Each payload is written at each cap where a caller places DIR; each read runs in a fresh
process after `sync`, and with --purge (macOS) after `purge` too, so it starts cold.
Prints Markdown: segments written and median read times by cap (framing counts, so a cap
holds fewer batches than it divides), and the cap the rule fixed in DESIGN §3.15 picks.
"""

import argparse
import itertools
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from client import PLACE
from pyarrow import ipc

from skb_arrow import transport

MiB, GiB = 1 << 20, 1 << 30
PAYLOADS = [1 * GiB, 2 * GiB]
CAPS: list[int | None] = [16 * MiB, 64 * MiB, 256 * MiB, 1024 * MiB, None]  # None: one
REPEATS = 3
MODES = ["whole", "streamed"]


def payload(total: int) -> pa.Table:
    """`total` bytes of int64, in 4 MiB batches."""
    step = 4 * MiB // 8
    return pa.Table.from_batches(
        [
            pa.record_batch(
                {"n": np.arange(i, min(i + step, total // 8), dtype=np.int64)}
            )
            for i in range(0, total // 8, step)
        ]
    )


def read(mode: str, paths: list[str]) -> float:
    """Seconds to read `paths`, touching every value: all at once, or by segment."""
    start = time.perf_counter()
    if mode == "whole":
        batches = [b for p in paths for b in ipc.open_stream(pa.memory_map(p))]
        pc.sum(pa.Table.from_batches(batches)["n"])
    else:
        for path in paths:
            for batch in ipc.open_stream(pa.memory_map(path)):
                pc.sum(batch["n"])
    return time.perf_counter() - start


def timed(mode: str, paths: list[str], purge: bool) -> float:
    os.sync()
    if purge:
        subprocess.run(["sudo", "purge"], check=True)
    child = subprocess.run(
        [sys.executable, __file__, "--read", mode, *paths],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(child.stdout)


def pick(medians: dict[int | None, float]) -> str:
    """The largest cap ≤ 512 MiB within 1.1× of the fastest, preferring 256 MiB."""
    fastest = min(medians.values())
    fit = [c for c, t in medians.items() if c and c <= 512 * MiB and t <= 1.1 * fastest]
    if not fit:
        return "none"
    return label(256 * MiB if 256 * MiB in fit else max(fit))


def label(cap: int | None) -> str:
    return f"{cap // MiB} MiB" if cap else "one segment"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--purge", action="store_true", help="evict caches (macOS)")
    parser.add_argument("--read", nargs="+", metavar=("MODE", "PATH"))
    args = parser.parse_args()
    if args.read:
        print(read(args.read[0], args.read[1:]))
        return
    print(f"### Segment read times: {sys.platform}, in {PLACE}\n")
    reads = "cold" if args.purge else "as cached"
    print(f"Median of {REPEATS}, seconds; reads {reads}.\n")
    for total in PAYLOADS:
        table = payload(total)
        times: dict[str, dict[int | None, float]] = {m: {} for m in MODES}
        segments: dict[int | None, int] = {}
        for cap in CAPS:
            directory = Path(tempfile.mkdtemp(dir=PLACE))
            try:
                names = transport.write(
                    directory, table, cap or 1 << 62, itertools.count()
                )
                segments[cap] = len(names)
                paths = [str(directory / name) for name in names]
                for mode in MODES:
                    runs = [timed(mode, paths, args.purge) for _ in range(REPEATS)]
                    times[mode][cap] = statistics.median(runs)
            finally:
                shutil.rmtree(directory)
        del table
        print(f"| {total // GiB} GiB | segments | " + " | ".join(MODES) + " |")
        print("|---|---|" + "---|" * len(MODES))
        for cap in CAPS:
            cells = " | ".join(f"{times[m][cap]:.3f}" for m in MODES)
            print(f"| {label(cap)} | {segments[cap]} | {cells} |")
        picks = ", ".join(f"{m}: {pick(times[m])}" for m in MODES)
        print(f"\nThe rule picks {picks}.\n")


if __name__ == "__main__":
    main()
