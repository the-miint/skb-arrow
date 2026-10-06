# skb-arrow

A process host that exposes [scikit-bio](https://scikit.bio) to non-Python callers. A caller
runs `skb-arrow`, speaks JSON lines on its stdin and stdout, and exchanges tables as Arrow IPC
in memory-mapped files. Each capability is one curated, typed, versioned scikit-bio operation:
`ancombc` and `mantel` so far.

## Platforms
- Linux x86_64, glibc 2.28 or later
- macOS 12 or later, on Apple silicon

Python 3.14 only. Every dependency installs from a wheel, and each is pinned exactly, so an
answer is reproducible per skb-arrow version.

## Install
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.6.8 or later, then:

```
uv tool install --managed-python --python 3.14 skb-arrow
uv tool update-shell   # if uv says its tool directory isn't on PATH; then open a new shell
skb-arrow --doctor
```

- `--python 3.14`: uv picks a tool's interpreter without checking the versions it supports,
  so with another default Python the install fails, or tries to build dependencies from
  source.
- `--managed-python`: uv's own CPython, not whichever 3.14 the machine has.

`skb-arrow --doctor` reports the interpreter, the platform, the pinned dependencies, numpy's
BLAS, the thread settings, and where segments go, then a `problem:` line per fault. It exits 1
if there is one.

## Use
A caller runs `skb-arrow --segment-dir DIR`. `skb-arrow --version` prints the version, the
protocol version, and each capability with its schema version.
- [Protocol](https://github.com/the-miint/skb-arrow/blob/main/docs/protocol.md): the control
  channel, the handshake, and versioning.
- [Transport](https://github.com/the-miint/skb-arrow/blob/main/docs/transport.md): segments,
  and where DIR goes.
- [Capabilities](https://github.com/the-miint/skb-arrow/blob/main/docs/capabilities.md): each
  capability's inputs, params, and output.
- [Errors](https://github.com/the-miint/skb-arrow/blob/main/docs/errors.md): error kinds and
  warnings.

These follow `main`. A release's own are under its tag:
`https://github.com/the-miint/skb-arrow/tree/v<version>/docs`.

## Threads
Each is read from the environment, and `--doctor` shows it:
- `NUMBA_NUM_THREADS`: numba's kernels (`mantel`); every core by default. Answers don't depend
  on it.
- `OPENBLAS_NUM_THREADS` for OpenBLAS (Linux, and macOS 12 and 13), `VECLIB_MAXIMUM_THREADS`
  for Accelerate (macOS 14 and later): numpy's and scipy's BLAS, which `--doctor` names.
  Answers are bit for bit on one machine at one count; across them, to floating-point
  tolerance.
- `OMP_NUM_THREADS`: OpenMP's.

## Containers
On Linux, DIR lives in `/dev/shm`. Docker's default 64 MiB `/dev/shm` holds less than one
default 256 MiB segment, and a call's inputs and outputs share DIR: run the container with a
larger `--shm-size`, or on Kubernetes mount an `emptyDir` with `medium: Memory` at `/dev/shm`.

## License
BSD-3-Clause.
