# skb-arrow

A process host that exposes [scikit-bio](https://scikit.bio) to non-Python callers. Callers
speak JSON lines on stdin/stdout and exchange data as Arrow IPC over memory-mapped files.

Not yet on PyPI. From a checkout:

```
uv tool install .
```

Design: [`docs/DESIGN.md`](docs/DESIGN.md).
