# skb-arrow

A process host that exposes [scikit-bio](https://scikit.bio) to non-Python callers. Callers
speak JSON lines on stdin/stdout and exchange data as Arrow IPC over memory-mapped files.

```
uv tool install skb-arrow
```

Design: [`docs/DESIGN.md`](docs/DESIGN.md).
