import errno
import sys
import traceback
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO, TypedDict


class HostError(Exception):
    kind = "internal"


class HostIncompatible(HostError):
    kind = "host_incompatible"


class InvalidInput(HostError):
    kind = "invalid_input"


class InvalidParam(HostError):
    kind = "invalid_param"


class Unsupported(HostError):
    kind = "unsupported"


class WarningEntry(TypedDict):
    category: str
    message: str
    count: int


_RESOURCE_ERRNOS = {errno.ENOSPC, errno.EDQUOT, errno.ENOMEM}
# Qualified names, so the classifier imports no capability library.
_BY_TYPE = {
    "builtins.ValueError": "invalid_input",
    "builtins.KeyError": "invalid_param",
    "patsy.PatsyError": "invalid_param",
    "builtins.ImportError": "unsupported",
}
_MAX_WARNINGS = 20


def classify(exc: Exception, *, capability: bool) -> dict[str, str]:
    """Error fields for `exc`; `capability`: it escaped a capability's run."""
    kind = _kind(exc, capability)
    text = _text(exc)
    message = text if isinstance(exc, HostError) else f"{type(exc).__name__}: {text}"
    fields = {"kind": kind, "message": message}
    if kind == "internal":
        fields["traceback"] = "".join(traceback.format_exception(exc))
    return fields


def _kind(exc: Exception, capability: bool) -> str:
    if isinstance(exc, HostError):
        return exc.kind
    if isinstance(exc, MemoryError) or (
        isinstance(exc, OSError) and exc.errno in _RESOURCE_ERRNOS
    ):
        return "resource"
    if not capability or _raised_in_skb_arrow(exc):
        return "internal"
    for cls in type(exc).__mro__:
        if kind := _BY_TYPE.get(_qualname(cls)):
            return kind
    return "internal"


def _raised_in_skb_arrow(exc: Exception) -> bool:
    tb = exc.__traceback__
    if tb is None:
        return False
    while tb.tb_next is not None:
        tb = tb.tb_next
    name = str(tb.tb_frame.f_globals.get("__name__", ""))
    return name == "skb_arrow" or name.startswith("skb_arrow.")


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _text(obj: object) -> str:
    try:
        return str(obj)
    except Exception:
        return f"<{type(obj).__name__}: str() failed>"


def _category(cls: type[Warning]) -> str:
    return cls.__qualname__ if cls.__module__ == "builtins" else _qualname(cls)


@contextmanager
def collect_warnings() -> Iterator[list[WarningEntry]]:
    """Record warnings; the yielded list is filled when the block exits."""
    if sys.flags.context_aware_warnings:
        # Filters then live in a context variable that no public API exposes.
        raise RuntimeError("context-aware warnings are unsupported")
    entries: list[WarningEntry] = []
    counts: dict[tuple[str, str], int] = {}
    overflow = 0

    def count(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        nonlocal overflow
        key = (_category(category), str(message))
        if key in counts or len(counts) < _MAX_WARNINGS:
            counts[key] = counts.get(key, 0) + 1
        else:
            overflow += 1

    with warnings.catch_warnings():
        # Order is kept so the first matching filter still wins; every action but
        # `ignore` becomes `always`, so repeats are counted and `error` never raises.
        warnings.filters = [
            f if f[0] == "ignore" else ("always", f[1], f[2], f[3], f[4])
            for f in warnings.filters
        ]
        warnings.simplefilter("always", append=True)
        warnings.showwarning = count
        try:
            yield entries
        finally:
            entries.extend(
                WarningEntry(category=c, message=m, count=n)
                for (c, m), n in counts.items()
            )
            if overflow:
                entries.append(
                    WarningEntry(
                        category="skb_arrow.WarningsTruncated",
                        message=f"warnings beyond the first {_MAX_WARNINGS} distinct",
                        count=overflow,
                    )
                )
