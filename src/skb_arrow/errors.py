import errno
import traceback
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypedDict, cast


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
# Qualified names, so optional libraries (patsy) are never imported.
_BY_TYPE = {
    "builtins.ValueError": "invalid_input",
    "builtins.TypeError": "invalid_input",
    "builtins.KeyError": "invalid_param",
    "patsy.PatsyError": "invalid_param",
    "builtins.ImportError": "unsupported",
}
_MAX_WARNINGS = 20


def classify(exc: Exception, *, capability: bool) -> dict[str, str]:
    """Error fields for `exc`; `capability`: it escaped a capability's run."""
    kind = _kind(exc, capability)
    message = str(exc) if isinstance(exc, HostError) else f"{type(exc).__name__}: {exc}"
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


@contextmanager
def collect_warnings() -> Iterator[list[WarningEntry]]:
    """Record warnings; the yielded list is filled when the block exits."""
    entries: list[WarningEntry] = []
    ignores = [f for f in warnings.filters if f[0] == "ignore"]
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        for _, message, category, module, lineno in reversed(ignores):
            warnings.filterwarnings(
                "ignore",
                message.pattern if message else "",
                cast(type[Warning], category),
                module.pattern if module else "",
                lineno,
            )
        try:
            yield entries
        finally:
            entries.extend(_summarize(log))


def _summarize(log: list[warnings.WarningMessage]) -> list[WarningEntry]:
    counts: dict[tuple[str, str], int] = {}
    for w in log:
        category = w.category.__qualname__
        if w.category.__module__ != "builtins":
            category = _qualname(w.category)
        key = (category, str(w.message))
        counts[key] = counts.get(key, 0) + 1
    entries = [
        WarningEntry(category=c, message=m, count=n) for (c, m), n in counts.items()
    ]
    if len(entries) <= _MAX_WARNINGS:
        return entries
    dropped = entries[_MAX_WARNINGS:]
    return [
        *entries[:_MAX_WARNINGS],
        WarningEntry(
            category="skb_arrow.WarningsTruncated",
            message=f"{len(dropped)} more distinct warnings",
            count=sum(e["count"] for e in dropped),
        ),
    ]
