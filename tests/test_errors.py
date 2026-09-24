import errno
import warnings

import pytest

from skb_arrow.errors import (
    HostIncompatible,
    InvalidInput,
    InvalidParam,
    Unsupported,
    classify,
    collect_warnings,
)

EXPLICIT = [
    (HostIncompatible, "host_incompatible"),
    (InvalidInput, "invalid_input"),
    (InvalidParam, "invalid_param"),
    (Unsupported, "unsupported"),
]
RESOURCE = [
    MemoryError(),
    OSError(errno.ENOSPC, "no space"),
    OSError(errno.EDQUOT, "quota"),
    OSError(errno.ENOMEM, "no memory"),
]


def raised_in(module: str, exc: Exception) -> Exception:
    """Raise `exc` from a frame whose module is `module`, as a library frame would."""
    try:
        exec(
            compile("raise exc", f"<{module}>", "exec"),
            {"__name__": module, "exc": exc},
        )
    except Exception as caught:
        return caught
    raise AssertionError("did not raise")


class PatsyError(Exception):
    pass


PatsyError.__module__ = "patsy"


class LookalikeError(Exception):
    pass


LookalikeError.__qualname__ = "PatsyError"


class LibWarning(UserWarning):
    pass


LibWarning.__module__ = "somelib.sub"


@pytest.mark.parametrize("capability", [True, False])
@pytest.mark.parametrize(("cls", "kind"), EXPLICIT)
def test_explicit_classes_carry_their_kind_even_from_skb_arrow_frames(
    cls: type[Exception], kind: str, capability: bool
) -> None:
    exc = raised_in("skb_arrow.capabilities.x", cls("why"))
    assert classify(exc, capability=capability) == {"kind": kind, "message": "why"}


@pytest.mark.parametrize("capability", [True, False])
@pytest.mark.parametrize("module", ["skb_arrow.transport", "somelib"])
@pytest.mark.parametrize("exc", RESOURCE, ids=repr)
def test_resource_exhaustion_is_resource_from_any_frame(
    exc: Exception, module: str, capability: bool
) -> None:
    assert classify(raised_in(module, exc), capability=capability)["kind"] == "resource"


def test_other_oserrors_are_not_resource() -> None:
    exc = raised_in("somelib", OSError(errno.ENOENT, "missing"))
    assert classify(exc, capability=True)["kind"] == "internal"


def test_capability_code_raising_itself_is_a_bug() -> None:
    result = classify(
        raised_in("skb_arrow.capabilities.x", ValueError("bad")), capability=True
    )
    assert result["kind"] == "internal"
    assert "ValueError: bad" in result["traceback"]


def test_frame_rule_matches_the_package_not_a_name_prefix() -> None:
    exc = raised_in("skb_arrowish", ValueError("bad"))
    assert classify(exc, capability=True)["kind"] == "invalid_input"


class ArrowLikeError(ValueError):
    pass


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (ValueError("x"), "invalid_input"),
        (TypeError("x"), "invalid_input"),
        (ArrowLikeError("x"), "invalid_input"),
        (KeyError("x"), "invalid_param"),
        (PatsyError("x"), "invalid_param"),
        (ImportError("x"), "unsupported"),
        (ModuleNotFoundError("x"), "unsupported"),
        (LookalikeError("x"), "internal"),
        (RuntimeError("x"), "internal"),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, Exception) else v,
)
def test_third_party_raises_map_by_type_along_the_mro(
    exc: Exception, kind: str
) -> None:
    result = classify(raised_in("somelib", exc), capability=True)
    assert result["kind"] == kind
    assert ("traceback" in result) == (kind == "internal")


def test_message_names_the_type_for_non_skb_arrow_errors() -> None:
    result = classify(raised_in("somelib", ValueError("bad")), capability=True)
    assert result["message"] == "ValueError: bad"


def test_machinery_errors_are_internal_unless_explicit() -> None:
    result = classify(raised_in("somelib", ValueError("bad")), capability=False)
    assert result["kind"] == "internal"
    assert "traceback" in result


def warn(message: str, category: type[Warning] = UserWarning) -> None:
    warnings.warn(message, category, stacklevel=1)


def test_every_occurrence_is_counted() -> None:
    with collect_warnings() as recorded:
        for _ in range(3):
            warn("same line")
    assert recorded == [{"category": "UserWarning", "message": "same line", "count": 3}]


def test_ignore_filters_in_effect_are_respected() -> None:
    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="noisy")
        with collect_warnings() as recorded:
            warn("old api", DeprecationWarning)
            warn("noisy library chatter")
            warn("kept")
    assert [w["message"] for w in recorded] == ["kept"]


def test_error_filters_do_not_raise_inside_a_call() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with collect_warnings() as recorded:
            warn("would have raised")
    assert recorded[0]["count"] == 1


def test_category_is_bare_for_builtins_and_qualified_otherwise() -> None:
    with collect_warnings() as recorded:
        warn("a", FutureWarning)
        warn("b", LibWarning)
    assert [w["category"] for w in recorded] == [
        "FutureWarning",
        "somelib.sub.LibWarning",
    ]


def test_dedupe_keeps_first_seen_order() -> None:
    with collect_warnings() as recorded:
        for message in ["b", "a", "b"]:
            warn(message)
    assert [(w["message"], w["count"]) for w in recorded] == [("b", 2), ("a", 1)]


def test_past_twenty_entries_one_truncation_entry_counts_the_rest() -> None:
    with collect_warnings() as recorded:
        for i in range(20):
            warn(f"kept {i}")
        for i, repeats in enumerate([1, 2, 3]):
            for _ in range(repeats):
                warn(f"dropped {i}")
    assert len(recorded) == 21
    assert recorded[-1] == {
        "category": "skb_arrow.WarningsTruncated",
        "message": "3 more distinct warnings",
        "count": 6,
    }


def test_warnings_are_kept_when_the_call_fails() -> None:
    with pytest.raises(RuntimeError), collect_warnings() as recorded:
        warn("before failure")
        raise RuntimeError
    assert recorded[0]["message"] == "before failure"


def test_collection_leaves_filters_as_they_were() -> None:
    before = list(warnings.filters)
    with collect_warnings():
        warn("x")
    assert warnings.filters == before
