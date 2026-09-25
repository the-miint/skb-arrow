import argparse
import sys
from importlib.metadata import version
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skb-arrow", color=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--version", action="store_true", help="print version, protocol, capabilities"
    )
    mode.add_argument(
        "--segment-dir", type=Path, metavar="DIR", help="serve, with segments in DIR"
    )
    args = parser.parse_args(argv)
    if args.segment_dir is not None:
        from skb_arrow import host

        return host.serve(args.segment_dir, *host.reserve())
    if not args.version:
        parser.print_usage(sys.stderr)
        return 2
    # Here, not at module level: nothing heavy may load before host.reserve().
    from skb_arrow import registry
    from skb_arrow.protocol import PROTOCOL_VERSION

    caps = ", ".join(f"{n}/{v}" for n, v in registry.schema_versions().items())
    print(f"skb-arrow {version('skb-arrow')}")
    print(f"protocol {PROTOCOL_VERSION}")
    print(f"capabilities: {caps}")
    return 0
