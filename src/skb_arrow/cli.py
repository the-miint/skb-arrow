import argparse
import sys
from importlib.metadata import version

from skb_arrow import registry
from skb_arrow.protocol import PROTOCOL_VERSION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skb-arrow")
    parser.add_argument(
        "--version", action="store_true", help="print version, protocol, capabilities"
    )
    if not parser.parse_args(argv).version:
        parser.print_usage(sys.stderr)
        return 2
    caps = ", ".join(f"{n}/{v}" for n, v in sorted(registry.CAPABILITIES.items()))
    print(f"skb-arrow {version('skb-arrow')}")
    print(f"protocol {PROTOCOL_VERSION}")
    print(f"capabilities: {caps or 'none'}")
    return 0
