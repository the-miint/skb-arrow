import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skb-arrow", color=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--version", action="store_true", help="print version, protocol, capabilities"
    )
    mode.add_argument(
        "--doctor", action="store_true", help="report on this install; 1 if unsound"
    )
    mode.add_argument(
        "--segment-dir", type=Path, metavar="DIR", help="serve, with segments in DIR"
    )
    args = parser.parse_args(argv)
    if args.segment_dir is not None:
        from skb_arrow import host

        return host.serve(args.segment_dir, *host.reserve())
    if not (args.version or args.doctor):
        parser.print_usage(sys.stderr)
        return 2
    # Here, not at module level: nothing heavy may load before host.reserve().
    from skb_arrow import doctor

    if args.doctor:
        return doctor.report()
    print("\n".join(doctor.versions()))
    return 0
