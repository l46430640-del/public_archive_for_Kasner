"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .verification import verify


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="python -m kasner_scattering")
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="verify committed formulas and results")
    rebuild_parser = commands.add_parser("rebuild", help="recompute a prescribed study and compare frozen observations")
    rebuild_parser.add_argument("--study", required=True, choices=("reference", "directed", "exchange", "wall", "carrier", "restricted-wall", "all"))
    rebuild_parser.add_argument("--output", type=Path, required=True)
    plot_parser = commands.add_parser("plot", help="generate the public result figure")
    plot_parser.add_argument("--output", type=Path, required=True)
    compare_parser = commands.add_parser("compare", help="compare an already completed rebuild")
    compare_parser.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "verify":
        result = verify()
    elif arguments.command == "rebuild":
        from .propagation.studies import run
        from .comparison import compare
        run(arguments.study, arguments.output)
        result = compare(arguments.output)
    elif arguments.command == "plot":
        from .plotting import plot_results
        result = {"files": [path.name for path in plot_results(arguments.output)]}
    else:
        from .comparison import compare
        result = compare(arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
