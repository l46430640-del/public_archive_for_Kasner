"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import plot_results, rebuild, run_wolfram, verify


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="python -m kasner_scattering")
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="verify committed formulas and results")
    rebuild_parser = commands.add_parser("rebuild", help="recompute all headline results")
    rebuild_parser.add_argument("--output", type=Path, required=True)
    plot_parser = commands.add_parser("plot", help="generate the public result figure")
    plot_parser.add_argument("--output", type=Path, required=True)
    commands.add_parser(
        "crosscheck-wolfram",
        help="run the optional independent Wolfram Language calculation",
    )
    return value


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "verify":
        result = verify()
    elif arguments.command == "rebuild":
        result = rebuild(arguments.output)
    elif arguments.command == "plot":
        result = {"files": [str(path) for path in plot_results(arguments.output)]}
    else:
        result = run_wolfram()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

