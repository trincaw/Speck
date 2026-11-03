"""Entry point of the `speck` command: loads configuration, builds the parser, dispatches."""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

from speck import __version__
from speck.cli.commands import MODULES
from speck.cli.options import Options
from speck.config import load_env_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speck",
        description="Speck - model-agnostic speech recognition with real-time streaming and live speaker labels",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="More logging (-v info, -vv debug)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    options = Options()
    for module in MODULES:
        module.register(sub, options)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    load_env_file()  # before building the parser: option defaults read the environment
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=(logging.WARNING, logging.INFO, logging.DEBUG)[min(args.verbose, 2)],
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit(130)
