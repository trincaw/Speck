"""Subcommands, grouped by domain. Each module exposes `register(subparsers, options)`."""

from speck.cli.commands import evaluate, info, live, speakers, transcribe

MODULES = (info, transcribe, live, speakers, evaluate)
