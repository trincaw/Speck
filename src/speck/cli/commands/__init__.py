"""Subcommands, grouped by domain. Each module exposes `register(subparsers, options)`."""

from speck.cli.commands import evaluate, info, live, setup, speakers, transcribe

MODULES = (setup, info, transcribe, live, speakers, evaluate)
