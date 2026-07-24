"""Console entry points for exact reproduction and the local Gradio workbench.

The reproduction parser stays independent of Gradio, so environment checks and
notebook runs remain usable on headless systems. Python's ``argparse`` module is
the standard-library command parser: https://docs.python.org/3/library/argparse.html.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import NoReturn

from .reproduction import DEFAULT_OUTPUT_ROOT, run_reproduction
from .types import ModelSource, PretrainedRecipe


def _positive_integer(value: str) -> int:
    """Parse a positive command-line integer with an actionable error."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _resolved_path(value: str) -> Path:
    """Normalize an output directory without creating it during argument parsing."""

    return Path(value).expanduser().resolve()


def parse_reproduce_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the stable public options for ``uv run bsf-reproduce``."""

    parser = argparse.ArgumentParser(
        prog="bsf-reproduce",
        description="Reproduce the unchanged upstream BSF README and notebooks.",
    )
    parser.add_argument(
        "--target",
        choices=("readme", "notebooks", "all"),
        default="all",
        help="upstream workflow to execute (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=_resolved_path,
        default=DEFAULT_OUTPUT_ROOT,
        help="ignored root for timestamped artifacts (default: outputs/runs)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_integer,
        default=7_200,
        help="per-notebook timeout in seconds (default: 7200)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="do not require CUDA during preflight (DINO examples remain slow)",
    )
    parser.add_argument(
        "--skip-hf-check",
        action="store_true",
        help="skip gated-model metadata preflight (execution may still need access)",
    )
    return parser.parse_args(argv)


def reproduce_main(argv: Sequence[str] | None = None) -> NoReturn:
    """Run the requested exact workflow, print JSON, and return a shell exit code."""

    arguments = parse_reproduce_args(argv)
    suite = run_reproduction(
        arguments.target,
        output_root=arguments.output_dir,
        timeout=arguments.timeout,
        require_cuda=not arguments.allow_cpu,
        check_hf=not arguments.skip_hf_check,
    )
    print(json.dumps(suite.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
    raise SystemExit(0 if suite.ok else 1)


def parse_ui_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional initial selections for the backwards-compatible UI launcher.

    ``choices`` gives users an immediate command-line error for values outside
    the trusted Hub catalog:
    https://docs.python.org/3/library/argparse.html#choices
    """

    parser = argparse.ArgumentParser(
        prog="bsf-ui",
        description="Launch the local block-sparse featurizer workbench.",
    )
    parser.add_argument(
        "--model-source",
        choices=[source.value for source in ModelSource],
        default=ModelSource.TRAIN.value,
        help="initial model workflow (default: train)",
    )
    parser.add_argument(
        "--pretrained-recipe",
        choices=[recipe.value for recipe in PretrainedRecipe],
        default=PretrainedRecipe.README_QUICKSTART.value,
        help="initial trusted Hugging Face checkpoint recipe",
    )
    arguments = parser.parse_args(argv)
    arguments.model_source = ModelSource(arguments.model_source)
    arguments.pretrained_recipe = PretrainedRecipe(arguments.pretrained_recipe)
    return arguments


def ui_main(argv: Sequence[str] | None = None) -> None:
    """Import the optional presentation layer lazily and launch it locally."""

    from .ui import launch_app

    arguments = parse_ui_args(argv)
    launch_app(
        default_model_source=arguments.model_source,
        default_pretrained_recipe=arguments.pretrained_recipe,
    )


__all__ = [
    "parse_reproduce_args",
    "parse_ui_args",
    "reproduce_main",
    "ui_main",
]
