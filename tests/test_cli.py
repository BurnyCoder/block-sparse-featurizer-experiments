"""Command-level contracts for the lightweight reproduction entry point."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bsf_experiments.cli import (
    parse_reproduce_args,
    parse_ui_args,
    reproduce_main,
    ui_main,
)
from bsf_experiments.types import ModelSource, PretrainedRecipe


def test_reproduce_cli_forwards_options_and_returns_failed_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The public command emits JSON and maps a failed suite to exit status one."""

    captured: dict[str, object] = {}

    def fake_run(target: str, **options: object) -> SimpleNamespace:
        captured.update(target=target, **options)
        return SimpleNamespace(
            ok=False,
            to_dict=lambda: {"ok": False, "status": "failed"},
        )

    monkeypatch.setattr("bsf_experiments.cli.run_reproduction", fake_run)
    arguments = [
        "--target",
        "readme",
        "--output-dir",
        str(tmp_path),
        "--timeout",
        "90",
        "--allow-cpu",
        "--skip-hf-check",
    ]

    parsed = parse_reproduce_args(arguments)
    assert parsed.target == "readme"
    with pytest.raises(SystemExit) as exit_info:
        reproduce_main(arguments)

    assert exit_info.value.code == 1
    assert captured == {
        "target": "readme",
        "output_root": tmp_path.resolve(),
        "timeout": 90,
        "require_cuda": False,
        "check_hf": False,
    }
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_reproduce_cli_rejects_nonpositive_timeout() -> None:
    """Invalid runtime limits fail during parsing with argparse's status two."""

    with pytest.raises(SystemExit) as exit_info:
        parse_reproduce_args(["--timeout", "0"])

    assert exit_info.value.code == 2


def test_ui_cli_defaults_preserve_local_training() -> None:
    """The existing no-argument launcher still opens in local training mode."""

    arguments = parse_ui_args([])

    assert arguments.model_source is ModelSource.TRAIN
    assert arguments.pretrained_recipe is PretrainedRecipe.README_QUICKSTART


def test_ui_cli_forwards_hub_startup_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launcher can preselect one pinned Hub recipe without an auth flag."""

    captured: dict[str, object] = {}

    def fake_launch(**options: object) -> None:
        captured.update(options)

    monkeypatch.setattr("bsf_experiments.ui.launch_app", fake_launch)

    ui_main(
        [
            "--model-source",
            "hugging_face",
            "--pretrained-recipe",
            "group_lasso_notebook",
        ]
    )

    assert captured == {
        "default_model_source": ModelSource.HUGGING_FACE,
        "default_pretrained_recipe": PretrainedRecipe.GROUP_LASSO_NOTEBOOK,
    }


def test_ui_cli_rejects_unknown_recipe() -> None:
    """Argparse rejects identifiers outside the static pretrained catalog."""

    with pytest.raises(SystemExit) as exit_info:
        parse_ui_args(["--pretrained-recipe", "latest"])

    assert exit_info.value.code == 2
