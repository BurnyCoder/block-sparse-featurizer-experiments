"""Command-level contracts for the lightweight reproduction entry point."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bsf_experiments.cli import parse_reproduce_args, reproduce_main


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
