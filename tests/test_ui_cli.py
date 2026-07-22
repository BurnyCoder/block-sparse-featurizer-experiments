"""CPU-only tests for UI adapters and command-line parsing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading

import gradio as gr
import pytest

from bsf_experiments.cli import parse_reproduce_args, reproduce_main
from bsf_experiments.config import AppConfig
from bsf_experiments.ui import (
    CONCEPT_HEADERS,
    WorkbenchController,
    build_app,
    format_training_event,
    normalize_upload_paths,
    preset_control_values,
    safe_ui_error_message,
    ui_error,
)
from bsf_experiments.types import ModelConfig, TrainingConfig, TrainingEvent


class _TouchingRegistry:
    """Count activity refreshes made by a streaming worker controller."""

    def __init__(self) -> None:
        self.touches = 0

    def touch(self, _session_id: str) -> None:
        """Record one TTL refresh without needing a real experiment session."""

        self.touches += 1


class _BlockingPipeline:
    """Deterministic pipeline double whose trainer stops only when cancelled."""

    def __init__(self, *, fail_log_read: bool = False) -> None:
        self.registry = _TouchingRegistry()
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.cancel_calls = 0
        self.fail_log_read = fail_log_read
        self.state = SimpleNamespace(
            images=object(),
            activations=object(),
            preprocessed_activations=object(),
            metrics={},
            concepts=[],
        )

    def train(self, _session_id, _config, *, progress_callback) -> None:
        """Emit progress, then remain alive until controller teardown cancels it."""

        progress_callback(TrainingEvent(1, 2, 0.5, 0.4, 2.0, 0, False, "running"))
        self.started.set()
        assert self.release.wait(timeout=3), "controller did not cancel its worker"
        self.state.metrics["cancelled"] = True
        self.finished.set()

    def cancel_training(self, _session_id: str) -> bool:
        """Release the fake trainer and record cooperative cancellation."""

        self.cancel_calls += 1
        self.release.set()
        return True

    def read_log(self, _session_id: str) -> str:
        """Return a harmless tail or trigger the controller's error cleanup path."""

        if self.fail_log_read:
            raise RuntimeError("log read failed")
        return "worker log"

    def snapshot(self, _session_id: str) -> SimpleNamespace:
        """Return the mutable fake state used by the full-pipeline generator."""

        return self.state

    def initialize_model(self, _session_id: str, _model: ModelConfig) -> None:
        """Satisfy the full-pipeline phase immediately before blocking training."""

    def encode(self, *_args, **_kwargs) -> None:
        """Remain unreachable after the fake trainer reports cancellation."""

        raise AssertionError("encoding must not run after cancellation")

    def evaluate(self, *_args, **_kwargs) -> None:
        """Remain unreachable after the fake trainer reports cancellation."""

        raise AssertionError("evaluation must not run after cancellation")

    def rank(self, *_args, **_kwargs) -> None:
        """Remain unreachable after the fake trainer reports cancellation."""

        raise AssertionError("ranking must not run after cancellation")


def test_normalize_upload_paths_accepts_current_gradio_file_shapes() -> None:
    """Uploaded values become paths whether Gradio supplies strings or FileData."""

    assert normalize_upload_paths(None) == ()
    assert normalize_upload_paths("one.png") == (Path("one.png"),)
    values = ["one.png", SimpleNamespace(path="two.jpg")]
    assert normalize_upload_paths(values) == (Path("one.png"), Path("two.jpg"))


def test_preset_values_cover_model_and_training_controls() -> None:
    """A preset updates every user-editable model/trainer field coherently."""

    values = preset_control_values("group_lasso_notebook")
    assert values[0] == "group_lasso"
    assert values[1:4] == (256, 3, 16)
    assert values[4] == pytest.approx(1e-2)
    assert values[5] == 8
    assert values[8] == 300


def test_progress_format_is_complete_and_readable() -> None:
    """The live status exposes all four requested training measurements."""

    text = format_training_event(TrainingEvent(2, 5, 0.3, 0.75, 8.0, 2, False, "ok"))
    assert "2/5" in text
    assert "R²=0.7500" in text
    assert "L0=8.000" in text
    assert "dead groups=2" in text
    assert len(CONCEPT_HEADERS) == 5


def test_browser_error_messages_are_secret_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lower-layer credential-bearing failure cannot leak through ``gr.Error``."""

    monkeypatch.setenv("HF_TOKEN", "fabricated-browser-only-secret")

    message = safe_ui_error_message(
        RuntimeError("request failed for fabricated-browser-only-secret")
    )

    assert "fabricated-browser-only-secret" not in message
    assert "[REDACTED]" in message


def test_ui_error_suppresses_sensitive_exception_print_and_chaining(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The actual browser exception is redacted, silent, and cause-suppressed."""

    secret = "fabricated-ui-handler-secret"
    monkeypatch.setenv("HF_TOKEN", secret)
    lower_error = RuntimeError(f"authorization failed for {secret}")

    with pytest.raises(gr.Error) as raised:
        ui_error(lower_error)

    browser_error = raised.value
    assert secret not in browser_error.message
    assert "[REDACTED]" in browser_error.message
    assert browser_error.print_exception is False
    assert browser_error.__cause__ is None
    assert browser_error.__suppress_context__ is True
    assert capsys.readouterr().err == ""


def test_training_stream_close_cancels_touches_and_fully_joins_worker() -> None:
    """Closing a partial training stream cannot release its GPU slot early."""

    pipeline = _BlockingPipeline()
    stream = WorkbenchController(pipeline).training_stream("session", TrainingConfig())

    assert "epoch 1/2" in next(stream)[0]
    assert pipeline.started.wait(timeout=1)
    stream.close()

    assert pipeline.cancel_calls == 1
    assert pipeline.finished.is_set()
    assert pipeline.registry.touches > 0
    assert not any(
        thread.name == "bsf-training-session" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_training_stream_error_cancels_and_fully_joins_worker() -> None:
    """An output-processing failure tears down the still-running trainer."""

    pipeline = _BlockingPipeline(fail_log_read=True)
    stream = WorkbenchController(pipeline).training_stream("session", TrainingConfig())

    with pytest.raises(RuntimeError, match="log read failed"):
        next(stream)

    assert pipeline.cancel_calls == 1
    assert pipeline.finished.is_set()
    assert pipeline.registry.touches > 0


def test_pipeline_stream_close_cancels_touches_and_fully_joins_worker() -> None:
    """The full-pipeline generator owns its background worker through close."""

    pipeline = _BlockingPipeline()
    stream = WorkbenchController(pipeline).pipeline_stream(
        "session",
        ModelConfig(n_groups=2, group_size=1, l0=1),
        TrainingConfig(),
        extraction_batch_size=2,
    )

    assert next(stream)[0] == "Initializing featurizer…"
    assert pipeline.started.wait(timeout=1)
    stream.close()

    assert pipeline.cancel_calls == 1
    assert pipeline.finished.is_set()
    assert pipeline.registry.touches > 0
    assert not any(
        thread.name == "bsf-pipeline-session" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_app_contract_contains_every_action_and_one_serial_gpu_queue(
    tmp_path: Path,
) -> None:
    """The rendered Blocks graph exposes the plan without parallel GPU mutation."""

    config = AppConfig(
        project_root=tmp_path,
        env_file=tmp_path / ".env",
        host="127.0.0.1",
        port=7860,
        output_dir=tmp_path / "outputs",
        log_level="INFO",
        max_upload_mb=4,
        session_ttl_seconds=60,
        device="cpu",
        hf_token_available=False,
    )
    app = build_app(config)
    try:
        assert len(app.bsf_pipeline.registry) == 0
        graph = app.get_config_file()
        buttons = {
            component["props"].get("value")
            for component in graph["components"]
            if component.get("type") == "button"
        }
        assert {
            "README Quickstart",
            "Grassmannian Notebook",
            "Group Lasso Notebook",
            "Vanilla Notebook",
            "Run Current Pipeline",
            "Check Environment",
            "Load Rabbits",
            "Load NPZ",
            "Load Uploaded Images",
            "Extract DINO Activations",
            "Center & Scale",
            "Initialize Model",
            "Train",
            "Stop Training",
            "Encode Features",
            "Reconstruct & Evaluate",
            "Rank Concepts",
            "Select Top N",
            "Render Concept Plot",
            "Save Checkpoint",
            "Load Checkpoint",
            "Export Results Bundle",
            "Export Arrays",
            "Reset Session",
        } <= buttons
        assert "per_axis_rgb" not in json.dumps(graph, default=str)

        load_dependencies = [
            dependency
            for dependency in graph["dependencies"]
            if any(target[1] == "load" for target in dependency["targets"])
        ]
        assert len(load_dependencies) == 1
        create_session = app.fns[load_dependencies[0]["id"]].fn
        first_session, first_log = create_session()
        second_session, second_log = create_session()
        assert first_session != second_session
        assert first_log != second_log
        assert first_log.is_file()
        assert second_log.is_file()
        assert len(app.bsf_pipeline.registry) == 2

        button_ids = {
            component["props"].get("value"): component["id"]
            for component in graph["components"]
            if component.get("type") == "button"
        }
        stop_dependencies = [
            dependency
            for dependency in graph["dependencies"]
            if (button_ids["Stop Training"], "click") in dependency["targets"]
        ]
        assert any(
            dependency["queue"] is False and dependency["cancels"] == []
            for dependency in stop_dependencies
        )

        load_checkpoint_dependency = next(
            dependency
            for dependency in graph["dependencies"]
            if (button_ids["Load Checkpoint"], "click") in dependency["targets"]
        )
        cleared_labels = {
            "Live R²",
            "Live L0",
            "Dead groups",
            "Every learned group",
            "Concept group IDs",
            "Concept manifolds and source overlays",
        }
        cleared_output_ids = {
            component["id"]
            for component in graph["components"]
            if component["props"].get("label") in cleared_labels
        }
        download_output_ids = {
            component["id"]
            for component in graph["components"]
            if component["props"].get("label") in {"Download PNG", "Download PDF"}
        }
        assert cleared_output_ids | download_output_ids <= set(
            load_checkpoint_dependency["outputs"]
        )

        for dependency in graph["dependencies"]:
            if not dependency["queue"] or dependency["targets"][0][1] == "load":
                continue
            backend = app.fns[dependency["id"]]
            assert backend.concurrency_id == "gpu"
            assert backend.concurrency_limit == 1
    finally:
        app.bsf_pipeline.close()


def test_reproduce_cli_parses_target_and_returns_failed_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The console command emits structured JSON and maps suite status to exit code."""

    args = parse_reproduce_args(["--target", "readme", "--output-dir", str(tmp_path)])
    assert args.target == "readme"
    suite = SimpleNamespace(ok=False, to_dict=lambda: {"ok": False, "status": "failed"})
    monkeypatch.setattr("bsf_experiments.cli.run_reproduction", lambda *_a, **_k: suite)

    with pytest.raises(SystemExit) as exit_info:
        reproduce_main(["--target", "readme", "--output-dir", str(tmp_path)])

    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
