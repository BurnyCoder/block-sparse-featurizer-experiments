"""Tests for the framework-neutral experiment orchestration boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from bsf_experiments.config import AppConfig
from bsf_experiments.pipeline import ExperimentPipeline
from bsf_experiments.sessions import SessionRegistry
from bsf_experiments.types import (
    DatasetConfig,
    DatasetKind,
    ExperimentStage,
    ModelConfig,
    PlotConfig,
    TrainingConfig,
    TrainingEvent,
)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    """Create a local-only, credential-free app configuration."""

    return AppConfig(
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


@pytest.fixture
def pipeline(app_config: AppConfig) -> ExperimentPipeline:
    """Return a pipeline whose registry is visible to state assertions."""

    instance = ExperimentPipeline(
        app_config,
        registry=SessionRegistry(app_config.session_ttl_seconds),
    )
    yield instance
    instance.close()


def test_session_state_is_server_side_and_phase_order_is_guarded(
    pipeline: ExperimentPipeline,
) -> None:
    """Only an opaque ID crosses the UI boundary and actions explain prerequisites."""

    session_id = pipeline.create_session()
    assert isinstance(session_id, str)
    assert pipeline.snapshot(session_id).stage is ExperimentStage.EMPTY
    with pytest.raises(ValueError, match="Load image data"):
        pipeline.extract_activations(session_id)


def test_stable_session_id_is_persisted_after_expiry_and_log_reads_touch_ttl(
    app_config: AppConfig,
) -> None:
    """A browser-held ID survives TTL recreation and log polling keeps it live."""

    now = [0.0]
    registry = SessionRegistry(app_config.session_ttl_seconds, clock=lambda: now[0])
    instance = ExperimentPipeline(app_config, registry=registry)
    try:
        session_id = instance.create_session("browser-session")
        first_session = registry.get(session_id, touch=False)

        now[0] = 61.0
        assert instance.ensure_session(session_id) == session_id
        recreated_session = registry.get(session_id, touch=False)
        assert recreated_session is not first_session
        assert recreated_session.state.artifacts["log"] == instance.log_path(session_id)

        now[0] = 110.0
        instance.read_log(session_id)
        now[0] = 169.0
        assert registry.get(session_id, touch=False) is recreated_session
    finally:
        instance.close()


def test_data_model_and_analysis_workflow_delegates_to_phases(
    pipeline: ExperimentPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper advances state while leaving implementations in phase modules."""

    images = np.ones((2, 4, 4, 3), dtype=np.uint8)
    activations = np.ones((2, 4, 3), dtype=np.float32)
    matrix = np.ones((8, 3), dtype=np.float32)
    codes = np.ones((8, 2, 1), dtype=np.float32)
    atoms = np.ones((2, 1, 3), dtype=np.float32)
    model = SimpleNamespace(d=3, n_groups=2, group_size=1)

    monkeypatch.setattr(
        "bsf_experiments.pipeline.load_dataset_images", lambda *_args, **_kwargs: images
    )
    monkeypatch.setattr(
        "bsf_experiments.pipeline.extract_dino_activations",
        lambda *_args, **_kwargs: activations,
    )
    monkeypatch.setattr(
        "bsf_experiments.pipeline.preprocess_dino_activations",
        lambda *_args, **_kwargs: (matrix, 2),
    )
    monkeypatch.setattr(
        "bsf_experiments.pipeline.create_model", lambda *_args, **_kwargs: model
    )
    monkeypatch.setattr(
        "bsf_experiments.pipeline.encode_features", lambda *_args, **_kwargs: codes
    )
    monkeypatch.setattr(
        "bsf_experiments.pipeline.evaluate_reconstruction",
        lambda *_args, **_kwargs: {"r2": 0.9, "mean_l0": 1.0, "dead_groups": 0},
    )
    monkeypatch.setattr(
        "bsf_experiments.pipeline.model_atoms", lambda *_args, **_kwargs: atoms
    )

    session_id = pipeline.create_session()
    pipeline.load_dataset(session_id, DatasetConfig(DatasetKind.BUNDLED_RABBITS))
    pipeline.extract_activations(session_id)
    pipeline.center_and_scale(session_id)
    pipeline.initialize_model(session_id, ModelConfig(n_groups=2, group_size=1, l0=1))
    pipeline.encode(session_id, device="cpu")
    metrics = pipeline.evaluate(session_id, device="cpu")
    concepts = pipeline.rank(session_id)

    state = pipeline.snapshot(session_id)
    assert state.stage is ExperimentStage.ANALYZED
    assert state.images is images
    assert state.activations is activations
    assert state.preprocessed_activations is matrix
    assert state.codes is codes
    assert state.atoms is atoms
    assert metrics["r2"] == pytest.approx(0.9)
    assert [record.group_id for record in concepts] == [0, 1]


def test_training_forwards_progress_and_honors_session_cancellation(
    pipeline: ExperimentPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training receives the registry token and stores structured live metrics."""

    session_id = pipeline.create_session()
    session = pipeline.registry.get(session_id)
    with session.locked_state() as state:
        state.preprocessed_activations = np.ones((8, 3), dtype=np.float32)
        state.model = SimpleNamespace(d=3)
        state.model_config = ModelConfig(n_groups=2, group_size=1, l0=1)
        state.stage = ExperimentStage.MODEL_READY

    received: list[TrainingEvent] = []

    def fake_train(model, _matrix, _config, *, progress_callback, should_stop):
        assert not should_stop()
        event = TrainingEvent(1, 1, 0.2, 0.8, 1.0, 0, False, "done")
        progress_callback(event)
        return model

    monkeypatch.setattr("bsf_experiments.pipeline.train_model", fake_train)
    pipeline.train(
        session_id,
        replace(TrainingConfig(), epochs=1, batch_size=4, device="cpu"),
        progress_callback=received.append,
    )

    assert received[-1].r2 == pytest.approx(0.8)
    assert pipeline.snapshot(session_id).stage is ExperimentStage.TRAINED
    assert pipeline.snapshot(session_id).metrics["r2"] == pytest.approx(0.8)
    assert pipeline.cancel_training(session_id) is False


def test_training_start_clears_metrics_rejects_overlap_and_cancels_current_token(
    pipeline: ExperimentPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup and cancellation share a lock, and a retrain has no stale metrics."""

    session_id = pipeline.create_session()
    session = pipeline.registry.get(session_id)
    model = SimpleNamespace(d=3)
    with session.locked_state() as state:
        state.preprocessed_activations = np.ones((8, 3), dtype=np.float32)
        state.model = model
        state.model_config = ModelConfig(n_groups=2, group_size=1, l0=1)
        state.metrics = {"r2": 0.99, "dead_groups": 7}
        state.codes = np.ones((8, 2, 1), dtype=np.float32)
        state.atoms = np.ones((2, 1, 3), dtype=np.float32)
        state.stage = ExperimentStage.TRAINED

    started = threading.Event()
    finish = threading.Event()
    errors: list[BaseException] = []

    def blocking_train(model, _matrix, _config, *, progress_callback, should_stop):
        assert pipeline.snapshot(session_id).metrics == {}
        assert pipeline.snapshot(session_id).codes is None
        started.set()
        assert finish.wait(timeout=2)
        assert should_stop()
        progress_callback(TrainingEvent(1, 1, cancelled=True, message="stopped"))
        return model

    def run_training() -> None:
        try:
            pipeline.train(
                session_id,
                replace(TrainingConfig(), epochs=1, batch_size=4, device="cpu"),
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    monkeypatch.setattr("bsf_experiments.pipeline.train_model", blocking_train)
    worker = threading.Thread(target=run_training)
    worker.start()
    assert started.wait(timeout=2)

    with pytest.raises(ValueError, match="already in progress"):
        pipeline.train(
            session_id,
            replace(TrainingConfig(), epochs=1, batch_size=4, device="cpu"),
        )
    assert pipeline.cancel_training(session_id) is True
    finish.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert errors == []
    assert pipeline.snapshot(session_id).metrics == {"cancelled": True}
    assert pipeline.cancel_training(session_id) is False


def test_render_and_exports_use_session_artifact_store(
    pipeline: ExperimentPipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Visualization and exports remain under the configured generated path."""

    session_id = pipeline.create_session()
    state = pipeline.registry.get(session_id).state
    state.images = np.ones((2, 4, 4, 3), dtype=np.uint8)
    state.codes = np.ones((8, 2, 1), dtype=np.float32)
    state.atoms = np.ones((2, 1, 3), dtype=np.float32)
    state.grid = 2
    state.concepts = []
    figure = SimpleNamespace(
        savefig=lambda path, **_kwargs: Path(path).write_bytes(b"x")
    )
    monkeypatch.setattr(
        "bsf_experiments.pipeline.render_concepts", lambda *_args, **_kwargs: figure
    )

    rendered, downloads = pipeline.visualize(
        session_id, [0], PlotConfig(n_img=1, max_points=8)
    )
    assert rendered is figure
    assert set(downloads) == {"png", "pdf"}
    assert all(
        path.is_relative_to(pipeline.config.output_dir) for path in downloads.values()
    )

    state.activations = np.ones((2, 4, 3), dtype=np.float32)
    array_path = pipeline.export_arrays(session_id)
    bundle_path = pipeline.export_results(session_id)
    assert array_path.suffix == ".npz"
    assert bundle_path.suffix == ".zip"
    assert tmp_path in bundle_path.parents


def test_reset_releases_state_but_preserves_downloadable_log(
    pipeline: ExperimentPipeline,
) -> None:
    """Reset drops large arrays while retaining the run's sanitized audit log."""

    session_id = pipeline.create_session()
    session = pipeline.registry.get(session_id)
    session.state.images = np.ones((1, 2, 2, 3), dtype=np.uint8)
    log_path = pipeline.log_path(session_id)

    pipeline.reset_session(session_id)

    state = pipeline.snapshot(session_id)
    assert state.images is None
    assert state.stage is ExperimentStage.EMPTY
    assert pipeline.log_path(session_id) == log_path
    assert log_path.is_file()


def test_reset_during_training_prevents_stale_worker_state_writes(
    pipeline: ExperimentPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled worker cannot repopulate the fresh state installed by reset."""

    session_id = pipeline.create_session()
    session = pipeline.registry.get(session_id)
    model = SimpleNamespace(d=3)
    with session.locked_state() as state:
        state.preprocessed_activations = np.ones((8, 3), dtype=np.float32)
        state.model = model
        state.model_config = ModelConfig(n_groups=2, group_size=1, l0=1)
        state.stage = ExperimentStage.MODEL_READY

    started = threading.Event()
    continue_worker = threading.Event()

    def slow_train(returned_model, _matrix, _config, *, progress_callback, should_stop):
        started.set()
        assert continue_worker.wait(timeout=2)
        assert should_stop()
        progress_callback(TrainingEvent(1, 1, cancelled=True, message="stopped"))
        return returned_model

    monkeypatch.setattr("bsf_experiments.pipeline.train_model", slow_train)
    worker = threading.Thread(
        target=pipeline.train,
        args=(
            session_id,
            replace(TrainingConfig(), epochs=1, batch_size=4, device="cpu"),
        ),
    )
    worker.start()
    assert started.wait(timeout=2)
    pipeline.reset_session(session_id)
    continue_worker.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    state = pipeline.snapshot(session_id)
    assert state.stage is ExperimentStage.EMPTY
    assert state.model is None
    assert state.metrics == {}


def test_reset_during_extraction_discards_the_stale_activation_result(
    pipeline: ExperimentPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DINO result computed for the prior state generation is never committed."""

    session_id = pipeline.create_session()
    session = pipeline.registry.get(session_id)
    with session.locked_state() as state:
        state.dataset_config = DatasetConfig(DatasetKind.BUNDLED_RABBITS)
        state.images = np.ones((1, 4, 4, 3), dtype=np.uint8)
        state.stage = ExperimentStage.DATA_LOADED

    started = threading.Event()
    finish = threading.Event()
    errors: list[BaseException] = []

    def blocking_extraction(*_args, **_kwargs):
        started.set()
        assert finish.wait(timeout=2)
        return np.ones((1, 4, 3), dtype=np.float32)

    def run_extraction() -> None:
        try:
            pipeline.extract_activations(session_id)
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    monkeypatch.setattr(
        "bsf_experiments.pipeline.extract_dino_activations", blocking_extraction
    )
    worker = threading.Thread(target=run_extraction)
    worker.start()
    assert started.wait(timeout=2)
    pipeline.reset_session(session_id)
    finish.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert "result was discarded" in str(errors[0])
    state = pipeline.snapshot(session_id)
    assert state.stage is ExperimentStage.EMPTY
    assert state.activations is None
    assert state.last_error is None


def test_reset_during_analysis_discards_stale_codes_and_atoms(
    pipeline: ExperimentPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reset wins deterministically when feature encoding finishes afterwards."""

    session_id = pipeline.create_session()
    session = pipeline.registry.get(session_id)
    with session.locked_state() as state:
        state.preprocessed_activations = np.ones((8, 3), dtype=np.float32)
        state.model = SimpleNamespace(d=3, n_groups=2, group_size=1)
        state.model_config = ModelConfig(n_groups=2, group_size=1, l0=1)
        state.stage = ExperimentStage.TRAINED

    started = threading.Event()
    finish = threading.Event()
    errors: list[BaseException] = []

    def blocking_encode(*_args, **_kwargs):
        started.set()
        assert finish.wait(timeout=2)
        return np.ones((8, 2, 1), dtype=np.float32)

    def run_encoding() -> None:
        try:
            pipeline.encode(session_id, device="cpu")
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    monkeypatch.setattr("bsf_experiments.pipeline.encode_features", blocking_encode)
    monkeypatch.setattr(
        "bsf_experiments.pipeline.model_atoms",
        lambda *_args, **_kwargs: np.ones((2, 1, 3), dtype=np.float32),
    )
    worker = threading.Thread(target=run_encoding)
    worker.start()
    assert started.wait(timeout=2)
    pipeline.reset_session(session_id)
    finish.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert "result was discarded" in str(errors[0])
    state = pipeline.snapshot(session_id)
    assert state.stage is ExperimentStage.EMPTY
    assert state.codes is None
    assert state.atoms is None
    assert state.last_error is None


def test_remove_session_defers_model_release_while_training(
    pipeline: ExperimentPipeline,
) -> None:
    """Browser unload requests cancellation but leaves an active model to its worker."""

    session_id = pipeline.create_session()
    session = pipeline.registry.get(session_id)
    moves: list[str] = []
    model = SimpleNamespace(d=3, to=moves.append)
    with session.locked_state() as state:
        state.model = model
        state.stage = ExperimentStage.TRAINING
    token = session.cancellation_token()

    pipeline.remove_session(session_id)

    assert token.is_set()
    assert pipeline.registry.get(session_id, touch=False) is session
    assert moves == []

    with session.locked_state() as state:
        state.stage = ExperimentStage.TRAINED
    pipeline.remove_session(session_id)
    with pytest.raises(KeyError, match="Unknown or expired"):
        pipeline.registry.get(session_id, touch=False)
    assert moves == ["cpu"]


def test_stored_error_is_redacted(
    pipeline: ExperimentPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Domain error state never retains a credential even when a dependency echoes it."""

    secret = "unit-test-secret-value"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setattr(
        "bsf_experiments.pipeline.load_dataset_images",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError(f"token={secret}")),
    )
    session_id = pipeline.create_session()

    with pytest.raises(ValueError):
        pipeline.load_dataset(session_id, DatasetConfig(DatasetKind.BUNDLED_RABBITS))

    stored_error = pipeline.snapshot(session_id).last_error
    assert stored_error is not None
    assert secret not in stored_error
    assert "[REDACTED]" in stored_error
