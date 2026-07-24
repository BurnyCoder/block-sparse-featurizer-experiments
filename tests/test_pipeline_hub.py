"""Tests for atomic pretrained-checkpoint integration in the pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from bsf_experiments.config import AppConfig
from bsf_experiments.hub_phase import HubCheckpointSpec, HubDownloadMetadata
from bsf_experiments.pipeline import ExperimentPipeline
from bsf_experiments.sessions import SessionRegistry
from bsf_experiments.types import (
    DatasetConfig,
    ExperimentStage,
    FeaturizerKind,
    ModelConfig,
    ModelSource,
    PretrainedRecipe,
    TrainingConfig,
)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    """Create isolated local application settings."""

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
def expected_config() -> ModelConfig:
    """Use a small model shape suitable for orchestration tests."""

    return ModelConfig(
        kind=FeaturizerKind.GRASSMANNIAN,
        n_groups=2,
        group_size=1,
        l0=1,
    )


def _spec(config: ModelConfig) -> HubCheckpointSpec:
    """Build valid immutable Hub metadata for an injected test catalog."""

    return HubCheckpointSpec(
        repo_id="BurnyCoder/test-bsf",
        revision="1" * 40,
        filename="checkpoint.pt",
        sha256=hashlib.sha256(b"fixture").hexdigest(),
        size_bytes=len(b"fixture"),
        max_bytes=1024,
        input_dim=3,
        model_config=config,
    )


def _pipeline(
    app_config: AppConfig,
    spec: HubCheckpointSpec,
    downloader,
) -> ExperimentPipeline:
    """Create a pipeline with no network dependency."""

    return ExperimentPipeline(
        app_config,
        registry=SessionRegistry(app_config.session_ttl_seconds),
        hub_catalog={PretrainedRecipe.README_QUICKSTART: spec},
        hub_downloader=downloader,
    )


def _download_client(
    spec: HubCheckpointSpec,
    checkpoint: Path,
    *,
    calls: list[dict[str, object]] | None = None,
    before_download=None,
):
    """Emulate the two-call ``hf_hub_download`` contract without network access."""

    def client(**kwargs):
        if calls is not None:
            calls.append(dict(kwargs))
        if kwargs["dry_run"]:
            return SimpleNamespace(
                commit_hash=spec.revision,
                file_size=checkpoint.stat().st_size,
                filename=spec.filename,
                is_cached=True,
            )
        if before_download is not None:
            before_download()
        return str(checkpoint)

    return client


def test_load_hub_checkpoint_atomically_replaces_model_and_clears_analysis(
    app_config: AppConfig,
    expected_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only a fully downloaded, schema-checked matching model reaches session state."""

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    new_model = SimpleNamespace(d=3)
    released: list[str] = []
    old_model = SimpleNamespace(d=3, to=released.append)
    observed_budget: list[int] = []

    def fake_restore(path: Path, **kwargs):
        assert path == checkpoint
        observed_budget.append(kwargs["max_uncompressed_bytes"])
        return new_model, expected_config

    monkeypatch.setattr("bsf_experiments.pipeline.restore_checkpoint", fake_restore)
    spec = _spec(expected_config)
    download_calls: list[dict[str, object]] = []
    pipeline = _pipeline(
        app_config,
        spec,
        _download_client(spec, checkpoint, calls=download_calls),
    )
    try:
        session_id = pipeline.create_session()
        session = pipeline.registry.get(session_id)
        with session.locked_state() as state:
            state.preprocessed_activations = np.ones((4, 3), dtype=np.float32)
            state.model = old_model
            state.model_config = expected_config
            state.codes = np.ones((4, 2, 1), dtype=np.float32)
            state.metrics["r2"] = 0.9
            state.stage = ExperimentStage.ANALYZED

        loaded = pipeline.load_hub_checkpoint(
            session_id,
            PretrainedRecipe.README_QUICKSTART,
        )

        state = pipeline.snapshot(session_id)
        assert loaded == expected_config
        assert state.model is new_model
        assert state.model_config == expected_config
        assert state.stage is ExperimentStage.MODEL_READY
        assert state.codes is None
        assert state.metrics == {}
        assert observed_budget == [1024]
        assert released == ["cpu"]
        assert [call["dry_run"] for call in download_calls] == [True, False]
        assert all(call["revision"] == spec.revision for call in download_calls)
        assert "checkpoint.hugging_face.preflight" in pipeline.read_log(session_id)
    finally:
        pipeline.close()


def test_injected_hub_client_cannot_bypass_checkpoint_integrity(
    app_config: AppConfig,
    expected_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Even an injected low-level client must pass the shared SHA-256 check."""

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"tamper!")
    spec = _spec(expected_config)
    pipeline = _pipeline(
        app_config,
        spec,
        _download_client(spec, checkpoint),
    )
    monkeypatch.setattr(
        "bsf_experiments.pipeline.restore_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "restore must not run after an integrity failure"
        ),
    )
    try:
        session_id = pipeline.create_session()

        with pytest.raises(ValueError, match="SHA-256"):
            pipeline.load_hub_checkpoint(
                session_id,
                PretrainedRecipe.README_QUICKSTART,
            )

        assert pipeline.snapshot(session_id).model is None
    finally:
        pipeline.close()


def test_load_hub_checkpoint_rejects_config_and_width_without_state_mutation(
    app_config: AppConfig,
    expected_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Catalog identity and activation width are checked before atomic replacement."""

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    old_model = SimpleNamespace(d=3)
    wrong_model = SimpleNamespace(d=4, to=lambda _device: None)
    wrong_config = ModelConfig(
        kind=FeaturizerKind.VANILLA,
        n_groups=2,
        group_size=1,
        l0=1,
    )
    spec = _spec(expected_config)
    pipeline = _pipeline(
        app_config,
        spec,
        _download_client(spec, checkpoint),
    )
    session_id = pipeline.create_session()
    session = pipeline.registry.get(session_id)
    with session.locked_state() as state:
        state.preprocessed_activations = np.ones((4, 3), dtype=np.float32)
        state.model = old_model
        state.model_config = expected_config
        state.stage = ExperimentStage.MODEL_READY

    monkeypatch.setattr(
        "bsf_experiments.pipeline.restore_checkpoint",
        lambda *_args, **_kwargs: (wrong_model, wrong_config),
    )
    try:
        with pytest.raises(ValueError, match="catalog model configuration"):
            pipeline.load_hub_checkpoint(
                session_id,
                PretrainedRecipe.README_QUICKSTART,
            )
        state = pipeline.snapshot(session_id)
        assert state.model is old_model
        assert state.model_config == expected_config

        monkeypatch.setattr(
            "bsf_experiments.pipeline.restore_checkpoint",
            lambda *_args, **_kwargs: (wrong_model, expected_config),
        )
        with pytest.raises(ValueError, match="input dimension"):
            pipeline.load_hub_checkpoint(
                session_id,
                PretrainedRecipe.README_QUICKSTART,
            )
        assert pipeline.snapshot(session_id).model is old_model
    finally:
        pipeline.close()


def test_load_hub_checkpoint_checks_catalog_width_without_activations_and_logs_preflight(
    app_config: AppConfig,
    expected_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Catalog width and non-secret Hub metadata apply before any dataset is loaded."""

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    metadata = HubDownloadMetadata(
        repo_id="BurnyCoder/test-bsf",
        filename="checkpoint.pt",
        resolved_commit="1" * 40,
        remote_size=7,
        is_cached=True,
    )

    def fake_download(_spec, *, downloader, metadata_callback):
        assert downloader is None
        metadata_callback(metadata)
        return checkpoint

    monkeypatch.setattr(
        "bsf_experiments.pipeline.download_hub_checkpoint",
        fake_download,
    )
    monkeypatch.setattr(
        "bsf_experiments.pipeline.restore_checkpoint",
        lambda *_args, **_kwargs: (
            SimpleNamespace(d=4, to=lambda _device: None),
            expected_config,
        ),
    )
    pipeline = ExperimentPipeline(
        app_config,
        registry=SessionRegistry(app_config.session_ttl_seconds),
        hub_catalog={PretrainedRecipe.README_QUICKSTART: _spec(expected_config)},
    )
    try:
        session_id = pipeline.create_session()
        with pytest.raises(ValueError, match="trusted catalog entry"):
            pipeline.load_hub_checkpoint(
                session_id,
                PretrainedRecipe.README_QUICKSTART,
            )
        assert pipeline.snapshot(session_id).model is None

        monkeypatch.setattr(
            "bsf_experiments.pipeline.restore_checkpoint",
            lambda *_args, **_kwargs: (SimpleNamespace(d=3), expected_config),
        )
        loaded = pipeline.load_hub_checkpoint(
            session_id,
            PretrainedRecipe.README_QUICKSTART,
        )

        assert loaded == expected_config
        assert pipeline.snapshot(session_id).stage is ExperimentStage.MODEL_READY
        log = pipeline.read_log(session_id)
        assert metadata.repo_id in log
        assert metadata.filename in log
        assert metadata.resolved_commit in log
        assert '"remote_size":7' in log
        assert '"is_cached":true' in log
    finally:
        pipeline.close()


def test_reset_during_hub_download_discards_stale_result(
    app_config: AppConfig,
    expected_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A completed network request cannot repopulate state replaced by reset."""

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    started = threading.Event()
    finish = threading.Event()
    errors: list[BaseException] = []
    downloaded_model = SimpleNamespace(d=3, to=lambda _device: None)

    def before_download() -> None:
        started.set()
        assert finish.wait(timeout=2)

    monkeypatch.setattr(
        "bsf_experiments.pipeline.restore_checkpoint",
        lambda *_args, **_kwargs: (downloaded_model, expected_config),
    )
    spec = _spec(expected_config)
    pipeline = _pipeline(
        app_config,
        spec,
        _download_client(spec, checkpoint, before_download=before_download),
    )
    session_id = pipeline.create_session()
    session = pipeline.registry.get(session_id)
    with session.locked_state() as state:
        state.preprocessed_activations = np.ones((4, 3), dtype=np.float32)
        state.stage = ExperimentStage.PREPROCESSED

    def worker() -> None:
        try:
            pipeline.load_hub_checkpoint(
                session_id,
                PretrainedRecipe.README_QUICKSTART,
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(timeout=2)
    pipeline.reset_session(session_id)
    finish.set()
    thread.join(timeout=2)

    try:
        assert not thread.is_alive()
        assert len(errors) == 1
        assert "result was discarded" in str(errors[0])
        state = pipeline.snapshot(session_id)
        assert state.stage is ExperimentStage.EMPTY
        assert state.model is None
    finally:
        pipeline.close()


def test_hub_pipeline_branch_skips_initialization_and_training(
    app_config: AppConfig,
    expected_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Hub failures propagate and the pretrained branch never silently retrains."""

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    spec = _spec(expected_config)
    pipeline = _pipeline(
        app_config,
        spec,
        _download_client(spec, checkpoint),
    )
    session_id = pipeline.create_session()
    calls: list[str] = []

    for phase, result in (
        ("load_dataset", None),
        ("extract_activations", None),
        ("center_and_scale", None),
        ("load_hub_checkpoint", expected_config),
        ("encode", None),
        ("evaluate", None),
        ("rank", []),
    ):
        monkeypatch.setattr(
            pipeline,
            phase,
            lambda *_args, _phase=phase, _result=result, **_kwargs: (
                calls.append(_phase),
                _result,
            )[1],
        )
    monkeypatch.setattr(
        pipeline,
        "initialize_model",
        lambda *_args, **_kwargs: pytest.fail("must not initialize in Hub mode"),
    )
    monkeypatch.setattr(
        pipeline,
        "train",
        lambda *_args, **_kwargs: pytest.fail("must not train in Hub mode"),
    )
    try:
        result = pipeline.run_current_pipeline(
            session_id,
            DatasetConfig(),
            expected_config,
            TrainingConfig(device="cpu"),
            model_source=ModelSource.HUGGING_FACE,
            pretrained_recipe=PretrainedRecipe.README_QUICKSTART,
        )
        assert result == []
        assert calls == [
            "load_dataset",
            "extract_activations",
            "center_and_scale",
            "load_hub_checkpoint",
            "encode",
            "evaluate",
            "rank",
        ]
    finally:
        pipeline.close()


def test_hub_pipeline_requires_recipe_and_propagates_download_error(
    app_config: AppConfig,
    expected_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid Hub selection is explicit and no training fallback is attempted."""

    pipeline = _pipeline(
        app_config,
        _spec(expected_config),
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("Hub unavailable")),
    )
    monkeypatch.setattr(pipeline, "load_dataset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "extract_activations", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "center_and_scale", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "train",
        lambda *_args, **_kwargs: pytest.fail("must not silently retrain"),
    )
    session_id = pipeline.create_session()
    try:
        with pytest.raises(ValueError, match="pretrained recipe"):
            pipeline.run_current_pipeline(
                session_id,
                DatasetConfig(),
                expected_config,
                TrainingConfig(device="cpu"),
                model_source=ModelSource.HUGGING_FACE,
            )

        with pytest.raises(RuntimeError, match="Hub unavailable"):
            pipeline.run_current_pipeline(
                session_id,
                DatasetConfig(),
                expected_config,
                TrainingConfig(device="cpu"),
                model_source=ModelSource.HUGGING_FACE,
                pretrained_recipe=PretrainedRecipe.README_QUICKSTART,
            )
    finally:
        pipeline.close()
