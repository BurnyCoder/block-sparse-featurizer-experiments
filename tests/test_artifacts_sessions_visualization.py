"""Focused tests for run artifacts, server sessions, and concept rendering."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json
from pathlib import Path
import zipfile

from matplotlib.figure import Figure
import numpy as np
import pytest
import torch

from bsf_experiments.artifacts import ArtifactStore, load_checkpoint
from bsf_experiments.model_phase import create_model
from bsf_experiments.sessions import SessionRegistry
from bsf_experiments.types import (
    ExperimentStage,
    FeaturizerKind,
    ModelConfig,
    PlotConfig,
)
from bsf_experiments.visualization_phase import render_concepts


class _Clock:
    """Controllable monotonic clock used to test expiry without sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current synthetic monotonic time."""

        return self.now


def test_artifact_store_exports_safe_checkpoint_arrays_figures_and_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One timestamped run contains portable exports and a restricted checkpoint."""

    fixed_time = datetime(2026, 7, 22, 12, 34, 56, 123456, tzinfo=UTC)
    store = ArtifactStore.create(tmp_path, "Rabbit run", now=fixed_time)
    config = ModelConfig(
        kind=FeaturizerKind.VANILLA,
        n_groups=2,
        group_size=1,
        l0=1,
    )
    model = create_model(config, input_dim=2)

    checkpoint_path = store.save_checkpoint(model, config)
    arrays_path = store.save_arrays(
        {"codes": np.arange(8, dtype=np.float32).reshape(4, 2)}
    )
    figure = Figure(figsize=(2, 2))
    figure.subplots().plot([0, 1], [0, 1])
    figure_paths = store.save_figure(figure, stem="concepts", dpi=60)
    bundle_path = store.save_result_bundle({"status": "passed", "r2": 0.75})

    assert store.run_dir.name == "20260722T123456.123456Z-rabbit-run"
    assert checkpoint_path.is_file()
    with np.load(arrays_path, allow_pickle=False) as saved:
        assert saved.files == ["codes"]
        np.testing.assert_array_equal(saved["codes"], np.arange(8).reshape(4, 2))
    assert set(figure_paths) == {"png", "pdf"}
    assert all(path.stat().st_size > 0 for path in figure_paths.values())

    observed: dict[str, object] = {}
    real_torch_load = torch.load

    def recording_load(*args: object, **kwargs: object) -> object:
        observed.update(kwargs)
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    loaded = load_checkpoint(checkpoint_path)

    assert observed["weights_only"] is True
    assert observed["map_location"] == "cpu"
    assert loaded.model_config == config
    assert loaded.input_dim == 2
    assert set(loaded.state_dict) == set(model.state_dict())
    assert all(tensor.device.type == "cpu" for tensor in loaded.state_dict.values())
    restored = loaded.build_model()
    assert not restored.training
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)

    with zipfile.ZipFile(bundle_path) as bundle:
        names = set(bundle.namelist())
        assert "result.json" in names
        assert "arrays.npz" in names
        assert "concepts.png" in names
        assert "concepts.pdf" in names
        assert bundle_path.name not in names
        result = json.loads(bundle.read("result.json"))
    assert result == {"r2": 0.75, "status": "passed"}


def test_checkpoint_loader_rejects_non_allowlisted_config_fields(
    tmp_path: Path,
) -> None:
    """Restricted unpickling is followed by an exact application-level schema check."""

    path = tmp_path / "untrusted.pt"
    torch.save(
        {
            "format_version": 1,
            "input_dim": 2,
            "model_config": {
                "kind": "vanilla",
                "n_groups": 2,
                "group_size": 1,
                "l0": 1,
                "coef": 0.01,
                "target_l0": 1,
                "gain": 10.0,
                "paper_version": False,
                "unexpected": "not allowed",
            },
            "state_dict": {"weight": torch.ones(1)},
        },
        path,
    )

    with pytest.raises(ValueError, match="model_config fields"):
        load_checkpoint(path)


def test_checkpoint_loader_rejects_claimed_dimensions_before_model_allocation(
    tmp_path: Path,
) -> None:
    """Tiny tensors cannot pair with huge metadata and induce a model allocation."""

    path = tmp_path / "dimension-bomb.pt"
    torch.save(
        {
            "format_version": 1,
            "input_dim": 1_000_000_000,
            "model_config": {
                "kind": "grassmannian",
                "n_groups": 2,
                "group_size": 1,
                "l0": 1,
                "coef": 0.01,
                "target_l0": 1,
                "gain": 10.0,
                "paper_version": False,
            },
            "state_dict": {
                "B_raw": torch.zeros((2, 2, 1)),
                "gamma": torch.ones(2),
            },
        },
        path,
    )

    with pytest.raises(ValueError, match="B_raw has shape"):
        load_checkpoint(path)


def test_checkpoint_loader_rejects_sparse_tensor_layout(tmp_path: Path) -> None:
    """Sparse metadata cannot conceal an enormous logical decoder tensor."""

    path = tmp_path / "sparse.pt"
    config = {
        "kind": "vanilla",
        "n_groups": 2,
        "group_size": 1,
        "l0": 1,
        "coef": 0.01,
        "target_l0": 1,
        "gain": 10.0,
        "paper_version": False,
    }
    sparse_decoder = torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.int64),
        torch.empty(0),
        size=(2, 2),
        check_invariants=True,
    )
    torch.save(
        {
            "format_version": 1,
            "input_dim": 2,
            "model_config": config,
            "state_dict": {
                "W_dec": sparse_decoder,
                "W_enc": torch.zeros((2, 2)),
                "b_enc": torch.zeros(2),
            },
        },
        path,
    )

    with pytest.raises(ValueError, match="dense strided layout"):
        load_checkpoint(path)


def test_checkpoint_writer_rejects_nonfinite_irrelevant_config_values(
    tmp_path: Path,
) -> None:
    """Every serialized field is validated even when its model variant ignores it."""

    valid_config = ModelConfig(
        kind=FeaturizerKind.VANILLA,
        n_groups=2,
        group_size=1,
        l0=1,
    )
    invalid_config = ModelConfig(
        kind=FeaturizerKind.VANILLA,
        n_groups=2,
        group_size=1,
        l0=1,
        coef=float("nan"),
    )
    model = create_model(valid_config, input_dim=2)

    with pytest.raises(ValueError, match="coef must be a finite nonnegative number"):
        ArtifactStore.create(tmp_path).save_checkpoint(model, invalid_config)


@pytest.mark.parametrize(
    ("model_kind", "checkpoint_kind"),
    (
        (FeaturizerKind.GRASSMANNIAN, FeaturizerKind.VANILLA),
        (FeaturizerKind.GROUP_LASSO, FeaturizerKind.GRASSMANNIAN),
        (FeaturizerKind.VANILLA, FeaturizerKind.GROUP_LASSO),
    ),
)
def test_checkpoint_writer_rejects_model_class_kind_mismatch(
    tmp_path: Path,
    model_kind: FeaturizerKind,
    checkpoint_kind: FeaturizerKind,
) -> None:
    """Checkpoint metadata cannot claim a different concrete BSF implementation."""

    model_config = ModelConfig(
        kind=model_kind,
        n_groups=3,
        group_size=1,
        l0=1,
        target_l0=1,
    )
    checkpoint_config = ModelConfig(
        kind=checkpoint_kind,
        n_groups=3,
        group_size=1,
        l0=1,
        target_l0=1,
    )
    model = create_model(model_config, input_dim=4)

    with pytest.raises(ValueError, match="kind does not match"):
        ArtifactStore.create(tmp_path).save_checkpoint(model, checkpoint_config)


@pytest.mark.parametrize(
    ("kind", "attribute", "mismatched_value"),
    (
        (FeaturizerKind.GRASSMANNIAN, "d", 5),
        (FeaturizerKind.GRASSMANNIAN, "n_groups", 4),
        (FeaturizerKind.GRASSMANNIAN, "n_groups", 3.0),
        (FeaturizerKind.GRASSMANNIAN, "group_size", 2),
        (FeaturizerKind.GRASSMANNIAN, "l0", 2),
        (FeaturizerKind.VANILLA, "d", 5),
        (FeaturizerKind.VANILLA, "n_groups", 4),
        (FeaturizerKind.VANILLA, "group_size", 2),
        (FeaturizerKind.VANILLA, "l0", 2),
        (FeaturizerKind.GROUP_LASSO, "d", 5),
        (FeaturizerKind.GROUP_LASSO, "n_groups", 4),
        (FeaturizerKind.GROUP_LASSO, "group_size", 2),
        (FeaturizerKind.GROUP_LASSO, "coef", 0.25),
        (FeaturizerKind.GROUP_LASSO, "target_l0", 2),
        (FeaturizerKind.GROUP_LASSO, "gain", 5.0),
        (FeaturizerKind.GROUP_LASSO, "paper_version", True),
    ),
)
def test_checkpoint_writer_rejects_every_runtime_behavior_mismatch(
    tmp_path: Path,
    kind: FeaturizerKind,
    attribute: str,
    mismatched_value: object,
) -> None:
    """All constructor attributes that affect the restored variant must agree."""

    config = ModelConfig(
        kind=kind,
        n_groups=3,
        group_size=1,
        l0=1,
        coef=0.01,
        target_l0=1,
        gain=10.0,
        paper_version=False,
    )
    model = create_model(config, input_dim=4)
    setattr(model, attribute, mismatched_value)

    with pytest.raises(ValueError, match=rf"\b{attribute}\b.*does not match"):
        ArtifactStore.create(tmp_path).save_checkpoint(
            model,
            config,
            input_dim=4,
        )


def test_result_bundle_recursively_sanitizes_sensitive_metadata(
    tmp_path: Path,
) -> None:
    """Neither loose nor zipped result metadata may retain credential values."""

    leaked_values = (
        "bundle-database-password-fabricated-leak",
        "bundle-access-token-fabricated-leak",
        "bundle-client-secret-fabricated-leak",
        "bundle-authorization-header-fabricated-leak",
    )
    store = ArtifactStore.create(tmp_path)
    bundle_path = store.save_result_bundle(
        {
            "metadata": {
                "databasePassword": leaked_values[0],
                "deeper": [
                    {
                        "accessToken": leaked_values[1],
                        "clientSecret": leaked_values[2],
                    }
                ],
                "diagnostic": (f"authorizationHeader={leaked_values[3]}"),
            },
            "status": "passed",
        }
    )

    loose_json = (store.run_dir / "result.json").read_text(encoding="utf-8")
    with zipfile.ZipFile(bundle_path) as bundle:
        zipped_json = bundle.read("result.json").decode("utf-8")

    for serialized in (loose_json, zipped_json):
        for leaked_value in leaked_values:
            assert leaked_value not in serialized
        parsed = json.loads(serialized)
        assert parsed["metadata"]["databasePassword"] == "[REDACTED]"
        assert parsed["metadata"]["deeper"][0] == {
            "accessToken": "[REDACTED]",
            "clientSecret": "[REDACTED]",
        }
        assert parsed["metadata"]["diagnostic"] == "[REDACTED]"


def test_timestamp_collision_gets_a_distinct_run_directory(tmp_path: Path) -> None:
    """Atomic directory creation keeps simultaneous runs from sharing artifacts."""

    fixed_time = datetime(2026, 7, 22, tzinfo=UTC)

    first = ArtifactStore.create(tmp_path, "same", now=fixed_time)
    second = ArtifactStore.create(tmp_path, "same", now=fixed_time)

    assert first.run_dir != second.run_dir
    assert second.run_dir.name.endswith("-01")


def test_session_registry_is_locked_cancellable_resettable_and_ttl_bounded() -> None:
    """Concurrent callers share one session while reset and expiry signal old work."""

    clock = _Clock()
    registry = SessionRegistry(ttl_seconds=10, clock=clock)

    with ThreadPoolExecutor(max_workers=4) as pool:
        sessions = list(
            pool.map(lambda _: registry.get_or_create("browser-1"), range(12))
        )
    session = sessions[0]
    assert all(candidate is session for candidate in sessions)

    old_token = session.cancellation_token()
    with ThreadPoolExecutor(max_workers=2) as pool:
        with session.locked_state() as state:
            state.stage = ExperimentStage.TRAINING
            get_future = pool.submit(registry.get, "browser-1")
            cancel_future = pool.submit(registry.cancel, "browser-1")
            assert get_future.result(timeout=1) is session
            assert cancel_future.result(timeout=1) is True
            assert old_token.is_set()

    reset_session = registry.reset("browser-1")
    assert reset_session is session
    assert reset_session.state.stage is ExperimentStage.EMPTY
    assert reset_session.cancellation_token() is not old_token
    assert not reset_session.cancellation_token().is_set()

    clock.now = 10.0
    assert registry.cleanup_expired() == ("browser-1",)
    assert reset_session.cancellation_token().is_set()
    with pytest.raises(KeyError):
        registry.get("browser-1")


def test_render_concepts_validates_and_delegates_every_effective_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid selection reaches upstream with the exact effective plot settings."""

    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    codes = np.ones((8, 2, 1), dtype=np.float32)
    atoms = np.ones((2, 1, 3), dtype=np.float32)
    config = PlotConfig(
        n_img=2,
        ncol_img=2,
        clip=95.0,
        saturation=1.2,
        drop_low_norm=0.0,
        max_points=8,
        point_size=3.0,
        concept_gap=0.4,
    )
    sentinel = object()
    observed: dict[str, object] = {}

    def fake_plot(*args: object, **kwargs: object) -> object:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return sentinel

    import bsf

    monkeypatch.setattr(bsf.viz, "plot_concepts", fake_plot)

    result = render_concepts(codes, atoms, images, [1], grid=2, config=config)

    assert result is sentinel
    assert observed["args"][3] == [1]  # type: ignore[index]
    assert observed["args"][4] == 2  # type: ignore[index]
    assert observed["kwargs"] == {
        "n_img": 2,
        "ncol_img": 2,
        "clip": 95.0,
        "saturation": 1.2,
        "drop_low_norm": 0.0,
        "max_points": 8,
        "point_size": 3.0,
        "concept_gap": 0.4,
    }


@pytest.mark.parametrize("selected", ([], [2], [0, 0]))
def test_render_concepts_rejects_invalid_selections(selected: list[int]) -> None:
    """Empty, out-of-range, and duplicate group selections fail before plotting."""

    with pytest.raises(ValueError):
        render_concepts(
            np.ones((8, 2, 1), dtype=np.float32),
            np.ones((2, 1, 3), dtype=np.float32),
            np.zeros((2, 4, 4, 3), dtype=np.uint8),
            selected,
            grid=2,
            config=PlotConfig(n_img=2, max_points=8),
        )


def test_render_concepts_rejects_groups_with_fewer_than_eight_visible_firings() -> None:
    """The wrapper raises an actionable error instead of upstream's blank band."""

    codes = np.ones((8, 1, 1), dtype=np.float32)
    codes[0, 0, 0] = 0.0

    with pytest.raises(ValueError, match=r"group 0 has 7"):
        render_concepts(
            codes,
            np.ones((1, 1, 3), dtype=np.float32),
            np.zeros((2, 4, 4, 3), dtype=np.uint8),
            [0],
            grid=2,
            config=PlotConfig(n_img=2, max_points=8),
        )


def test_render_concepts_requires_three_atom_feature_dimensions() -> None:
    """The upstream 3D projection cannot index atoms with fewer coordinates."""

    with pytest.raises(ValueError, match="three feature dimensions"):
        render_concepts(
            np.ones((8, 1, 1), dtype=np.float32),
            np.ones((1, 1, 2), dtype=np.float32),
            np.zeros((2, 4, 4, 3), dtype=np.uint8),
            [0],
            grid=2,
            config=PlotConfig(n_img=2, max_points=8),
        )
