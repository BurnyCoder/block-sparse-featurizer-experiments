"""Tests for BSF construction, encoding, evaluation, and concept ranking."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

import bsf
from bsf_experiments.analysis_phase import (
    encode_features,
    evaluate_reconstruction,
    model_atoms,
    rank_concepts,
    resolve_device,
    select_top_concepts,
)
from bsf_experiments.model_phase import create_model, validate_model_config
from bsf_experiments.types import FeaturizerKind, ModelConfig


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    (
        (FeaturizerKind.GRASSMANNIAN, bsf.GrassmannianBSF),
        (FeaturizerKind.GROUP_LASSO, bsf.GroupLassoBSF),
        (FeaturizerKind.VANILLA, bsf.VanillaBSF),
    ),
)
def test_create_model_builds_each_supported_featurizer(
    kind: FeaturizerKind, expected_type: type[bsf.BSF]
) -> None:
    """The factory exposes exactly the three variants supported upstream."""

    config = ModelConfig(
        kind=kind,
        n_groups=4,
        group_size=2,
        l0=2,
        target_l0=2,
        coef=0.03,
        gain=5.0,
    )

    model = create_model(config, input_dim=6)

    assert isinstance(model, expected_type)
    assert model.d == 6
    assert model.n_groups == 4
    assert model.group_size == 2
    if isinstance(model, bsf.GroupLassoBSF):
        assert model.coef == pytest.approx(0.03)
        assert model.target_l0 == 2
        assert model.gain == pytest.approx(5.0)


@pytest.mark.parametrize(
    "config",
    (
        ModelConfig(n_groups=0),
        ModelConfig(group_size=0),
        ModelConfig(n_groups=4, l0=5),
        ModelConfig(kind=FeaturizerKind.GROUP_LASSO, n_groups=4, target_l0=5),
        ModelConfig(kind=FeaturizerKind.GROUP_LASSO, coef=-0.1),
        ModelConfig(kind=FeaturizerKind.GROUP_LASSO, gain=0.0),
    ),
)
def test_validate_model_config_rejects_invalid_sparsity(config: ModelConfig) -> None:
    """Invalid dimensions and sparsity cannot reach fragile upstream math."""

    with pytest.raises(ValueError):
        validate_model_config(config, input_dim=6)


class _IdentityBlocks(torch.nn.Module):
    """Tiny deterministic BSF-compatible model for analysis tests."""

    d = 2
    n_groups = 2
    group_size = 1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Treat each input coordinate as one one-dimensional concept."""

        return x.reshape(-1, 2, 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Invert ``encode`` exactly."""

        return z.reshape(-1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an exact reconstruction and its grouped code."""

        z = self.encode(x)
        return self.decode(z), z


class _InvalidCodes(_IdentityBlocks):
    """Return caller-selected bad codes while preserving the BSF interface."""

    def __init__(self, codes: torch.Tensor) -> None:
        """Retain a fixture tensor so both analysis entry points see it."""

        super().__init__()
        self._codes = codes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat the invalid fixture for the current batch length."""

        return self._codes[: len(x)].to(device=x.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pair the invalid codes with an otherwise valid reconstruction."""

        return x.clone(), self.encode(x)


class _InvalidReconstruction(_IdentityBlocks):
    """Return an infinite reconstruction to exercise metric validation."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep code shape valid so reconstruction validation is isolated."""

        reconstruction = x.clone()
        reconstruction[0, 0] = torch.inf
        return reconstruction, self.encode(x)


class _InvalidAtoms(_IdentityBlocks):
    """Expose an atom tensor whose axes disagree with model metadata."""

    def atoms(self) -> torch.Tensor:
        """Swap group and block extents while retaining three dimensions."""

        return torch.zeros((1, 2, self.d), dtype=torch.float32)


def test_encode_and_evaluate_are_batched_and_restore_training_mode() -> None:
    """Analysis produces complete arrays and metrics without mutating model mode."""

    model = _IdentityBlocks().train()
    x = np.array([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]], dtype=np.float32)

    encoded = encode_features(model, x, device="cpu", batch_size=2)
    metrics = evaluate_reconstruction(model, x, device="cpu", batch_size=2)

    np.testing.assert_array_equal(encoded, x.reshape(3, 2, 1))
    assert metrics == {
        "r2": pytest.approx(1.0),
        "mean_l0": pytest.approx(4 / 3),
        "dead_groups": 0,
        "tokens": 3,
    }
    assert model.training is True


def test_analysis_rejects_incompatible_or_empty_activations() -> None:
    """Analysis validates rank and feature width before invoking a model."""

    model = _IdentityBlocks()
    with pytest.raises(ValueError):
        encode_features(model, np.empty((0, 2), dtype=np.float32), device="cpu")
    with pytest.raises(ValueError):
        evaluate_reconstruction(model, np.ones((3, 4), dtype=np.float32), device="cpu")


@pytest.mark.parametrize(
    "codes",
    (
        torch.zeros((3, 1, 2), dtype=torch.float32),
        torch.full((3, 2, 1), float("nan"), dtype=torch.float32),
    ),
)
@pytest.mark.parametrize("operation", (encode_features, evaluate_reconstruction))
def test_analysis_rejects_wrong_shape_or_nonfinite_codes(
    codes: torch.Tensor,
    operation: Callable[..., object],
) -> None:
    """Every code batch must exactly match model metadata and contain finite values."""

    model = _InvalidCodes(codes).train()
    activations = np.ones((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"Model (encode|code) output"):
        operation(model, activations, device="cpu", batch_size=3)

    assert model.training is True


def test_evaluation_rejects_nonfinite_reconstruction_before_metrics() -> None:
    """NaN or infinity cannot silently become an invalid reconstruction metric."""

    with pytest.raises(ValueError, match="reconstruction.*non-finite"):
        evaluate_reconstruction(
            _InvalidReconstruction(),
            np.ones((3, 2), dtype=np.float32),
            device="cpu",
            batch_size=3,
        )


def test_model_atoms_must_match_model_dimensions_exactly() -> None:
    """Visualization artifacts use declared group, block, and feature dimensions."""

    with pytest.raises(ValueError, match="expected shape"):
        model_atoms(_InvalidAtoms())


@pytest.mark.parametrize(
    "device",
    (
        "CUDA",
        "cuda:",
        "cuda:-1",
        "cuda:+1",
        "cuda:١",
        "cuda:1:2",
        "cuda0",
        " cuda",
    ),
)
def test_analysis_rejects_malformed_device_names(device: str) -> None:
    """Analysis accepts only the exact configured CPU/CUDA device grammar."""

    with pytest.raises(ValueError, match="auto, cpu, cuda, or cuda:<index>"):
        resolve_device(device)


def test_analysis_rejects_out_of_range_cuda_before_moving_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinal validation happens before model transfer or batched inference."""

    class TrackingIdentity(_IdentityBlocks):
        """Record any attempted device transfer so the ordering is observable."""

        moved = False

        def to(self, *_args: object, **_kwargs: object) -> TrackingIdentity:
            """Mark an invalid transfer instead of asking a CPU test host for CUDA."""

            self.moved = True
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    model = TrackingIdentity()

    with pytest.raises(ValueError, match="CUDA device index 1 is out of range"):
        encode_features(
            model,
            np.ones((2, 2), dtype=np.float32),
            device="cuda:1",
            batch_size=1,
        )

    assert model.moved is False


def test_rank_and_select_concepts_use_energy_with_minimum_firing_filter() -> None:
    """Ranking mirrors the README: descending energy after a firing threshold."""

    z = np.array(
        [
            [[3.0], [1.0], [0.0]],
            [[0.0], [2.0], [5.0]],
            [[4.0], [0.0], [0.0]],
            [[0.0], [3.0], [0.0]],
        ],
        dtype=np.float32,
    )

    records = rank_concepts(z, min_firings=2)

    assert [record.group_id for record in records] == [0, 1]
    assert [record.rank for record in records] == [1, 2]
    assert records[0].energy == pytest.approx(25.0)
    assert records[0].firing_count == 2
    assert records[0].firing_rate == pytest.approx(0.5)
    assert select_top_concepts(records, 1) == [0]
    with pytest.raises(ValueError):
        select_top_concepts(records, 0)
