"""Tests for validated delegation to the upstream cancellable trainer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bsf_experiments.training_phase import train_model, validate_training_config
from bsf_experiments.model_phase import create_model
from bsf_experiments.types import FeaturizerKind, ModelConfig, TrainingConfig


@pytest.mark.parametrize("kind", tuple(FeaturizerKind))
def test_each_real_featurizer_completes_tiny_cpu_training(
    kind: FeaturizerKind,
) -> None:
    """Exercise the shared upstream trainer against every public model variant."""

    model = create_model(
        ModelConfig(
            kind=kind,
            n_groups=3,
            group_size=2,
            l0=1,
            target_l0=1,
            coef=0.01,
            gain=2.0,
        ),
        input_dim=4,
    )
    rng = np.random.default_rng(7)
    activations = rng.normal(size=(8, 4)).astype(np.float32)
    events = []

    trained = train_model(
        model,
        activations,
        TrainingConfig(
            epochs=1,
            lr=1e-3,
            batch_size=4,
            snr=0.0,
            log_every=1,
            seed=7,
            device="cpu",
        ),
        progress_callback=events.append,
    )

    assert trained is model
    assert len(events) == 1
    assert events[0].epoch == 1
    assert events[0].loss is not None and np.isfinite(events[0].loss)
    assert events[0].r2 is not None and np.isfinite(events[0].r2)


def test_training_config_rejects_batch_larger_than_tokens() -> None:
    """Prevent the upstream iterator's empty-batch division failure."""

    with pytest.raises(ValueError, match="cannot exceed token count"):
        validate_training_config(TrainingConfig(batch_size=5), token_count=4)


@pytest.mark.parametrize(
    ("field", "value"),
    [("epochs", 0), ("batch_size", 0), ("log_every", 0), ("lr", 0.0), ("snr", -1.0)],
)
def test_training_config_rejects_invalid_numbers(
    field: str, value: int | float
) -> None:
    """Every numeric UI field fails early with its own actionable name."""

    values = {field: value}
    with pytest.raises(ValueError, match=field):
        validate_training_config(TrainingConfig(**values), token_count=2048)


def test_train_model_delegates_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate upstream dictionaries without duplicating the optimizer loop."""

    model = SimpleNamespace(d=2)
    calls: dict[str, object] = {}

    def fake_train(
        passed_model: object, values: np.ndarray, **kwargs: object
    ) -> object:
        calls.update(kwargs)
        callback = kwargs["progress_callback"]
        assert callable(callback)
        callback(
            {
                "epoch": 1,
                "total_epochs": 1,
                "loss": 0.5,
                "r2": 0.7,
                "mean_l0": 2.0,
                "dead_groups": 1,
                "cancelled": False,
                "message": "done",
            }
        )
        return passed_model

    import bsf

    monkeypatch.setattr(bsf, "train", fake_train)
    events = []
    result = train_model(
        model,
        np.ones((2, 2), dtype=np.float32),
        TrainingConfig(epochs=1, batch_size=2, log_every=1, device="cpu"),
        progress_callback=events.append,
        should_stop=lambda: False,
    )

    assert result is model
    assert events[0].message == "done"
    assert calls["device"] == "cpu"
