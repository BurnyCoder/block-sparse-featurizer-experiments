"""Named configurations matching the upstream README and starter notebooks."""

from __future__ import annotations

from dataclasses import dataclass

from .types import FeaturizerKind, ModelConfig, TrainingConfig


@dataclass(frozen=True, slots=True)
class ExperimentPreset:
    """Pair the model and trainer settings used by one upstream example."""

    label: str
    model: ModelConfig
    training: TrainingConfig


PRESETS: dict[str, ExperimentPreset] = {
    "readme": ExperimentPreset(
        label="README Quickstart",
        model=ModelConfig(
            kind=FeaturizerKind.GRASSMANNIAN,
            n_groups=256,
            group_size=3,
            l0=16,
        ),
        training=TrainingConfig(epochs=60, lr=4e-4, batch_size=2048, snr=0.1),
    ),
    "grassmannian_notebook": ExperimentPreset(
        label="Grassmannian Notebook",
        model=ModelConfig(
            kind=FeaturizerKind.GRASSMANNIAN,
            n_groups=256,
            group_size=3,
            l0=8,
        ),
        training=TrainingConfig(epochs=300, lr=3e-3, batch_size=2048, snr=0.1),
    ),
    "group_lasso_notebook": ExperimentPreset(
        label="Group Lasso Notebook",
        model=ModelConfig(
            kind=FeaturizerKind.GROUP_LASSO,
            n_groups=256,
            group_size=3,
            target_l0=8,
        ),
        training=TrainingConfig(epochs=300, lr=4e-4, batch_size=2048, snr=0.1),
    ),
    "vanilla_notebook": ExperimentPreset(
        label="Vanilla Notebook",
        model=ModelConfig(
            kind=FeaturizerKind.VANILLA,
            n_groups=256,
            group_size=3,
            l0=8,
        ),
        training=TrainingConfig(epochs=300, lr=3e-3, batch_size=2048, snr=0.1),
    ),
}


def get_preset(name: str) -> ExperimentPreset:
    """Return a named immutable preset or an actionable error for UI callers."""

    try:
        return PRESETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown preset: {name}") from error


__all__ = ["ExperimentPreset", "PRESETS", "get_preset"]
