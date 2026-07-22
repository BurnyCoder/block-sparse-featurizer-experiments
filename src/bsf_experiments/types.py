"""Shared typed contracts for the experiment pipeline and its presentation layers.

The configuration objects are deliberately independent of Gradio so notebooks,
the CLI, tests, and the UI all drive the same application behavior. Python's
standard-library dataclasses are documented at
https://docs.python.org/3/library/dataclasses.html.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class DatasetKind(StrEnum):
    """Image sources supported by the upstream DINO activation workflow."""

    BUNDLED_RABBITS = "bundled_rabbits"
    NPZ = "npz"
    UPLOADED_IMAGES = "uploaded_images"


class FeaturizerKind(StrEnum):
    """The three concrete BSF implementations exported by the pinned library."""

    GRASSMANNIAN = "grassmannian"
    GROUP_LASSO = "group_lasso"
    VANILLA = "vanilla"


class ExperimentStage(StrEnum):
    """Coarse states used to guard pipeline actions and explain UI readiness."""

    EMPTY = "empty"
    DATA_LOADED = "data_loaded"
    ACTIVATIONS_READY = "activations_ready"
    PREPROCESSED = "preprocessed"
    MODEL_READY = "model_ready"
    TRAINING = "training"
    TRAINED = "trained"
    ANALYZED = "analyzed"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Select an image source and the fixed DINOv3 extraction behavior."""

    kind: DatasetKind = DatasetKind.BUNDLED_RABBITS
    paths: tuple[Path, ...] = ()
    extraction_batch_size: int = 64
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Describe any of the three upstream block-sparse featurizers."""

    kind: FeaturizerKind = FeaturizerKind.GRASSMANNIAN
    n_groups: int = 256
    group_size: int = 3
    l0: int = 16
    coef: float = 1e-2
    target_l0: int = 16
    gain: float = 10.0
    paper_version: bool = False


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Parameters accepted by the shared upstream trainer."""

    epochs: int = 60
    lr: float = 4e-4
    batch_size: int = 2048
    snr: float = 0.1
    device: str = "auto"
    log_every: int = 5
    seed: int = 0


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Effective ``bsf.viz.plot_concepts`` controls (excluding its no-op flag)."""

    n_img: int = 10
    ncol_img: int = 5
    clip: float = 98.0
    saturation: float = 1.0
    drop_low_norm: float = 0.0
    max_points: int = 5_000
    point_size: float = 4.0
    concept_gap: float = 0.6


@dataclass(slots=True)
class ExperimentState:
    """Server-side state for one session; large objects never enter browser state."""

    session_id: str
    stage: ExperimentStage = ExperimentStage.EMPTY
    dataset_config: DatasetConfig | None = None
    images: Any | None = None
    activations: Any | None = None
    preprocessed_activations: Any | None = None
    grid: int | None = None
    model_config: ModelConfig | None = None
    model: Any | None = None
    codes: Any | None = None
    atoms: Any | None = None
    metrics: dict[str, float | int] = field(default_factory=dict)
    concepts: list[ConceptRecord] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    last_error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def touch(self) -> None:
        """Record session activity so the registry can expire stale GPU state."""

        self.updated_at = datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    """One structured progress report emitted by the cancellable trainer."""

    epoch: int
    total_epochs: int
    loss: float | None = None
    r2: float | None = None
    mean_l0: float | None = None
    dead_groups: int | None = None
    cancelled: bool = False
    message: str = ""


@dataclass(frozen=True, slots=True)
class ConceptRecord:
    """Serializable concept activity statistics shown in ranking tables."""

    rank: int
    group_id: int
    firing_count: int
    firing_rate: float
    energy: float

    def as_dict(self) -> dict[str, int | float]:
        """Return a table/export-friendly representation without framework types."""

        return {
            "rank": self.rank,
            "group_id": self.group_id,
            "firing_count": self.firing_count,
            "firing_rate": self.firing_rate,
            "energy": self.energy,
        }


__all__ = [
    "ConceptRecord",
    "DatasetConfig",
    "DatasetKind",
    "ExperimentStage",
    "ExperimentState",
    "FeaturizerKind",
    "ModelConfig",
    "PlotConfig",
    "TrainingConfig",
    "TrainingEvent",
]
