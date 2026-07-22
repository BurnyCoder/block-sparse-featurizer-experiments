"""Tests for the stable data contracts shared by the pipeline and UI."""

from __future__ import annotations

from bsf_experiments.types import (
    ConceptRecord,
    DatasetConfig,
    DatasetKind,
    ExperimentStage,
    ExperimentState,
    FeaturizerKind,
    ModelConfig,
    PlotConfig,
    TrainingConfig,
    TrainingEvent,
)


def test_configuration_defaults_describe_the_readme_workflow() -> None:
    """Defaults are useful, immutable values suitable for preset replacement."""

    dataset = DatasetConfig()
    model = ModelConfig()
    training = TrainingConfig()
    plotting = PlotConfig()

    assert dataset.kind is DatasetKind.BUNDLED_RABBITS
    assert dataset.extraction_batch_size == 64
    assert model.kind is FeaturizerKind.GRASSMANNIAN
    assert (model.n_groups, model.group_size, model.l0) == (256, 3, 16)
    assert (training.epochs, training.lr, training.batch_size) == (60, 4e-4, 2048)
    assert not hasattr(plotting, "per_axis_rgb")
    assert plotting.n_img == 10


def test_experiment_state_and_result_records_are_explicit() -> None:
    """The registry can track phase progress while analysis records stay serializable."""

    state = ExperimentState(session_id="session-1")
    event = TrainingEvent(
        epoch=1,
        total_epochs=10,
        loss=0.5,
        r2=0.25,
        mean_l0=3.0,
        dead_groups=1,
    )
    concept = ConceptRecord(
        rank=1,
        group_id=7,
        firing_count=12,
        firing_rate=0.5,
        energy=30.0,
    )

    assert state.stage is ExperimentStage.EMPTY
    state.stage = ExperimentStage.DATA_LOADED
    state.metrics["r2"] = 0.25
    assert state.stage is ExperimentStage.DATA_LOADED
    assert event.total_epochs == 10
    assert concept.group_id == 7
