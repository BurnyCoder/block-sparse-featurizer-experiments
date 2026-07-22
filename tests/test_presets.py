"""Preserve exact upstream example parameters as a regression contract."""

from bsf_experiments.presets import PRESETS, get_preset
from bsf_experiments.types import FeaturizerKind


def test_all_upstream_workflows_have_presets() -> None:
    """The UI must expose one button for each documented end-to-end example."""

    assert set(PRESETS) == {
        "readme",
        "grassmannian_notebook",
        "group_lasso_notebook",
        "vanilla_notebook",
    }


def test_notebook_presets_match_upstream_parameters() -> None:
    """Keep the three 300-epoch reference settings distinct and exact."""

    grassmannian = get_preset("grassmannian_notebook")
    group_lasso = get_preset("group_lasso_notebook")
    vanilla = get_preset("vanilla_notebook")

    assert grassmannian.model.kind is FeaturizerKind.GRASSMANNIAN
    assert grassmannian.model.l0 == 8
    assert grassmannian.training.lr == 3e-3
    assert group_lasso.model.kind is FeaturizerKind.GROUP_LASSO
    assert group_lasso.model.target_l0 == 8
    assert group_lasso.training.lr == 4e-4
    assert vanilla.model.kind is FeaturizerKind.VANILLA
    assert vanilla.model.l0 == 8
    assert vanilla.training.lr == 3e-3
    assert {
        preset.training.epochs for preset in (grassmannian, group_lasso, vanilla)
    } == {300}
