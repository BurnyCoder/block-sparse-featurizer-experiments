"""Validated construction of the three featurizers in the pinned submodule.

The constructor mapping follows the upstream implementation and README at
https://github.com/BurnyCoder/block-sparse-featurizer#three-featurizer-variants.
Application code calls this factory instead of branching on model classes.
"""

from __future__ import annotations

import math
from typing import Any

from .types import FeaturizerKind, ModelConfig


def _positive_integer(value: int, name: str) -> None:
    """Reject booleans and nonpositive values used as tensor dimensions."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_model_config(config: ModelConfig, *, input_dim: int) -> FeaturizerKind:
    """Validate dimensions and variant-specific sparsity before allocation."""

    try:
        kind = FeaturizerKind(config.kind)
    except ValueError as error:
        raise ValueError(f"Unsupported featurizer kind: {config.kind}") from error
    _positive_integer(input_dim, "input_dim")
    _positive_integer(config.n_groups, "n_groups")
    _positive_integer(config.group_size, "group_size")
    if config.group_size > input_dim:
        raise ValueError("group_size cannot exceed the activation dimension")

    if kind in (FeaturizerKind.GRASSMANNIAN, FeaturizerKind.VANILLA):
        _positive_integer(config.l0, "l0")
        if config.l0 > config.n_groups:
            raise ValueError("l0 cannot exceed n_groups")
    if kind is FeaturizerKind.GROUP_LASSO:
        if not math.isfinite(config.coef) or config.coef < 0:
            raise ValueError("coef must be a finite nonnegative number")
        if not math.isfinite(config.gain) or config.gain <= 0:
            raise ValueError("gain must be a finite positive number")
        if not config.paper_version:
            _positive_integer(config.target_l0, "target_l0")
            if config.target_l0 > config.n_groups:
                raise ValueError("target_l0 cannot exceed n_groups")
    return kind


def create_model(config: ModelConfig, *, input_dim: int) -> Any:
    """Instantiate one upstream BSF using only parameters meaningful to its kind."""

    kind = validate_model_config(config, input_dim=input_dim)
    # Import lazily so configuration validation and docs do not require PyTorch startup.
    import bsf

    common = {
        "d": input_dim,
        "n_groups": config.n_groups,
        "group_size": config.group_size,
    }
    if kind is FeaturizerKind.GRASSMANNIAN:
        return bsf.GrassmannianBSF(**common, l0=config.l0)
    if kind is FeaturizerKind.GROUP_LASSO:
        return bsf.GroupLassoBSF(
            **common,
            coef=config.coef,
            target_l0=config.target_l0,
            gain=config.gain,
            paper_version=config.paper_version,
        )
    if kind is FeaturizerKind.VANILLA:
        return bsf.VanillaBSF(**common, l0=config.l0)
    # ``validate_model_config`` makes this unreachable while retaining type safety.
    raise ValueError(f"Unsupported featurizer kind: {kind}")


__all__ = ["create_model", "validate_model_config"]
