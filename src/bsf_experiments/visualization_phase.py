"""Validated, thread-safe adapter around the upstream concept visualization.

Global context
--------------
The pinned library's ``plot_concepts`` function is the single supported visual:
https://github.com/BurnyCoder/block-sparse-featurizer/blob/main/bsf/viz.py#L112-L177.
It silently renders a blank band for concepts with fewer than eight firings and
assumes mutually compatible array shapes. This boundary turns those assumptions
into actionable errors before allocating a potentially large figure.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import threading
from typing import Any

import numpy as np

from .types import PlotConfig


MIN_MANIFOLD_FIRINGS = 8
FIRING_THRESHOLD = 1e-6

# Matplotlib documents artist race conditions and requires callers to serialize
# threaded access: https://matplotlib.org/stable/users/faq.html#work-with-threads.
_MATPLOTLIB_LOCK = threading.RLock()


def _positive_integer(value: object, name: str) -> int:
    """Return a positive plain integer, rejecting booleans and floats."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_plot_config(config: PlotConfig, *, image_count: int) -> None:
    """Reject plot settings that fail or create misleading empty image slots."""

    _positive_integer(config.n_img, "n_img")
    _positive_integer(config.ncol_img, "ncol_img")
    _positive_integer(config.max_points, "max_points")
    if config.n_img > image_count:
        raise ValueError("n_img cannot exceed the number of source images")
    if config.max_points < MIN_MANIFOLD_FIRINGS:
        raise ValueError(f"max_points must be at least {MIN_MANIFOLD_FIRINGS}")
    if not math.isfinite(config.clip) or not 0 < config.clip <= 100:
        raise ValueError("clip must be a finite percentile in (0, 100]")
    if not math.isfinite(config.saturation) or config.saturation < 0:
        raise ValueError("saturation must be a finite nonnegative number")
    if not math.isfinite(config.drop_low_norm) or not 0 <= config.drop_low_norm < 1:
        raise ValueError("drop_low_norm must be a finite fraction in [0, 1)")
    if not math.isfinite(config.point_size) or config.point_size <= 0:
        raise ValueError("point_size must be a finite positive number")
    if not math.isfinite(config.concept_gap) or config.concept_gap < 0:
        raise ValueError("concept_gap must be a finite nonnegative number")


def _selected_groups(selected: Sequence[int], *, n_groups: int) -> list[int]:
    """Return unique, in-range plain group IDs in the user's chosen order."""

    if isinstance(selected, (str, bytes)):
        raise ValueError("Select concepts as a sequence of integer group IDs")
    groups = list(selected)
    if not groups:
        raise ValueError("Select at least one concept to visualize")
    for group_id in groups:
        if isinstance(group_id, bool) or not isinstance(group_id, int):
            raise ValueError("Selected concept IDs must be integers")
        if not 0 <= group_id < n_groups:
            raise ValueError(f"Selected concept ID is out of range: {group_id}")
    if len(groups) != len(set(groups)):
        raise ValueError("Selected concept IDs must be unique")
    return groups


def _finite_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    """Convert an array-like value and enforce finite, nonempty dimensionality."""

    array = np.asarray(value)
    if array.ndim != ndim or not all(size > 0 for size in array.shape):
        raise ValueError(f"{name} must have a nonempty {ndim}-dimensional shape")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    return array


def concept_firing_counts(
    codes: np.ndarray,
    *,
    threshold: float = FIRING_THRESHOLD,
) -> np.ndarray:
    """Return each group's firing count using upstream's block-norm threshold."""

    array = _finite_array(codes, name="codes", ndim=3)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative number")
    heat = np.linalg.norm(array.astype(np.float64), axis=-1)
    return np.count_nonzero(heat > threshold, axis=0)


def _validate_retained_firings(
    heat: np.ndarray,
    groups: Sequence[int],
    *,
    drop_low_norm: float,
) -> None:
    """Ensure upstream's optional quantile drop still leaves a 3D plottable cloud."""

    problems: list[str] = []
    for group_id in groups:
        firing_norms = heat[heat[:, group_id] > FIRING_THRESHOLD, group_id]
        retained = len(firing_norms)
        # Mirror the upstream condition exactly before checking the retained size.
        if drop_low_norm > 0 and retained > MIN_MANIFOLD_FIRINGS:
            cutoff = np.quantile(firing_norms, drop_low_norm)
            retained = int(np.count_nonzero(firing_norms >= cutoff))
        if retained < MIN_MANIFOLD_FIRINGS:
            problems.append(f"group {group_id} has {retained}")
    if problems:
        raise ValueError(
            "Every selected concept must retain at least "
            f"{MIN_MANIFOLD_FIRINGS} firing patches; " + ", ".join(problems)
        )


def validate_visualization_inputs(
    codes: Any,
    atoms: Any,
    images: Any,
    selected: Sequence[int],
    *,
    grid: int,
    config: PlotConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Validate and return contiguous arrays ready for upstream plotting."""

    code_array = _finite_array(codes, name="codes", ndim=3)
    atom_array = _finite_array(atoms, name="atoms", ndim=3)
    image_array = np.asarray(images)
    if image_array.ndim != 4 or not all(size > 0 for size in image_array.shape):
        raise ValueError(
            "images must have nonempty shape (images, height, width, channels)"
        )
    if image_array.shape[-1] not in (3, 4):
        raise ValueError("images must have RGB or RGBA channels")
    if (
        not np.issubdtype(image_array.dtype, np.number)
        or not np.isfinite(image_array).all()
    ):
        raise ValueError("images must contain only finite numeric values")

    side = _positive_integer(grid, "grid")
    n_tokens, n_groups, group_size = code_array.shape
    if atom_array.shape[:2] != (n_groups, group_size):
        raise ValueError(
            "atoms must match codes in group and block dimensions: "
            f"received codes {code_array.shape}, atoms {atom_array.shape}"
        )
    if atom_array.shape[2] < 3:
        raise ValueError(
            "atoms must have at least three feature dimensions for the 3D manifold"
        )
    expected_tokens = len(image_array) * side * side
    if n_tokens != expected_tokens:
        raise ValueError(
            "codes token count must equal images * grid²: "
            f"expected {expected_tokens}, received {n_tokens}"
        )
    groups = _selected_groups(selected, n_groups=n_groups)
    validate_plot_config(config, image_count=len(image_array))
    heat = np.linalg.norm(code_array.astype(np.float64), axis=-1)
    _validate_retained_firings(heat, groups, drop_low_norm=config.drop_low_norm)

    return (
        np.ascontiguousarray(code_array),
        np.ascontiguousarray(atom_array),
        np.ascontiguousarray(image_array),
        groups,
    )


def render_concepts(
    codes: Any,
    atoms: Any,
    images: Any,
    selected: Sequence[int],
    *,
    grid: int,
    config: PlotConfig | None = None,
) -> Any:
    """Validate inputs and render the upstream manifold-plus-overlay figure."""

    settings = config or PlotConfig()
    code_array, atom_array, image_array, groups = validate_visualization_inputs(
        codes,
        atoms,
        images,
        selected,
        grid=grid,
        config=settings,
    )
    from bsf import viz

    with _MATPLOTLIB_LOCK:
        return viz.plot_concepts(
            code_array,
            atom_array,
            image_array,
            groups,
            grid,
            n_img=settings.n_img,
            ncol_img=settings.ncol_img,
            clip=settings.clip,
            saturation=settings.saturation,
            drop_low_norm=settings.drop_low_norm,
            max_points=settings.max_points,
            point_size=settings.point_size,
            concept_gap=settings.concept_gap,
        )


__all__ = [
    "FIRING_THRESHOLD",
    "MIN_MANIFOLD_FIRINGS",
    "concept_firing_counts",
    "render_concepts",
    "validate_plot_config",
    "validate_visualization_inputs",
]
