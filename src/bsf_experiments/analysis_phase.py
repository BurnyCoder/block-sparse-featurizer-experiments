"""Framework-neutral encoding, reconstruction metrics, and concept ranking.

Inference runs under ``torch.inference_mode`` to avoid autograd bookkeeping as
recommended by PyTorch for evaluation-only work:
https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
import torch

from .config import resolve_torch_device
from .types import ConceptRecord


def _activation_matrix(model: Any, activations: np.ndarray) -> np.ndarray:
    """Validate a finite nonempty matrix against the model's input dimension."""

    matrix = np.asarray(activations, dtype=np.float32)
    if matrix.ndim != 2 or not all(size > 0 for size in matrix.shape):
        raise ValueError("Activations must have nonempty shape (tokens, features)")
    expected_dim = getattr(model, "d", None)
    if expected_dim is None or matrix.shape[1] != expected_dim:
        raise ValueError(
            f"Activation width {matrix.shape[1]} does not match model dimension {expected_dim}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Activations contain non-finite values")
    return np.ascontiguousarray(matrix)


def _batch_size(value: int) -> int:
    """Validate the positive batch size shared by encoding and evaluation."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("batch_size must be a positive integer")
    return value


def _model_dimension(model: Any, name: str) -> int:
    """Read one positive integer dimension from the upstream model contract."""

    value = getattr(model, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Model {name} must be a positive integer; received {value!r}")
    return value


def _validate_codes(
    model: Any,
    codes: Any,
    *,
    tokens: int,
    output_name: str,
) -> torch.Tensor:
    """Require a finite tensor whose axes exactly match declared model metadata.

    ``torch.isfinite`` is the documented element-wise finite-value predicate:
    https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html.
    """

    expected = (
        tokens,
        _model_dimension(model, "n_groups"),
        _model_dimension(model, "group_size"),
    )
    if not isinstance(codes, torch.Tensor):
        raise ValueError(
            f"Model {output_name} output must be a PyTorch tensor with shape {expected}"
        )
    actual = tuple(codes.shape)
    if actual != expected:
        raise ValueError(
            f"Model {output_name} output has shape {actual}; expected {expected} "
            "from (tokens, model.n_groups, model.group_size)"
        )
    if not bool(torch.isfinite(codes).all().item()):
        raise ValueError(f"Model {output_name} output contains non-finite values")
    return codes


def _validate_reconstruction(
    reconstruction: Any, *, expected: torch.Tensor
) -> torch.Tensor:
    """Require a finite reconstruction tensor with the exact input shape."""

    expected_shape = tuple(expected.shape)
    if not isinstance(reconstruction, torch.Tensor):
        raise ValueError(
            "Model reconstruction output must be a PyTorch tensor with shape "
            f"{expected_shape}"
        )
    actual = tuple(reconstruction.shape)
    if actual != expected_shape:
        raise ValueError(
            f"Model reconstruction output has shape {actual}; expected {expected_shape}"
        )
    if not bool(torch.isfinite(reconstruction).all().item()):
        raise ValueError("Model reconstruction output contains non-finite values")
    return reconstruction


def resolve_device(device: str) -> str:
    """Resolve ``auto`` and reject an unavailable or malformed CUDA request."""

    return resolve_torch_device(device, torch_module=torch)


def encode_features(
    model: Any,
    activations: np.ndarray,
    *,
    device: str = "auto",
    batch_size: int = 20_000,
) -> np.ndarray:
    """Encode all activation tokens in bounded batches and return CPU float32 codes."""

    resolved_device = resolve_device(device)
    size = _batch_size(batch_size)
    matrix = _activation_matrix(model, activations)
    was_training = bool(model.training)
    batches: list[np.ndarray] = []
    try:
        model.to(resolved_device).eval()
        with torch.inference_mode():
            for start in range(0, len(matrix), size):
                batch = torch.as_tensor(
                    matrix[start : start + size],
                    dtype=torch.float32,
                    device=resolved_device,
                )
                codes = _validate_codes(
                    model,
                    model.encode(batch),
                    tokens=len(batch),
                    output_name="encode",
                )
                batches.append(codes.float().cpu().numpy())
    finally:
        model.train(was_training)
    return np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)


def evaluate_reconstruction(
    model: Any,
    activations: np.ndarray,
    *,
    device: str = "auto",
    batch_size: int = 20_000,
) -> dict[str, float | int]:
    """Compute global reconstruction R², mean active blocks, and dead groups."""

    resolved_device = resolve_device(device)
    size = _batch_size(batch_size)
    matrix = _activation_matrix(model, activations)
    was_training = bool(model.training)
    feature_mean = matrix.mean(axis=0, keepdims=True, dtype=np.float64)
    total_sum_squares = float(np.square(matrix.astype(np.float64) - feature_mean).sum())
    residual_sum_squares = 0.0
    active_total = 0
    alive: torch.Tensor | None = None
    try:
        model.to(resolved_device).eval()
        with torch.inference_mode():
            for start in range(0, len(matrix), size):
                batch = torch.as_tensor(
                    matrix[start : start + size],
                    dtype=torch.float32,
                    device=resolved_device,
                )
                reconstruction, codes = model(batch)
                reconstruction = _validate_reconstruction(
                    reconstruction, expected=batch
                )
                codes = _validate_codes(
                    model,
                    codes,
                    tokens=len(batch),
                    output_name="code",
                )
                residual_sum_squares += float(
                    (batch.double() - reconstruction.double()).square().sum().item()
                )
                active = codes.norm(dim=-1) > 1e-6
                active_total += int(active.sum().item())
                batch_alive = active.any(dim=0).cpu()
                alive = batch_alive if alive is None else alive | batch_alive
    finally:
        model.train(was_training)

    denominator = max(total_sum_squares, 1e-12)
    n_groups = int(getattr(model, "n_groups"))
    alive_count = int(alive.sum().item()) if alive is not None else 0
    metrics: dict[str, float | int] = {
        "r2": float(1.0 - residual_sum_squares / denominator),
        "mean_l0": float(active_total / len(matrix)),
        "dead_groups": n_groups - alive_count,
        "tokens": len(matrix),
    }
    if not math.isfinite(float(metrics["r2"])) or not math.isfinite(
        float(metrics["mean_l0"])
    ):
        raise ValueError("Reconstruction metrics contain non-finite values")
    return metrics


def model_atoms(model: Any) -> np.ndarray:
    """Copy per-concept decoder atoms to NumPy for visualization and export."""

    was_training = bool(model.training)
    try:
        model.eval()
        with torch.inference_mode():
            atom_tensor = model.atoms()
    finally:
        model.train(was_training)
    expected = (
        _model_dimension(model, "n_groups"),
        _model_dimension(model, "group_size"),
        _model_dimension(model, "d"),
    )
    if not isinstance(atom_tensor, torch.Tensor):
        raise ValueError(
            f"Model atoms must be a PyTorch tensor with expected shape {expected}"
        )
    actual = tuple(atom_tensor.shape)
    if actual != expected:
        raise ValueError(f"Model atoms have shape {actual}; expected shape {expected}")
    if not bool(torch.isfinite(atom_tensor).all().item()):
        raise ValueError("Model atoms contain non-finite values")
    atoms = atom_tensor.float().cpu().numpy()
    return np.ascontiguousarray(atoms, dtype=np.float32)


def rank_concepts(
    codes: np.ndarray,
    *,
    min_firings: int = 0,
    firing_threshold: float = 1e-6,
) -> list[ConceptRecord]:
    """Rank concepts by total squared block norm after a firing-count filter."""

    array = np.asarray(codes)
    if array.ndim != 3 or not all(size > 0 for size in array.shape):
        raise ValueError("Codes must have nonempty shape (tokens, groups, block)")
    if not np.isfinite(array).all():
        raise ValueError("Codes contain non-finite values")
    if (
        isinstance(min_firings, bool)
        or not isinstance(min_firings, int)
        or min_firings < 0
    ):
        raise ValueError("min_firings must be a nonnegative integer")
    if not math.isfinite(firing_threshold) or firing_threshold < 0:
        raise ValueError("firing_threshold must be a finite nonnegative number")

    # Float64 norms avoid energy overflow for otherwise valid float32 model output.
    heat = np.linalg.norm(array.astype(np.float64), axis=-1)
    firing_counts = (heat > firing_threshold).sum(axis=0)
    energies = np.square(heat).sum(axis=0)
    ordered_groups = np.argsort(-energies, kind="stable")
    records: list[ConceptRecord] = []
    for group_id in ordered_groups:
        count = int(firing_counts[group_id])
        if count < min_firings:
            continue
        records.append(
            ConceptRecord(
                rank=len(records) + 1,
                group_id=int(group_id),
                firing_count=count,
                firing_rate=float(count / array.shape[0]),
                energy=float(energies[group_id]),
            )
        )
    return records


def select_top_concepts(records: Sequence[ConceptRecord], count: int) -> list[int]:
    """Return group IDs for the first ``count`` ranked records."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Concept count must be a positive integer")
    return [record.group_id for record in records[:count]]


# A concise alias keeps pipeline code readable without duplicating behavior.
evaluate_model = evaluate_reconstruction


__all__ = [
    "encode_features",
    "evaluate_model",
    "evaluate_reconstruction",
    "model_atoms",
    "rank_concepts",
    "resolve_device",
    "select_top_concepts",
]
