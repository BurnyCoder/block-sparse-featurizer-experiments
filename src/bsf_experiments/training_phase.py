"""Validated, cancellable access to the shared upstream BSF training loop.

The research implementation remains the source of truth for optimization. This
adapter only validates UI input, seeds reproducibly, resolves the device, and
translates the upstream progress dictionaries into application contracts.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import random
from typing import Any

import numpy as np
import torch

from .analysis_phase import resolve_device
from .types import TrainingConfig, TrainingEvent


ProgressCallback = Callable[[TrainingEvent], None]
StopPredicate = Callable[[], bool]


def validate_training_config(config: TrainingConfig, *, token_count: int) -> None:
    """Reject settings that the upstream full-batch iterator cannot execute."""

    if token_count <= 0:
        raise ValueError("Training requires at least one activation token")
    for value, name in (
        (config.epochs, "epochs"),
        (config.batch_size, "batch_size"),
        (config.log_every, "log_every"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.batch_size > token_count:
        raise ValueError(
            f"batch_size ({config.batch_size}) cannot exceed token count ({token_count})"
        )
    if not math.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("lr must be a finite positive number")
    if not math.isfinite(config.snr) or config.snr < 0:
        raise ValueError("snr must be a finite nonnegative number")
    if (
        isinstance(config.seed, bool)
        or not isinstance(config.seed, int)
        or config.seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer")


def _training_event(payload: dict[str, Any]) -> TrainingEvent:
    """Convert the documented upstream callback payload to a stable dataclass."""

    return TrainingEvent(
        epoch=int(payload.get("epoch", 0)),
        total_epochs=int(payload.get("total_epochs", 0)),
        loss=float(payload["loss"]) if payload.get("loss") is not None else None,
        r2=float(payload["r2"]) if payload.get("r2") is not None else None,
        mean_l0=(
            float(payload["mean_l0"]) if payload.get("mean_l0") is not None else None
        ),
        dead_groups=(
            int(payload["dead_groups"])
            if payload.get("dead_groups") is not None
            else None
        ),
        cancelled=bool(payload.get("cancelled", False)),
        message=str(payload.get("message", "")),
    )


def train_model(
    model: Any,
    activations: np.ndarray,
    config: TrainingConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    should_stop: StopPredicate | None = None,
) -> Any:
    """Train through ``bsf.train`` and return the same partially/fully trained model.

    PyTorch documents manual seeding as the basis for reproducible random number
    streams: https://docs.pytorch.org/docs/stable/notes/randomness.html
    """

    matrix = np.asarray(activations, dtype=np.float32)
    if matrix.ndim != 2 or not all(size > 0 for size in matrix.shape):
        raise ValueError("Activations must have nonempty shape (tokens, features)")
    if not np.isfinite(matrix).all():
        raise ValueError("Activations contain non-finite values")
    if matrix.shape[1] != getattr(model, "d", None):
        raise ValueError("Activation width does not match the model dimension")
    validate_training_config(config, token_count=len(matrix))

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = resolve_device(config.device)

    def forward_progress(payload: dict[str, Any]) -> None:
        """Keep the upstream module independent of outer application types."""

        if progress_callback is not None:
            progress_callback(_training_event(payload))

    # The optional hooks are added to the pinned fork by this project's upstream
    # PR; ordinary notebook callers omit them and retain the original behavior.
    import bsf

    return bsf.train(
        model,
        matrix,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        snr=config.snr,
        device=device,
        log_every=config.log_every,
        progress_callback=forward_progress if progress_callback is not None else None,
        should_stop=should_stop,
    )


__all__ = [
    "ProgressCallback",
    "StopPredicate",
    "train_model",
    "validate_training_config",
]
