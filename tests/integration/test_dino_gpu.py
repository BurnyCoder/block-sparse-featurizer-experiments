"""Local-only integration coverage for the fixed gated DINOv3 backbone.

The ``gpu`` marker follows pytest's registered-marker workflow and is excluded
from ordinary runs: https://docs.pytest.org/en/stable/example/markers.html.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from bsf_experiments.backbone_identity import DINO_MODEL_ID, DINO_REVISION
from bsf_experiments.config import load_app_config
from bsf_experiments.data_phase import (
    extract_dino_activations,
    load_bundled_images,
    preprocess_dino_activations,
)


RABBIT_SUBSET_SIZE = 2
EXPECTED_PATCHES = 196
EXPECTED_FEATURES = 768


def _require_gated_gpu_environment() -> None:
    """Skip with remediation when CUDA, a token, or gated access is unavailable."""

    if not torch.cuda.is_available():
        pytest.skip("GPU integration requires a CUDA-capable PyTorch environment")

    config = load_app_config()
    token = os.getenv("HF_TOKEN", "").strip()
    if not config.hf_token_available or not token:
        pytest.skip(
            "GPU integration requires HF_TOKEN in .env after accepting the DINOv3 terms"
        )

    # Hugging Face documents this as a metadata-only access probe, avoiding a
    # large download when the gated grant is absent:
    # https://huggingface.co/docs/huggingface_hub/package_reference/file_download#get-hf-file-metadata
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    try:
        get_hf_file_metadata(
            hf_hub_url(DINO_MODEL_ID, "config.json", revision=DINO_REVISION),
            token=token,
        )
    # The actionable skip deliberately omits credential-bearing exception details.
    except Exception:
        pytest.skip(
            "HF_TOKEN could not access the gated DINOv3 model; verify its terms and grant"
        )


@pytest.mark.gpu
def test_fixed_dino_extraction_and_preprocessing_on_rabbits() -> None:
    """Extract two rabbits and verify the exact BSF activation convention."""

    _require_gated_gpu_environment()
    images = load_bundled_images()[:RABBIT_SUBSET_SIZE]

    activations = extract_dino_activations(
        images,
        device="cuda",
        batch_size=RABBIT_SUBSET_SIZE,
    )

    assert activations.shape == (
        RABBIT_SUBSET_SIZE,
        EXPECTED_PATCHES,
        EXPECTED_FEATURES,
    )
    assert activations.dtype == np.float32
    assert np.isfinite(activations).all()

    matrix, grid = preprocess_dino_activations(activations)

    assert grid == 14
    assert grid * grid == EXPECTED_PATCHES
    assert matrix.shape == (
        RABBIT_SUBSET_SIZE * EXPECTED_PATCHES,
        EXPECTED_FEATURES,
    )
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    mean_squared_token_norm = float(
        np.square(matrix.astype(np.float64)).sum(axis=1).mean()
    )
    assert mean_squared_token_norm == pytest.approx(
        EXPECTED_FEATURES,
        rel=5e-4,
        abs=0.1,
    )
