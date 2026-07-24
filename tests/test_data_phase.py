"""Tests for image ingestion and DINO activation preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bsf_experiments.data_phase import (
    center_and_scale_activations,
    extract_dino_activations,
    load_image_files,
    load_npz_images,
    patch_grid,
    validate_rgb_images,
)


def test_load_npz_images_accepts_only_nonempty_uint8_rgb_arr_0(tmp_path: Path) -> None:
    """A valid archive round-trips without permitting pickled objects."""

    expected = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
    path = tmp_path / "images.npz"
    np.savez(path, expected)

    actual = load_npz_images(path)

    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous


@pytest.mark.parametrize(
    "array",
    (
        np.empty((0, 4, 4, 3), dtype=np.uint8),
        np.zeros((2, 4, 4), dtype=np.uint8),
        np.zeros((2, 4, 4, 4), dtype=np.uint8),
        np.zeros((2, 4, 4, 3), dtype=np.float32),
    ),
)
def test_validate_rgb_images_rejects_invalid_arrays(array: np.ndarray) -> None:
    """Malformed, empty, non-RGB, and non-byte image batches fail clearly."""

    with pytest.raises(ValueError):
        validate_rgb_images(array)


def test_load_npz_images_rejects_missing_arr_0(tmp_path: Path) -> None:
    """Named arrays cannot silently substitute for the documented archive key."""

    path = tmp_path / "images.npz"
    np.savez(path, images=np.zeros((1, 4, 4, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="arr_0"):
        load_npz_images(path)


def test_load_npz_rejects_compressed_decoded_amplification_before_array_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny compressed upload cannot expand beyond the decoded-byte budget."""

    path = tmp_path / "compressed-amplification.npz"
    # Keep the member below the bounded header-overhead threshold so this case
    # specifically exercises shape/dtype header inspection rather than only
    # the ZIP central-directory size fast path.
    decoded = np.zeros((1, 32, 64, 3), dtype=np.uint8)
    np.savez_compressed(path, decoded)
    byte_budget = 2_048
    assert path.stat().st_size < byte_budget < decoded.nbytes

    def fail_if_materialized(*_args: object, **_kwargs: object) -> object:
        """Prove ZIP/header inspection rejects the array before ``np.load``."""

        raise AssertionError("np.load must not run for an over-budget decoded array")

    monkeypatch.setattr(np, "load", fail_if_materialized)

    with pytest.raises(ValueError, match="decoded.*configured upload limit"):
        load_npz_images(path, max_total_bytes=byte_budget)


def test_load_image_files_converts_supported_formats_to_rgb_uint8(
    tmp_path: Path,
) -> None:
    """Palette and grayscale inputs become a uniform RGB batch."""

    png_path = tmp_path / "one.png"
    webp_path = tmp_path / "two.webp"
    Image.new("L", (5, 4), color=60).save(png_path)
    # Lossless WebP keeps the assertion focused on mode conversion, not codec loss.
    Image.new("RGBA", (5, 4), color=(10, 20, 30, 128)).save(webp_path, lossless=True)

    images = load_image_files((png_path, webp_path))

    assert images.shape == (2, 4, 5, 3)
    assert images.dtype == np.uint8
    np.testing.assert_array_equal(images[0, 0, 0], (60, 60, 60))
    np.testing.assert_array_equal(images[1, 0, 0], (10, 20, 30))


def test_load_image_files_rejects_decoded_amplification_before_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compressed loose images are budgeted by RGB pixels before decoding them."""

    path = tmp_path / "compressed-amplification.png"
    Image.new("RGB", (256, 256), color=0).save(path, optimize=True)
    byte_budget = 2_048
    assert path.stat().st_size < byte_budget < 256 * 256 * 3

    def fail_if_converted(*_args: object, **_kwargs: object) -> object:
        """Prove the dimension preflight runs before Pillow allocates RGB pixels."""

        raise AssertionError("convert must not run for an over-budget decoded image")

    monkeypatch.setattr(Image.Image, "convert", fail_if_converted)

    with pytest.raises(ValueError, match="decoded.*configured upload limit"):
        load_image_files((path,), max_total_bytes=byte_budget)


def test_load_image_files_rejects_mismatched_sizes_and_extensions(
    tmp_path: Path,
) -> None:
    """Stacking requires equal dimensions and an explicit supported file type."""

    first = tmp_path / "one.png"
    second = tmp_path / "two.jpg"
    Image.new("RGB", (4, 4)).save(first)
    Image.new("RGB", (5, 4)).save(second)

    with pytest.raises(ValueError, match="same dimensions"):
        load_image_files((first, second))

    unsupported = tmp_path / "image.bmp"
    Image.new("RGB", (4, 4)).save(unsupported)
    with pytest.raises(ValueError, match="PNG, JPEG, or WebP"):
        load_image_files((unsupported,))


def test_center_and_scale_activations_matches_upstream_convention() -> None:
    """Preprocessing subtracts per-position means and normalizes token energy."""

    activations = np.array(
        [[[2.0, 4.0], [6.0, 8.0]], [[4.0, 8.0], [10.0, 14.0]]],
        dtype=np.float32,
    )
    positional_mean = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    flattened = center_and_scale_activations(
        activations, positional_mean=positional_mean
    )

    expected_centered = (activations - positional_mean).reshape(-1, 2)
    expected = expected_centered / np.sqrt(np.square(expected_centered).sum(1).mean())
    expected *= np.sqrt(2)
    np.testing.assert_allclose(flattened, expected, rtol=1e-6)
    assert np.square(flattened).sum(1).mean() == pytest.approx(2.0)


def test_extraction_and_preprocessing_use_upstream_fixed_dino_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime extraction pins DINO while preprocessing keeps upstream semantics."""

    from bsf import data as upstream_data

    images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    positional_mean = np.ones((4, 3), dtype=np.float32)
    raw = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) + 2
    calls: dict[str, object] = {}

    def fake_dino(
        received: np.ndarray,
        *,
        device: str | None,
        batch_size: int,
        model_id: str,
        revision: str,
    ) -> np.ndarray:
        calls.update(
            images=received,
            device=device,
            batch_size=batch_size,
            model_id=model_id,
            revision=revision,
        )
        return raw

    monkeypatch.setattr(upstream_data, "POS_MEAN", positional_mean)
    monkeypatch.setattr(upstream_data, "dino_activations", fake_dino)

    extracted = extract_dino_activations(images, device="auto", batch_size=2)
    processed = center_and_scale_activations(extracted)

    np.testing.assert_array_equal(extracted, raw)
    assert calls["images"] is images
    assert calls["device"] is None
    assert calls["batch_size"] == 2
    assert calls["model_id"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert calls["revision"] == "5931719e67bbdb9737e363e781fb0c67687896bc"
    expected_centered = (raw - positional_mean).reshape(-1, 3)
    expected = expected_centered / np.sqrt(np.square(expected_centered).sum(1).mean())
    expected *= np.sqrt(3)
    np.testing.assert_allclose(processed, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "device",
    (
        "CUDA",
        "cuda:",
        "cuda:-1",
        "cuda:+1",
        "cuda:١",
        "cuda:1:2",
        "cuda0",
        " cuda",
    ),
)
def test_extract_dino_rejects_malformed_device_before_upstream_work(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """Only the documented exact device grammar may reach DINO extraction."""

    from bsf import data as upstream_data

    def unexpected_extraction(*_args: object, **_kwargs: object) -> np.ndarray:
        """Fail if malformed input reaches the heavyweight upstream extractor."""

        pytest.fail("DINO extraction must not start for a malformed device")

    monkeypatch.setattr(upstream_data, "dino_activations", unexpected_extraction)

    with pytest.raises(ValueError, match="auto, cpu, cuda, or cuda:<index>"):
        extract_dino_activations(
            np.zeros((1, 8, 8, 3), dtype=np.uint8),
            device=device,
            batch_size=1,
        )


def test_extract_dino_rejects_out_of_range_cuda_before_upstream_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA ordinal must name one of the devices PyTorch reports as visible."""

    import torch
    from bsf import data as upstream_data

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        upstream_data,
        "dino_activations",
        lambda *_args, **_kwargs: pytest.fail(
            "DINO extraction must not start for an out-of-range CUDA ordinal"
        ),
    )

    with pytest.raises(ValueError, match="CUDA device index 1 is out of range"):
        extract_dino_activations(
            np.zeros((1, 8, 8, 3), dtype=np.uint8),
            device="cuda:1",
            batch_size=1,
        )


def test_center_and_scale_activations_rejects_wrong_mean_and_zero_energy() -> None:
    """Incompatible positional artifacts and degenerate data fail before training."""

    activations = np.zeros((2, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        center_and_scale_activations(
            activations, positional_mean=np.zeros((3, 3), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="zero energy"):
        center_and_scale_activations(
            activations, positional_mean=np.zeros((4, 3), dtype=np.float32)
        )


def test_patch_grid_requires_a_positive_square_patch_count() -> None:
    """Only square token layouts can be rendered as image overlays."""

    assert patch_grid(196) == 14
    with pytest.raises(ValueError):
        patch_grid(195)
    with pytest.raises(ValueError):
        patch_grid(0)
