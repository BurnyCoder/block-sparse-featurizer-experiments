"""Image ingestion, DINO extraction, and upstream-compatible preprocessing.

NPZ archives are opened with ``allow_pickle=False`` because NumPy documents
that loading object arrays can execute arbitrary code:
https://numpy.org/doc/stable/reference/generated/numpy.load.html.

The NPZ preflight uses ZIP member sizes and the public NumPy header readers so
decoded array bytes are bounded before allocation:
https://docs.python.org/3/library/zipfile.html#zipfile.ZipInfo.file_size and
https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .backbone_identity import DINO_MODEL_ID, DINO_REVISION
from .config import resolve_torch_device, validate_device_setting
from .types import DatasetConfig, DatasetKind


SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_NPY_MEMBER_NAME = "arr_0.npy"
_MAX_NPY_HEADER_BYTES = 10_000
_MAX_NPY_CONTAINER_OVERHEAD = _MAX_NPY_HEADER_BYTES + 12


def bundled_rabbit_path() -> Path:
    """Locate the pinned upstream rabbit archive from the outer checkout."""

    return (
        Path(__file__).resolve().parents[2]
        / "vendor"
        / "block-sparse-featurizer"
        / "rabbit.npz"
    )


def validate_rgb_images(images: Any) -> np.ndarray:
    """Validate and return a contiguous nonempty ``(N,H,W,3)`` uint8 batch."""

    if not isinstance(images, np.ndarray):
        raise TypeError("Images must be provided as a NumPy array")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Images must have shape (N, H, W, 3)")
    if not all(size > 0 for size in images.shape):
        raise ValueError("Images must be nonempty")
    if images.dtype != np.uint8:
        raise ValueError("Images must use uint8 RGB values")
    return np.ascontiguousarray(images)


def _validate_byte_budget(max_total_bytes: int | None) -> int | None:
    """Return a positive optional byte budget shared by encoded and decoded data."""

    if max_total_bytes is None:
        return None
    if (
        isinstance(max_total_bytes, bool)
        or not isinstance(max_total_bytes, int)
        or max_total_bytes <= 0
    ):
        raise ValueError("The upload limit must be a positive integer number of bytes")
    return max_total_bytes


def _check_file_size(path: Path, max_total_bytes: int | None) -> None:
    """Reject a file exceeding the caller's configured upload budget."""

    if max_total_bytes is not None and path.stat().st_size > max_total_bytes:
        raise ValueError(f"File exceeds the configured upload limit: {path.name}")


def _read_npy_header(member: Any) -> tuple[tuple[int, ...], np.dtype[Any], int]:
    """Read one bounded public NPY header and report its decoded-data offset."""

    try:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
                member, max_header_size=_MAX_NPY_HEADER_BYTES
            )
        elif version == (2, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
                member, max_header_size=_MAX_NPY_HEADER_BYTES
            )
        else:
            raise ValueError(
                f"NPZ arr_0 uses unsupported NPY format version {version}; "
                "save it again with NumPy as a plain uint8 RGB array"
            )
    except EOFError as error:
        raise ValueError("NPZ arr_0.npy has a truncated NumPy header") from error
    return tuple(shape), np.dtype(dtype), int(member.tell())


def _validate_rgb_metadata(shape: tuple[int, ...], dtype: np.dtype[Any]) -> int:
    """Validate RGB array metadata and calculate bytes without allocating pixels."""

    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError(f"NPZ arr_0 must have shape (N, H, W, 3); received {shape}")
    if not all(isinstance(size, int) and size > 0 for size in shape):
        raise ValueError("NPZ arr_0 dimensions must all be positive integers")
    if dtype.hasobject:
        raise ValueError("NPZ arr_0 object arrays are not allowed")
    if dtype != np.dtype(np.uint8):
        raise ValueError(f"NPZ arr_0 must use uint8 RGB values; received {dtype}")
    return math.prod(shape) * dtype.itemsize


def _preflight_npz_array(archive_path: Path, byte_budget: int | None) -> None:
    """Bound and validate the sole ``arr_0.npy`` member before NumPy loads it."""

    try:
        with zipfile.ZipFile(archive_path) as archive:
            matching = [
                info for info in archive.infolist() if info.filename == _NPY_MEMBER_NAME
            ]
            if not matching:
                raise ValueError("NPZ archive must contain an arr_0 image array")
            if len(matching) != 1:
                raise ValueError(
                    "NPZ archive must contain exactly one arr_0 image array"
                )
            member_info = matching[0]
            if (
                byte_budget is not None
                and member_info.file_size > byte_budget + _MAX_NPY_CONTAINER_OVERHEAD
            ):
                raise ValueError(
                    "NPZ arr_0 decoded data exceeds the configured upload limit "
                    "before allocation"
                )
            with archive.open(member_info, mode="r") as member:
                shape, dtype, data_offset = _read_npy_header(member)
            decoded_bytes = _validate_rgb_metadata(shape, dtype)
            if byte_budget is not None and decoded_bytes > byte_budget:
                raise ValueError(
                    f"NPZ arr_0 decoded data requires {decoded_bytes} bytes and "
                    f"exceeds the configured upload limit of {byte_budget} bytes"
                )
            if data_offset + decoded_bytes != member_info.file_size:
                raise ValueError(
                    "NPZ arr_0.npy size does not match the dimensions in its header"
                )
    except zipfile.BadZipFile as error:
        raise ValueError(f"Could not read NPZ archive: {archive_path.name}") from error


def load_npz_images(
    path: str | Path, *, max_total_bytes: int | None = None
) -> np.ndarray:
    """Load the documented non-pickled ``arr_0`` RGB image array from NPZ."""

    archive_path = Path(path)
    if archive_path.suffix.lower() != ".npz":
        raise ValueError("The uploaded dataset must be an .npz archive")
    if not archive_path.is_file():
        raise FileNotFoundError(f"NPZ archive does not exist: {archive_path}")
    byte_budget = _validate_byte_budget(max_total_bytes)
    _check_file_size(archive_path, byte_budget)
    _preflight_npz_array(archive_path, byte_budget)

    try:
        # NumPy's context manager closes the underlying ZipFile deterministically.
        with np.load(archive_path, allow_pickle=False) as archive:
            if "arr_0" not in archive.files:
                raise ValueError("NPZ archive must contain an arr_0 image array")
            # NPZ reads already produce an independent ndarray; avoiding a second
            # copy keeps peak memory close to the preflighted decoded byte count.
            images = archive["arr_0"]
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "arr_0" in str(error):
            raise
        raise ValueError(f"Could not load NPZ archive: {archive_path.name}") from error
    return validate_rgb_images(images)


def load_bundled_images(path: str | Path | None = None) -> np.ndarray:
    """Load the pinned upstream rabbit dataset through the same validation path."""

    return load_npz_images(path or bundled_rabbit_path())


def load_image_files(
    paths: Sequence[str | Path], *, max_total_bytes: int | None = None
) -> np.ndarray:
    """Load PNG/JPEG/WebP files, apply EXIF orientation, and convert to RGB uint8.

    Pillow's ``Image.convert`` is the standard mode conversion API:
    https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert.
    """

    image_paths = tuple(Path(path) for path in paths)
    if not image_paths:
        raise ValueError("Select at least one image")
    byte_budget = _validate_byte_budget(max_total_bytes)

    images: list[np.ndarray] = []
    consumed_bytes = 0
    decoded_bytes = 0
    expected_size: tuple[int, int] | None = None
    for path in image_paths:
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError("Uploaded images must be PNG, JPEG, or WebP files")
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        consumed_bytes += path.stat().st_size
        if byte_budget is not None and consumed_bytes > byte_budget:
            raise ValueError("Uploaded images exceed the configured upload limit")

        try:
            with Image.open(path) as opened:
                width, height = opened.size
                decoded_bytes += width * height * 3
                if byte_budget is not None and decoded_bytes > byte_budget:
                    raise ValueError(
                        f"Uploaded decoded RGB image data requires {decoded_bytes} bytes and "
                        f"exceeds the configured upload limit of {byte_budget} bytes"
                    )
                # EXIF transposition prevents phone images from appearing rotated.
                rgb_image = ImageOps.exif_transpose(opened).convert("RGB")
                rgb_image.load()
                size = rgb_image.size
                image_array = np.asarray(rgb_image, dtype=np.uint8).copy()
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"Could not decode image: {path.name}") from error
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ValueError("All uploaded images must have the same dimensions")
        images.append(image_array)

    return validate_rgb_images(np.stack(images, axis=0))


def load_dataset_images(
    config: DatasetConfig,
    *,
    bundled_path: str | Path | None = None,
    max_total_bytes: int | None = None,
) -> np.ndarray:
    """Dispatch a typed source configuration to one validated image loader."""

    if config.kind is DatasetKind.BUNDLED_RABBITS:
        if config.paths:
            raise ValueError("Bundled rabbit data does not accept uploaded paths")
        return load_bundled_images(bundled_path)
    if config.kind is DatasetKind.NPZ:
        if len(config.paths) != 1:
            raise ValueError("NPZ data requires exactly one archive")
        return load_npz_images(config.paths[0], max_total_bytes=max_total_bytes)
    if config.kind is DatasetKind.UPLOADED_IMAGES:
        return load_image_files(config.paths, max_total_bytes=max_total_bytes)
    raise ValueError(f"Unsupported dataset kind: {config.kind}")


def extract_dino_activations(
    images: np.ndarray,
    *,
    device: str = "auto",
    batch_size: int = 64,
) -> np.ndarray:
    """Delegate fixed-backbone extraction to upstream and validate its tensor shape."""

    if batch_size <= 0:
        raise ValueError("Activation extraction batch size must be greater than zero")
    try:
        requested_device = validate_device_setting(device)
    except ValueError as error:
        raise ValueError(
            "Extraction device must be auto, cpu, cuda, or cuda:<index>"
        ) from error
    if requested_device.startswith("cuda"):
        resolve_torch_device(requested_device)
    # A contiguous image copy can be large, so reject the device before allocating it.
    valid_images = validate_rgb_images(images)

    # Import lazily so image-only validation remains usable without model downloads.
    from bsf import data as upstream_data

    activations = upstream_data.dino_activations(
        valid_images,
        device=None if requested_device == "auto" else requested_device,
        batch_size=batch_size,
        model_id=DINO_MODEL_ID,
        revision=DINO_REVISION,
    )
    acts = np.asarray(activations, dtype=np.float32)
    expected_tail = tuple(upstream_data.POS_MEAN.shape)
    if (
        acts.ndim != 3
        or acts.shape[0] != len(valid_images)
        or acts.shape[1:] != expected_tail
    ):
        raise ValueError(
            "DINO activations must have shape "
            f"(N, {expected_tail[0]}, {expected_tail[1]}); received {acts.shape}"
        )
    if not np.isfinite(acts).all():
        raise ValueError("DINO activations contain non-finite values")
    return np.ascontiguousarray(acts)


def patch_grid(n_patches: int) -> int:
    """Return the side of a square patch grid, raising instead of asserting."""

    if n_patches <= 0:
        raise ValueError("Patch count must be greater than zero")
    grid = int(round(n_patches**0.5))
    if grid * grid != n_patches:
        raise ValueError(f"{n_patches} patches do not form a square grid")
    return grid


def center_and_scale_activations(
    activations: np.ndarray,
    *,
    positional_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the exact centering/scaling convention from the upstream README.

    Per-position centering removes DINO's positional main effect. The flattened
    token matrix is then scaled until its mean squared row norm equals ``d``.
    """

    acts = np.asarray(activations, dtype=np.float32)
    if acts.ndim != 3 or not all(size > 0 for size in acts.shape):
        raise ValueError("Activations must have nonempty shape (N, patches, features)")
    if not np.isfinite(acts).all():
        raise ValueError("Activations contain non-finite values")
    if positional_mean is None:
        from bsf.data import POS_MEAN

        positional_mean = POS_MEAN
    mean = np.asarray(positional_mean, dtype=np.float32)
    if mean.shape != acts.shape[1:]:
        raise ValueError(
            f"Positional mean shape {mean.shape} does not match activations {acts.shape[1:]}"
        )
    if not np.isfinite(mean).all():
        raise ValueError("Positional mean contains non-finite values")

    centered = (acts - mean).reshape(-1, acts.shape[-1])
    root_mean_squared_norm = float(np.sqrt(np.square(centered).sum(1).mean()))
    if not np.isfinite(root_mean_squared_norm) or root_mean_squared_norm <= 0:
        raise ValueError("Centered activations have zero energy and cannot be scaled")
    scaled = centered / root_mean_squared_norm * np.sqrt(centered.shape[1])
    return np.ascontiguousarray(scaled, dtype=np.float32)


def preprocess_dino_activations(activations: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the training matrix and visualization grid as one reusable phase."""

    return center_and_scale_activations(activations), patch_grid(activations.shape[1])


__all__ = [
    "SUPPORTED_IMAGE_SUFFIXES",
    "bundled_rabbit_path",
    "center_and_scale_activations",
    "extract_dino_activations",
    "load_bundled_images",
    "load_dataset_images",
    "load_image_files",
    "load_npz_images",
    "patch_grid",
    "preprocess_dino_activations",
    "validate_rgb_images",
]
