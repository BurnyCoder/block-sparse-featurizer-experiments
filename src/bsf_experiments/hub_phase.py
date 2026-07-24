"""Resolve allowlisted BSF checkpoints from immutable Hugging Face revisions.

Global context
--------------
The collection page is a mutable discovery surface, so runtime downloads use a
local catalog of full commit hashes and SHA-256 digests instead. Hugging Face's
documented ``dry_run`` result exposes the remote size before transfer:
https://huggingface.co/docs/huggingface_hub/package_reference/file_download.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any

from huggingface_hub import hf_hub_download

from .model_phase import validate_model_config
from .presets import PRESETS
from .types import ModelConfig, PretrainedRecipe


_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_HUB_CHECKPOINT_MAX_BYTES = 16 * 1024 * 1024

HubDownloader = Callable[..., Any]
HubMetadataCallback = Callable[["HubDownloadMetadata"], None]


@dataclass(frozen=True, slots=True)
class HubCheckpointSpec:
    """Trusted identity, resource budget, and model schema for one Hub artifact."""

    repo_id: str
    revision: str
    filename: str
    sha256: str
    max_bytes: int
    input_dim: int
    model_config: ModelConfig


@dataclass(frozen=True, slots=True)
class HubDownloadMetadata:
    """Safe preflight fields suitable for complete session logging."""

    repo_id: str
    filename: str
    resolved_commit: str
    remote_size: int
    is_cached: bool


def _published_spec(
    repo_name: str,
    preset_name: str,
    *,
    revision: str,
    sha256: str,
) -> HubCheckpointSpec:
    """Build one immutable record for a quality-gated public checkpoint."""

    return HubCheckpointSpec(
        repo_id=f"BurnyCoder/{repo_name}",
        revision=revision,
        filename="checkpoint.pt",
        sha256=sha256,
        max_bytes=DEFAULT_HUB_CHECKPOINT_MAX_BYTES,
        input_dim=768,
        model_config=PRESETS[preset_name].model,
    )


# ``MappingProxyType`` provides a read-only view over a private dictionary:
# https://docs.python.org/3/library/types.html#types.MappingProxyType.
CHECKPOINT_CATALOG: Mapping[PretrainedRecipe, HubCheckpointSpec] = MappingProxyType(
    {
        PretrainedRecipe.GRASSMANNIAN_NOTEBOOK: _published_spec(
            "bsf-dinov3-rabbits-grassmannian-notebook",
            "grassmannian_notebook",
            revision="6d874cd7c713d0464ec1769cb667f08aeb43720e",
            sha256=("f029890c4fa34fe9dcaf350d03870b5b3f035daa3d1fe97c457299d76754748d"),
        ),
        PretrainedRecipe.GROUP_LASSO_NOTEBOOK: _published_spec(
            "bsf-dinov3-rabbits-group-lasso-notebook",
            "group_lasso_notebook",
            revision="c0e9c501963ed28d022ecce5fd7b7beafed4720f",
            sha256=("46d8d0a68e263f4518350f9334959d3d349bf26d347be013f748da8aa660fde8"),
        ),
        PretrainedRecipe.VANILLA_NOTEBOOK: _published_spec(
            "bsf-dinov3-rabbits-vanilla-notebook",
            "vanilla_notebook",
            revision="bcfdceb086e57f2d5f64d0036a1d08cdc8610442",
            sha256=("b87e2a9548abf5c909152ef2f7f89085bbd614a81981db24e976362849aa9d06"),
        ),
        PretrainedRecipe.README_QUICKSTART: _published_spec(
            "bsf-dinov3-rabbits-readme-quickstart",
            "readme",
            revision="4f1fc7b7ce3da7c8a41325fc06ee4d3aee11a2ca",
            sha256=("449e7dfe65587f2959d8263197805e2f2f33cc7c391dbeb7949539fb58a8e321"),
        ),
    }
)


def _validate_spec(spec: HubCheckpointSpec, *, recipe: PretrainedRecipe) -> None:
    """Fail before network access when any trusted catalog field is incomplete."""

    if not isinstance(spec, HubCheckpointSpec):
        raise TypeError(f"Hub catalog entry for {recipe.value} has an invalid type")
    if not isinstance(spec.repo_id, str) or not _REPO_ID.fullmatch(spec.repo_id):
        raise ValueError(f"Invalid Hugging Face model repository ID: {spec.repo_id!r}")
    if not isinstance(spec.revision, str) or not _FULL_GIT_SHA.fullmatch(spec.revision):
        raise ValueError("Hub checkpoint revision must be a full lowercase Git SHA")
    if not isinstance(spec.filename, str) or "\\" in spec.filename:
        raise ValueError("Hub checkpoint filename must be a relative POSIX path")
    filename = PurePosixPath(spec.filename)
    if (
        filename.is_absolute()
        or filename.as_posix() != spec.filename
        or any(part in {"", ".", ".."} for part in filename.parts)
    ):
        raise ValueError("Hub checkpoint filename must be a normalized relative path")
    if not isinstance(spec.sha256, str) or not _SHA256.fullmatch(spec.sha256):
        raise ValueError(
            "Hub checkpoint SHA-256 must be 64 lowercase hexadecimal digits"
        )
    if (
        isinstance(spec.max_bytes, bool)
        or not isinstance(spec.max_bytes, int)
        or spec.max_bytes <= 0
    ):
        raise ValueError("Hub checkpoint download limit must be a positive integer")
    if (
        isinstance(spec.input_dim, bool)
        or not isinstance(spec.input_dim, int)
        or spec.input_dim <= 0
    ):
        raise ValueError("Hub checkpoint input_dim must be a positive integer")
    if not isinstance(spec.model_config, ModelConfig):
        raise TypeError("Hub checkpoint model_config must be a ModelConfig")
    validate_model_config(
        spec.model_config,
        input_dim=spec.input_dim,
    )


def get_hub_checkpoint_spec(
    recipe: PretrainedRecipe | str,
    *,
    catalog: Mapping[PretrainedRecipe, HubCheckpointSpec] | None = None,
) -> HubCheckpointSpec:
    """Return one validated catalog entry, accepting stable serialized enum values."""

    try:
        selected = PretrainedRecipe(recipe)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unknown pretrained recipe: {recipe!r}") from error
    records = CHECKPOINT_CATALOG if catalog is None else catalog
    try:
        spec = records[selected]
    except KeyError as error:
        raise ValueError(
            f"No Hugging Face checkpoint is configured for {selected.value}."
        ) from error
    _validate_spec(spec, recipe=selected)
    return spec


def _positive_file_size(value: object, *, context: str) -> int:
    """Return trustworthy size metadata without accepting booleans as integers."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must report a positive integer file size")
    return value


def _sha256_file(path: Path) -> str:
    """Hash a cached checkpoint incrementally without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_hub_checkpoint(
    spec: HubCheckpointSpec,
    *,
    downloader: HubDownloader | None = None,
    metadata_callback: HubMetadataCallback | None = None,
) -> Path:
    """Download one pinned checkpoint after remote and local resource checks.

    The client receives no explicit token, allowing public repositories to work
    anonymously while retaining Hugging Face's standard environment-based auth
    and cache behavior. ``force_download`` is intentionally absent so a verified
    cache entry can be reused.
    """

    # Validate injected records too; derive a recipe only for actionable errors.
    client = hf_hub_download if downloader is None else downloader
    matching_recipe = next(
        (
            recipe
            for recipe, catalog_spec in CHECKPOINT_CATALOG.items()
            if catalog_spec.repo_id == spec.repo_id
        ),
        PretrainedRecipe.README_QUICKSTART,
    )
    _validate_spec(spec, recipe=matching_recipe)
    common = {
        "repo_id": spec.repo_id,
        "filename": spec.filename,
        "repo_type": "model",
        "revision": spec.revision,
    }
    dry_run = client(**common, dry_run=True)
    commit_hash = getattr(dry_run, "commit_hash", None)
    if commit_hash != spec.revision:
        raise ValueError(
            "Hub checkpoint resolved to an unexpected revision; "
            f"expected {spec.revision}, received {commit_hash!r}."
        )
    advertised_size = _positive_file_size(
        getattr(dry_run, "file_size", None),
        context="Hub checkpoint preflight",
    )
    if advertised_size > spec.max_bytes:
        raise ValueError(
            "Hub checkpoint exceeds its configured download limit: "
            f"{advertised_size} > {spec.max_bytes} bytes."
        )
    is_cached = getattr(dry_run, "is_cached", False)
    if not isinstance(is_cached, bool):
        raise ValueError("Hub checkpoint preflight returned invalid cache metadata")
    if metadata_callback is not None:
        metadata_callback(
            HubDownloadMetadata(
                repo_id=spec.repo_id,
                filename=spec.filename,
                resolved_commit=commit_hash,
                remote_size=advertised_size,
                is_cached=is_cached,
            )
        )

    downloaded = client(**common, dry_run=False)
    if not isinstance(downloaded, (str, Path)):
        raise TypeError("Hugging Face download did not return a local filesystem path")
    path = Path(downloaded)
    if not path.is_file():
        raise FileNotFoundError(f"Downloaded Hub checkpoint does not exist: {path}")
    local_size = _positive_file_size(
        path.stat().st_size,
        context="Downloaded Hub checkpoint",
    )
    if local_size > spec.max_bytes:
        raise ValueError(
            "Downloaded Hub checkpoint exceeds its configured download limit: "
            f"{local_size} > {spec.max_bytes} bytes."
        )
    if local_size != advertised_size:
        raise ValueError(
            "Downloaded Hub checkpoint size does not match preflight metadata: "
            f"{local_size} != {advertised_size} bytes."
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != spec.sha256:
        raise ValueError(
            "Downloaded Hub checkpoint SHA-256 does not match the trusted catalog."
        )
    return path


__all__ = [
    "CHECKPOINT_CATALOG",
    "DEFAULT_HUB_CHECKPOINT_MAX_BYTES",
    "HubDownloadMetadata",
    "HubCheckpointSpec",
    "HubDownloader",
    "HubMetadataCallback",
    "download_hub_checkpoint",
    "get_hub_checkpoint_spec",
]
