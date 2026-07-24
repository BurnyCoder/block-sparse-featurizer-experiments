"""Train, evidence, and stage the four exact upstream BSF recipes.

Global context
--------------
This module is the release boundary for pretrained checkpoints.  It deliberately
calls the original Goodfire constructors, preprocessing operations, and
``bsf.train`` signatures instead of the application's seeded training adapter.
Each recipe is intended to run through a fresh ``python -m`` process, which also
guarantees that ``bsf`` is imported from the verified detached upstream checkout.

Hugging Face publication remains an explicit second step.  Helpers return
reviewable ``hf`` CLI argument tuples and rely on the CLI's documented
environment/credential-store authentication rather than placing tokens in
process arguments:
https://huggingface.co/docs/huggingface_hub/en/guides/cli
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import einops
import numpy as np
import torch

from .analysis_phase import evaluate_reconstruction
from .artifacts import (
    CHECKPOINT_FORMAT_VERSION,
    ArtifactStore,
    restore_checkpoint,
    save_checkpoint,
)
from .backbone_identity import DINO_MODEL_ID, DINO_REVISION
from .reproduction import (
    TimestampedLogger,
    capture_output,
    environment_secrets,
    sanitize_text,
)
from .types import FeaturizerKind, ModelConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "outputs" / "hub-release"
DEFAULT_EXACT_SOURCE_ROOT = PROJECT_ROOT / "outputs" / "upstream-goodfire-0bf2d9a"

UPSTREAM_REPOSITORY = "https://github.com/goodfire-ai/block-sparse-featurizer"
APPLICATION_REPOSITORY = (
    "https://github.com/BurnyCoder/block-sparse-featurizer-experiments"
)
UPSTREAM_COMMIT = "0bf2d9a6ae959452d57bc169374c8902135e0f02"
# Retain the release-specific public name used by manifests and existing callers.
EXPECTED_DINO_REVISION = DINO_REVISION
EXPECTED_GOODFIRE_LICENSE_SHA256 = (
    "8f203fe135347c3a1c997bfdeab1ddee5e97f02662edad135d19e63333d76961"
)
EXPECTED_DINO_LICENSE_SHA256 = (
    "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"
)
MINIMUM_RECONSTRUCTION_R2 = 0.70
MAX_RELEASE_CHECKPOINT_BYTES = 16 * 1024 * 1024
COLLECTION_SLUG = (
    "BurnyCoder/block-sparse-featurizers-on-dinov3-rabbits-6a629047facccb1d34e808c2"
)
DEFAULT_COLLECTION_URL = f"https://huggingface.co/collections/{COLLECTION_SLUG}"

# Git object IDs are content-addressed proof that every executable/input source
# matches the exact upstream revision, even when the checkout was made from a fork.
EXPECTED_SOURCE_BLOBS: dict[str, str] = {
    "README.md": "d76be59a54cfc06203f1b56fc90361dd880e51ae",
    "starters/01_grassmannian.ipynb": ("921e5e1018d0e5d177e27df1655d7de411d75d51"),
    "starters/02_group_lasso.ipynb": ("4575196480bb68bf6d1ecd28d96ea2f05fd4200d"),
    "starters/03_vanilla.ipynb": "b132cde470b0716b7796566f0b06a871fab5bfd7",
    "bsf/data.py": "99a147ddc3ec6182d8aba2a13b514f1d7171b276",
    "bsf/train.py": "eca1533d5eb58e4a73b481a85626626ea9ac376e",
    "bsf/grassmannian.py": "0dda05b1df8d9299f0248d019d3758e560e1da2a",
    "bsf/group_lasso.py": "fcd5eedf10b764010870d25427eb4ab2b6f82a39",
    "bsf/vanilla.py": "7851ebde14e2d4c58fe35e222e0ec5773b923106",
    "rabbit.npz": "004d22c31b1208a9a403c99834cee528c8a26ee9",
    "bsf/pos_mean.npy": "44d19799957751f45c619a0791361943be6162bd",
    "LICENSE": "1e044f6e143a217dec9281ed0640c00c113c7efa",
}

STAGED_FILENAMES = frozenset(
    {
        "checkpoint.pt",
        "manifest.json",
        "README.md",
        "LICENSE-goodfire.txt",
        "LICENSE-dinov3.md",
    }
)
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReleaseRecipe:
    """One constructor/trainer invocation copied from an upstream example."""

    recipe_id: str
    title: str
    repo_id: str
    source_path: str
    constructor_name: str
    constructor_kwargs: Mapping[str, int | str]
    train_kwargs: Mapping[str, int | float]
    model_config: ModelConfig
    effective_epochs: int
    effective_lr: float
    effective_batch_size: int = 2048
    effective_snr: float = 0.1


def _frozen(values: Mapping[str, int | float | str]) -> Mapping[str, Any]:
    """Prevent accidental mutation of exact constructor/trainer argument sets."""

    return MappingProxyType(dict(values))


RELEASE_RECIPES: Mapping[str, ReleaseRecipe] = MappingProxyType(
    {
        "grassmannian-notebook": ReleaseRecipe(
            recipe_id="grassmannian-notebook",
            title="Grassmannian BSF — starter notebook",
            repo_id="BurnyCoder/bsf-dinov3-rabbits-grassmannian-notebook",
            source_path="starters/01_grassmannian.ipynb",
            constructor_name="GrassmannianBSF",
            constructor_kwargs=_frozen(
                {
                    "d": "activation_width",
                    "n_groups": 256,
                    "group_size": 3,
                    "l0": 8,
                }
            ),
            train_kwargs=_frozen({"epochs": 300, "lr": 3e-3}),
            model_config=ModelConfig(
                kind=FeaturizerKind.GRASSMANNIAN,
                n_groups=256,
                group_size=3,
                l0=8,
            ),
            effective_epochs=300,
            effective_lr=3e-3,
        ),
        "group-lasso-notebook": ReleaseRecipe(
            recipe_id="group-lasso-notebook",
            title="Group Lasso BSF — starter notebook",
            repo_id="BurnyCoder/bsf-dinov3-rabbits-group-lasso-notebook",
            source_path="starters/02_group_lasso.ipynb",
            constructor_name="GroupLassoBSF",
            constructor_kwargs=_frozen(
                {
                    "d": "activation_width",
                    "n_groups": 256,
                    "group_size": 3,
                    "target_l0": 8,
                }
            ),
            train_kwargs=_frozen({"epochs": 300}),
            model_config=ModelConfig(
                kind=FeaturizerKind.GROUP_LASSO,
                n_groups=256,
                group_size=3,
                coef=1e-2,
                target_l0=8,
                gain=10.0,
                paper_version=False,
            ),
            effective_epochs=300,
            effective_lr=4e-4,
        ),
        "vanilla-notebook": ReleaseRecipe(
            recipe_id="vanilla-notebook",
            title="Vanilla BSF — starter notebook",
            repo_id="BurnyCoder/bsf-dinov3-rabbits-vanilla-notebook",
            source_path="starters/03_vanilla.ipynb",
            constructor_name="VanillaBSF",
            constructor_kwargs=_frozen(
                {
                    "d": "activation_width",
                    "n_groups": 256,
                    "group_size": 3,
                    "l0": 8,
                }
            ),
            train_kwargs=_frozen({"epochs": 300, "lr": 3e-3}),
            model_config=ModelConfig(
                kind=FeaturizerKind.VANILLA,
                n_groups=256,
                group_size=3,
                l0=8,
            ),
            effective_epochs=300,
            effective_lr=3e-3,
        ),
        "readme-quickstart": ReleaseRecipe(
            recipe_id="readme-quickstart",
            title="Grassmannian BSF — README quickstart",
            repo_id="BurnyCoder/bsf-dinov3-rabbits-readme-quickstart",
            source_path="README.md",
            constructor_name="GrassmannianBSF",
            constructor_kwargs=_frozen(
                {"d": 768, "n_groups": 256, "group_size": 3, "l0": 16}
            ),
            train_kwargs=_frozen({"epochs": 60}),
            model_config=ModelConfig(
                kind=FeaturizerKind.GRASSMANNIAN,
                n_groups=256,
                group_size=3,
                l0=16,
            ),
            effective_epochs=60,
            effective_lr=4e-4,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class ReleaseResult:
    """Paths and evidence emitted by one gated local training run."""

    recipe_id: str
    repo_id: str
    run_dir: Path
    stage_dir: Path
    checkpoint_path: Path
    manifest_path: Path
    metrics: Mapping[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result for terminal automation."""

        return {
            "recipe_id": self.recipe_id,
            "repo_id": self.repo_id,
            "run_dir": str(self.run_dir),
            "stage_dir": str(self.stage_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "manifest_path": str(self.manifest_path),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class PublicationPlan:
    """Reviewable Hugging Face CLI commands; executing them is caller-controlled."""

    repo_id: str
    stage_dir: Path
    commands: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return command arrays so no shell interpolation is required."""

        return {
            "repo_id": self.repo_id,
            "stage_dir": str(self.stage_dir),
            "commands": [list(command) for command in self.commands],
        }


def _recipe(recipe_id: str) -> ReleaseRecipe:
    """Resolve a bounded catalog key with an actionable error."""

    try:
        return RELEASE_RECIPES[recipe_id]
    except KeyError as error:
        raise ValueError(f"Unknown release recipe: {recipe_id}") from error


def _git(root: Path, *arguments: str) -> str:
    """Run one read-only Git query without a shell or caller-controlled options."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            f"Could not verify exact upstream source with git {' '.join(arguments)}"
        ) from error
    return completed.stdout.strip()


def verify_exact_source(source_root: str | Path) -> dict[str, Any]:
    """Require an unchanged checkout of the exact Goodfire source and data blobs."""

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Exact upstream checkout does not exist: {root}")
    commit = _git(root, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"Release training requires exact upstream commit {UPSTREAM_COMMIT}; "
            f"found {commit or 'unknown'}"
        )
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ValueError("Exact upstream checkout has modified or untracked files")
    ignored_package_files = _git(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "bsf",
    )
    if ignored_package_files:
        raise ValueError(
            "Exact upstream package contains ignored shadow/cache files; "
            "remove them before release training"
        )

    found_blobs: dict[str, str] = {}
    for relative, expected_blob in EXPECTED_SOURCE_BLOBS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required upstream source is missing: {relative}")
        found = _git(root, "rev-parse", f"HEAD:{relative}")
        if found != expected_blob:
            raise ValueError(
                f"Upstream blob mismatch for {relative}: "
                f"expected {expected_blob}, found {found}"
            )
        found_blobs[relative] = found
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": commit,
        "blobs": found_blobs,
    }


def _load_exact_upstream(source_root: Path) -> Any:
    """Import ``bsf`` only when a fresh process can bind it to the exact checkout."""

    expected_package = (source_root / "bsf").resolve()
    existing = sys.modules.get("bsf")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        existing_package = (
            Path(existing_file).resolve().parent if existing_file is not None else None
        )
        if existing_package != expected_package:
            raise RuntimeError(
                "bsf was already imported from another checkout; invoke this recipe "
                "through a fresh bsf-hub-release process"
            )
        return existing

    sys.path.insert(0, str(source_root))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        upstream = importlib.import_module("bsf")
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)
    imported_file = Path(upstream.__file__).resolve().parent
    if imported_file != expected_package:
        raise RuntimeError(
            f"Imported bsf from {imported_file}; expected exact source {expected_package}"
        )
    return upstream


def resolve_dino_revision(model_id: str = DINO_MODEL_ID) -> str:
    """Resolve the gated DINO repository to an immutable full commit SHA."""

    from huggingface_hub import HfApi

    revision = HfApi().model_info(model_id).sha
    if not isinstance(revision, str) or not _FULL_SHA.fullmatch(revision):
        raise ValueError(f"Could not resolve a full DINO revision for {model_id}")
    return revision


def download_dino_license(revision: str) -> Path:
    """Download DINO's license from the same immutable revision used for evidence."""

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=DINO_MODEL_ID,
            filename="LICENSE.md",
            revision=revision,
        )
    )


def _sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading a checkpoint into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_evidence(array: np.ndarray) -> dict[str, Any]:
    """Describe and hash the exact contiguous bytes supplied to training."""

    contiguous = np.ascontiguousarray(array)
    return {
        "sha256": hashlib.sha256(memoryview(contiguous)).hexdigest(),
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
    }


def _environment_evidence() -> dict[str, Any]:
    """Capture reproducibility metadata without serializing arbitrary environment."""

    packages = {}
    for distribution in ("bsf", "numpy", "torch", "transformers", "einops"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _validate_exact_data_shapes(
    images: np.ndarray, activations: np.ndarray, matrix: np.ndarray, grid: int
) -> None:
    """Refuse to label any differently shaped dataset as the original workflow."""

    expected = {
        "images": (300, 224, 224, 3),
        "activations": (300, 196, 768),
        "matrix": (58_800, 768),
    }
    actual = {
        "images": tuple(images.shape),
        "activations": tuple(activations.shape),
        "matrix": tuple(matrix.shape),
    }
    for name, expected_shape in expected.items():
        if actual[name] != expected_shape:
            raise ValueError(
                f"Exact {name} shape must be {expected_shape}; found {actual[name]}"
            )
    if grid != 14:
        raise ValueError(f"Exact DINO patch grid must be 14; found {grid}")


def _construct_model(upstream: Any, recipe: ReleaseRecipe, input_dim: int) -> Any:
    """Make the exact upstream constructor call recorded by the selected example."""

    kwargs = {
        key: input_dim if value == "activation_width" else value
        for key, value in recipe.constructor_kwargs.items()
    }
    constructor = getattr(upstream, recipe.constructor_name)
    return constructor(**kwargs)


def _json_model_config(config: ModelConfig) -> dict[str, Any]:
    """Serialize the hardened checkpoint config using primitive values."""

    payload = asdict(config)
    payload["kind"] = config.kind.value
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable UTF-8 evidence for human review and checksum catalogs."""

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _model_card(
    recipe: ReleaseRecipe,
    manifest: Mapping[str, Any],
    collection_url: str,
) -> str:
    """Render a standalone model card with provenance, limitations, and reuse."""

    metrics = manifest["metrics"]
    source = manifest["source"]
    dino = manifest["dino"]
    recipe_enum = recipe.recipe_id.replace("-", "_").upper()
    return f"""---
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license
library_name: block-sparse-featurizer
tags:
- block-sparse-featurizer
- sparse-autoencoder
- dinov3
---

# {recipe.title}

This is a trained block-sparse featurizer reproduced from
[Goodfire's original repository]({UPSTREAM_REPOSITORY}) at immutable commit
`{source["commit"]}`. It is part of the
[Block-Sparse Featurizers on DINOv3 Rabbits collection]({collection_url}).

## Exact recipe

- Source: `{recipe.source_path}` (Git blob `{source["blobs"][recipe.source_path]}`)
- Constructor arguments: `{json.dumps(dict(recipe.constructor_kwargs), sort_keys=True)}`
- `bsf.train` arguments: `{json.dumps(dict(recipe.train_kwargs), sort_keys=True)}`
- Effective defaults: batch size {recipe.effective_batch_size}, SNR {recipe.effective_snr},
  learning rate {recipe.effective_lr}
- DINO backbone: `{DINO_MODEL_ID}` at `{dino["revision"]}`
- Reconstruction R²: {float(metrics["r2"]):.6f}
- Mean active blocks: {metrics.get("mean_l0", "not recorded")}
- Dead groups: {metrics.get("dead_groups", "not recorded")}

The full machine-readable provenance, input hashes, environment, and metrics are
in `manifest.json`.

## Loading

The hardened loader and immutable catalog live in
[`block-sparse-featurizer-experiments`]({APPLICATION_REPOSITORY}). Install that
application from a revision containing this model's Hub commit:

```bash
git clone --recurse-submodules {APPLICATION_REPOSITORY}.git
cd block-sparse-featurizer-experiments
uv sync --frozen
```

```python
from bsf_experiments.artifacts import restore_checkpoint
from bsf_experiments.hub_phase import (
    download_hub_checkpoint,
    get_hub_checkpoint_spec,
)
from bsf_experiments.types import PretrainedRecipe

spec = get_hub_checkpoint_spec(PretrainedRecipe.{recipe_enum})
path = download_hub_checkpoint(spec)
model, model_config = restore_checkpoint(path)
```

The application catalog pins a full Hub commit, file size, input width, and
SHA-256 before this code restores the checkpoint. Loading avoids BSF retraining;
new images still require the same DINOv3 patch-token extraction,
positional-mean subtraction, and RMS scaling.

## Limitations and licenses

The checkpoint was trained only on the bundled 300-image rabbit dataset and
should not be assumed to generalize to other data. Training intentionally
preserved the original unseeded stochastic behavior, so an independent
reproduction need not be bit-identical.

Goodfire's BSF software terms are in `LICENSE-goodfire.txt`. DINOv3's license,
which governs the backbone and its materials, is in `LICENSE-dinov3.md`; review
those terms before redistribution or use.
"""


def validate_staging_directory(stage_dir: str | Path) -> Path:
    """Require the curated five-file release surface and reject symlinks/extras."""

    root = Path(stage_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Release staging directory does not exist: {root}")
    entries = {path.name for path in root.iterdir()}
    missing = STAGED_FILENAMES - entries
    extra = entries - STAGED_FILENAMES
    if missing:
        raise ValueError(f"Release staging is missing files: {sorted(missing)}")
    if extra:
        raise ValueError(f"Release staging contains unexpected files: {sorted(extra)}")
    for name in STAGED_FILENAMES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Release staging entry must be a regular file: {name}")
    return root


def validate_release_bundle(
    stage_dir: str | Path,
    *,
    expected_recipe_id: str | None = None,
    expected_collection_url: str = DEFAULT_COLLECTION_URL,
) -> Path:
    """Cross-check a staged manifest against its hardened checkpoint.

    Publication planning treats a manually supplied stage as untrusted local
    input. The same restricted checkpoint loader used by the application checks
    the v1 schema before its architecture and feature width are compared with
    the manifest.
    """

    root = validate_staging_directory(stage_dir)
    checkpoint_path = root / "checkpoint.pt"
    checkpoint_size = checkpoint_path.stat().st_size
    if checkpoint_size > MAX_RELEASE_CHECKPOINT_BYTES:
        raise ValueError("Release checkpoint exceeds the publication size limit")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Release manifest must be valid UTF-8 JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Release manifest must use schema_version 1")
    metrics = manifest.get("metrics")
    r2 = metrics.get("r2") if isinstance(metrics, dict) else None
    acceptance = manifest.get("acceptance")
    if (
        isinstance(r2, bool)
        or not isinstance(r2, (int, float))
        or not math.isfinite(float(r2))
        or float(r2) < MINIMUM_RECONSTRUCTION_R2
        or not isinstance(acceptance, dict)
        or acceptance.get("passed") is not True
        or acceptance.get("minimum_reconstruction_r2") != MINIMUM_RECONSTRUCTION_R2
    ):
        raise ValueError("Release manifest does not satisfy the reconstruction R² gate")
    randomness = manifest.get("randomness")
    initial_seed = (
        randomness.get("torch_initial_seed") if isinstance(randomness, dict) else None
    )
    if (
        not isinstance(randomness, dict)
        or randomness.get("manual_seed_set") is not False
        or isinstance(initial_seed, bool)
        or not isinstance(initial_seed, int)
        or initial_seed < 0
    ):
        raise ValueError("Release manifest randomness evidence is invalid")
    timing = manifest.get("timing")
    duration = timing.get("duration_seconds") if isinstance(timing, dict) else None
    if (
        not isinstance(timing, dict)
        or not isinstance(timing.get("started_at"), str)
        or not isinstance(timing.get("finished_at"), str)
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0
    ):
        raise ValueError("Release manifest timing evidence is invalid")
    if not isinstance(manifest.get("environment"), dict) or not manifest["environment"]:
        raise ValueError("Release manifest environment evidence is missing")
    source = manifest.get("source")
    if (
        not isinstance(source, dict)
        or source.get("repository") != UPSTREAM_REPOSITORY
        or source.get("commit") != UPSTREAM_COMMIT
        or source.get("blobs") != EXPECTED_SOURCE_BLOBS
    ):
        raise ValueError("Release manifest does not identify the exact upstream commit")
    dino = manifest.get("dino")
    if (
        not isinstance(dino, dict)
        or dino.get("model_id") != DINO_MODEL_ID
        or dino.get("patch_grid") != 14
        or dino.get("revision") != EXPECTED_DINO_REVISION
    ):
        raise ValueError("Release manifest does not pin the exact DINO revision")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("Release manifest checkpoint metadata is missing")
    expected_sha256 = checkpoint.get("sha256")
    if (
        checkpoint.get("filename") != "checkpoint.pt"
        or checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION
        or isinstance(checkpoint.get("size_bytes"), bool)
        or checkpoint.get("size_bytes") != checkpoint_size
        or not isinstance(expected_sha256, str)
        or not _SHA256.fullmatch(expected_sha256)
        or _sha256_file(checkpoint_path) != expected_sha256
    ):
        raise ValueError("Release manifest checkpoint identity does not match its file")
    recipe = manifest.get("recipe")
    if not isinstance(recipe, dict) or not isinstance(recipe.get("model_config"), dict):
        raise ValueError("Release manifest model configuration is missing")
    recipe_id = recipe.get("id")
    try:
        exact_recipe = RELEASE_RECIPES[recipe_id]
    except (KeyError, TypeError) as error:
        raise ValueError("Release manifest recipe identity is invalid") from error
    if expected_recipe_id is not None and recipe_id != expected_recipe_id:
        raise ValueError(
            "Release manifest recipe does not match the selected publication target"
        )
    expected_effective = {
        "epochs": exact_recipe.effective_epochs,
        "lr": exact_recipe.effective_lr,
        "batch_size": exact_recipe.effective_batch_size,
        "snr": exact_recipe.effective_snr,
        "mixed_precision": False,
        "early_stopping": False,
        "manual_seed": False,
    }
    if (
        recipe.get("title") != exact_recipe.title
        or recipe.get("repo_id") != exact_recipe.repo_id
        or recipe.get("source_path") != exact_recipe.source_path
        or recipe.get("constructor") != exact_recipe.constructor_name
        or recipe.get("constructor_kwargs") != dict(exact_recipe.constructor_kwargs)
        or recipe.get("train_kwargs") != dict(exact_recipe.train_kwargs)
        or recipe.get("effective_hyperparameters") != expected_effective
        or recipe.get("model_config") != _json_model_config(exact_recipe.model_config)
    ):
        raise ValueError("Release manifest does not match the exact upstream recipe")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Release manifest input evidence is missing")
    for evidence_name, source_path in (
        ("rabbit_npz", "rabbit.npz"),
        ("positional_mean", "bsf/pos_mean.npy"),
    ):
        evidence = inputs.get(evidence_name)
        if (
            not isinstance(evidence, dict)
            or evidence.get("git_blob") != EXPECTED_SOURCE_BLOBS[source_path]
            or not isinstance(evidence.get("sha256"), str)
            or not _SHA256.fullmatch(evidence["sha256"])
        ):
            raise ValueError(f"Release manifest {evidence_name} evidence is invalid")
    activation_evidence = inputs.get("preprocessed_activations")
    shape = (
        activation_evidence.get("shape")
        if isinstance(activation_evidence, dict)
        else None
    )
    if (
        not isinstance(activation_evidence, dict)
        or shape != [58_800, 768]
        or activation_evidence.get("dtype") != "float64"
        or not isinstance(activation_evidence.get("sha256"), str)
        or not _SHA256.fullmatch(activation_evidence["sha256"])
    ):
        raise ValueError(
            "Release manifest must describe the exact float64 rabbit activation matrix"
        )
    collection_url = manifest.get("collection_url")
    if collection_url != expected_collection_url:
        raise ValueError("Release manifest does not reference the expected collection")
    licenses = manifest.get("licenses")
    expected_licenses = {
        "goodfire": {
            "filename": "LICENSE-goodfire.txt",
            "sha256": EXPECTED_GOODFIRE_LICENSE_SHA256,
            "git_blob": EXPECTED_SOURCE_BLOBS["LICENSE"],
        },
        "dinov3": {
            "filename": "LICENSE-dinov3.md",
            "sha256": EXPECTED_DINO_LICENSE_SHA256,
            "revision": EXPECTED_DINO_REVISION,
        },
    }
    if licenses != expected_licenses:
        raise ValueError("Release manifest license evidence is invalid")
    if (
        _sha256_file(root / "LICENSE-goodfire.txt") != EXPECTED_GOODFIRE_LICENSE_SHA256
        or _sha256_file(root / "LICENSE-dinov3.md") != EXPECTED_DINO_LICENSE_SHA256
    ):
        raise ValueError("Release license files do not match their pinned sources")
    try:
        actual_card = (root / "README.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError("Release model card must be valid UTF-8") from error
    expected_card = _model_card(exact_recipe, manifest, collection_url)
    if actual_card != expected_card:
        raise ValueError("Release model card does not match the exact manifest")
    for name in STAGED_FILENAMES - {"checkpoint.pt"}:
        try:
            staged_text = (root / name).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(
                f"Release text file must be valid UTF-8: {name}"
            ) from error
        if sanitize_text(staged_text, environment_secrets()) != staged_text:
            raise ValueError(f"Release text file contains credential material: {name}")

    model, model_config = restore_checkpoint(
        checkpoint_path,
        max_uncompressed_bytes=MAX_RELEASE_CHECKPOINT_BYTES,
    )
    if _json_model_config(model_config) != recipe["model_config"]:
        raise ValueError(
            "Release manifest model configuration does not match its checkpoint"
        )
    if int(model.d) != 768:
        raise ValueError(
            "Release manifest activation width does not match its checkpoint"
        )
    return root


def _stage_release(
    *,
    run_dir: Path,
    source_root: Path,
    checkpoint_path: Path,
    manifest: Mapping[str, Any],
    recipe: ReleaseRecipe,
    dino_license_path: Path,
    collection_url: str,
) -> Path:
    """Atomically create a directory containing only files approved for upload."""

    destination = run_dir / "stage"
    temporary = run_dir / f".stage-{uuid4().hex}"
    temporary.mkdir(mode=0o750)
    try:
        shutil.copy2(checkpoint_path, temporary / "checkpoint.pt")
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "README.md").write_text(
            _model_card(recipe, manifest, collection_url),
            encoding="utf-8",
        )
        shutil.copy2(source_root / "LICENSE", temporary / "LICENSE-goodfire.txt")
        shutil.copy2(dino_license_path, temporary / "LICENSE-dinov3.md")
        validate_release_bundle(
            temporary,
            expected_recipe_id=recipe.recipe_id,
            expected_collection_url=collection_url,
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_staging_directory(destination)


def train_exact_recipe(
    recipe_id: str,
    *,
    source_root: str | Path = DEFAULT_EXACT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_RELEASE_ROOT,
    collection_url: str = DEFAULT_COLLECTION_URL,
) -> ReleaseResult:
    """Run one exact recipe, gate its metric, and curate a publishable local folder.

    This function never calls ``manual_seed``, enables mixed precision, performs
    early stopping, or publishes remotely. The original ``bsf.train`` defaults
    remain defaults because only arguments present in the source example are
    passed to it.
    """

    recipe = _recipe(recipe_id)
    source = Path(source_root).expanduser().resolve()
    source_evidence = verify_exact_source(source)
    store = ArtifactStore.create(output_root, f"hub-release-{recipe.recipe_id}")
    run_dir = store.run_dir
    log_path = run_dir / "training.log"
    secrets = environment_secrets()
    started_at = datetime.now(UTC)
    monotonic_start = time.perf_counter()
    initial_seed = int(torch.initial_seed())

    with TimestampedLogger(log_path, secrets=secrets) as logger:
        logger.info(
            f"Starting exact release recipe={recipe.recipe_id} "
            f"source_commit={UPSTREAM_COMMIT}"
        )
        with capture_output(logger):
            upstream = _load_exact_upstream(source)
            dino_revision_before = resolve_dino_revision(DINO_MODEL_ID)
            if dino_revision_before != EXPECTED_DINO_REVISION:
                raise RuntimeError(
                    "DINO main no longer resolves to the exact release revision "
                    f"{EXPECTED_DINO_REVISION}"
                )
            images = upstream.data.load_rabbit_images(source / "rabbit.npz")
            activations = upstream.data.dino_activations(images)
            dino_revision_after = resolve_dino_revision(DINO_MODEL_ID)
            if dino_revision_after != dino_revision_before:
                raise RuntimeError(
                    "DINO repository revision changed during activation extraction"
                )

            # These are the literal centering, einops flattening, and RMS scaling
            # operations shared by the README and all three starter notebooks.
            centered = activations - upstream.data.POS_MEAN
            matrix = einops.rearrange(centered, "n p d -> (n p) d")
            matrix = (
                matrix / np.sqrt((matrix**2).sum(1).mean()) * np.sqrt(matrix.shape[1])
            )
            grid = int(upstream.data.patch_grid(activations.shape[1]))
            _validate_exact_data_shapes(images, activations, matrix, grid)

            model = _construct_model(upstream, recipe, matrix.shape[1])
            trained = upstream.train(model, matrix, **dict(recipe.train_kwargs))
            if trained is not model:
                raise ValueError(
                    "Upstream bsf.train must return the same model instance"
                )
            metrics = evaluate_reconstruction(
                model,
                matrix,
                device="auto",
                batch_size=2048,
            )

        r2 = float(metrics.get("r2", math.nan))
        if not math.isfinite(r2) or r2 < MINIMUM_RECONSTRUCTION_R2:
            failure = {
                "recipe_id": recipe.recipe_id,
                "minimum_r2": MINIMUM_RECONSTRUCTION_R2,
                "metrics": dict(metrics),
            }
            _write_json(run_dir / "failed-gate.json", failure)
            message = (
                f"Reconstruction R² gate failed: {r2!r} < "
                f"{MINIMUM_RECONSTRUCTION_R2:.2f}; no release was staged"
            )
            logger.error(message)
            raise ValueError(message)

        checkpoint_path = save_checkpoint(
            run_dir / "checkpoint.pt",
            model,
            recipe.model_config,
            input_dim=matrix.shape[1],
        )
        finished_at = datetime.now(UTC)
        duration_seconds = time.perf_counter() - monotonic_start
        checkpoint_evidence = {
            "filename": "checkpoint.pt",
            "format_version": 1,
            "sha256": _sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        }
        dino_license_path = download_dino_license(dino_revision_before)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "recipe": {
                "id": recipe.recipe_id,
                "title": recipe.title,
                "repo_id": recipe.repo_id,
                "source_path": recipe.source_path,
                "constructor": recipe.constructor_name,
                "constructor_kwargs": dict(recipe.constructor_kwargs),
                "train_kwargs": dict(recipe.train_kwargs),
                "effective_hyperparameters": {
                    "epochs": recipe.effective_epochs,
                    "lr": recipe.effective_lr,
                    "batch_size": recipe.effective_batch_size,
                    "snr": recipe.effective_snr,
                    "mixed_precision": False,
                    "early_stopping": False,
                    "manual_seed": False,
                },
                "model_config": _json_model_config(recipe.model_config),
            },
            "source": source_evidence,
            "dino": {
                "model_id": DINO_MODEL_ID,
                "revision": dino_revision_before,
                "patch_grid": grid,
            },
            "inputs": {
                "rabbit_npz": {
                    "sha256": _sha256_file(source / "rabbit.npz"),
                    "git_blob": EXPECTED_SOURCE_BLOBS["rabbit.npz"],
                },
                "positional_mean": {
                    "sha256": _sha256_file(source / "bsf/pos_mean.npy"),
                    "git_blob": EXPECTED_SOURCE_BLOBS["bsf/pos_mean.npy"],
                },
                "preprocessed_activations": _array_evidence(matrix),
            },
            "randomness": {
                "manual_seed_set": False,
                "torch_initial_seed": initial_seed,
            },
            "environment": _environment_evidence(),
            "timing": {
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_seconds": duration_seconds,
            },
            "metrics": dict(metrics),
            "acceptance": {
                "minimum_reconstruction_r2": MINIMUM_RECONSTRUCTION_R2,
                "passed": True,
            },
            "checkpoint": checkpoint_evidence,
            "licenses": {
                "goodfire": {
                    "filename": "LICENSE-goodfire.txt",
                    "sha256": _sha256_file(source / "LICENSE"),
                    "git_blob": EXPECTED_SOURCE_BLOBS["LICENSE"],
                },
                "dinov3": {
                    "filename": "LICENSE-dinov3.md",
                    "sha256": _sha256_file(dino_license_path),
                    "revision": dino_revision_before,
                },
            },
            "collection_url": collection_url,
        }
        local_manifest = run_dir / "manifest.json"
        _write_json(local_manifest, manifest)
        stage_dir = _stage_release(
            run_dir=run_dir,
            source_root=source,
            checkpoint_path=checkpoint_path,
            manifest=manifest,
            recipe=recipe,
            dino_license_path=dino_license_path,
            collection_url=collection_url,
        )
        logger.info(f"Release gate passed r2={r2:.6f}; curated staging={stage_dir}")

    return ReleaseResult(
        recipe_id=recipe.recipe_id,
        repo_id=recipe.repo_id,
        run_dir=run_dir,
        stage_dir=stage_dir,
        checkpoint_path=checkpoint_path,
        manifest_path=local_manifest,
        metrics=dict(metrics),
    )


def build_publication_plan(
    recipe_id: str,
    *,
    stage_dir: str | Path,
    collection_slug: str = COLLECTION_SLUG,
) -> PublicationPlan:
    """Build token-free create/upload/collection commands for one gated stage."""

    recipe = _recipe(recipe_id)
    stage = validate_release_bundle(
        stage_dir,
        expected_recipe_id=recipe.recipe_id,
        expected_collection_url=f"https://huggingface.co/collections/{collection_slug}",
    )
    note = (
        f"{recipe.title}; exact source {recipe.source_path}; "
        "see manifest.json for measured R²."
    )
    commands = (
        (
            "hf",
            "repos",
            "create",
            recipe.repo_id,
            "--type",
            "model",
            "--exist-ok",
        ),
        (
            "hf",
            "upload",
            recipe.repo_id,
            str(stage),
            ".",
            "--commit-message",
            f"Publish exact {recipe.recipe_id} BSF checkpoint",
        ),
        (
            "hf",
            "collections",
            "add-item",
            collection_slug,
            recipe.repo_id,
            "model",
            "--note",
            note,
            "--exists-ok",
        ),
    )
    return PublicationPlan(recipe.repo_id, stage, commands)


def fresh_process_train_command(
    recipe_id: str,
    *,
    source_root: str | Path = DEFAULT_EXACT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_RELEASE_ROOT,
    collection_url: str = DEFAULT_COLLECTION_URL,
) -> tuple[str, ...]:
    """Return a no-shell command that trains one recipe in a fresh interpreter."""

    _recipe(recipe_id)
    return (
        sys.executable,
        "-m",
        "bsf_experiments.hub_release",
        "train-one",
        recipe_id,
        "--source-root",
        str(Path(source_root).expanduser().resolve()),
        "--output-root",
        str(Path(output_root).expanduser().resolve()),
        "--collection-url",
        collection_url,
    )


def _parser() -> argparse.ArgumentParser:
    """Build the module/console-script command parser."""

    parser = argparse.ArgumentParser(
        prog="bsf-hub-release",
        description="Train and stage exact upstream BSF checkpoints.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser(
        "train-one", help="Train one exact recipe and emit its local result as JSON."
    )
    train.add_argument("recipe", choices=tuple(RELEASE_RECIPES))
    train.add_argument("--source-root", type=Path, default=DEFAULT_EXACT_SOURCE_ROOT)
    train.add_argument("--output-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    train.add_argument("--collection-url", default=DEFAULT_COLLECTION_URL)

    publish = subcommands.add_parser(
        "plan-publish", help="Emit reviewable token-free hf CLI command arrays."
    )
    publish.add_argument("recipe", choices=tuple(RELEASE_RECIPES))
    publish.add_argument("--stage-dir", type=Path, required=True)
    publish.add_argument("--collection-slug", default=COLLECTION_SLUG)

    all_commands = subcommands.add_parser(
        "plan-train-all", help="Emit four fresh-process training commands."
    )
    all_commands.add_argument(
        "--source-root", type=Path, default=DEFAULT_EXACT_SOURCE_ROOT
    )
    all_commands.add_argument("--output-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    all_commands.add_argument("--collection-url", default=DEFAULT_COLLECTION_URL)
    return parser


def release_main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for exact local training and publication planning."""

    arguments = _parser().parse_args(argv)
    # The CLI keeps the ignored token in the environment, where Transformers and
    # the ``hf`` command discover it without copying it into argv or JSON.
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
        if arguments.command == "train-one":
            result = train_exact_recipe(
                arguments.recipe,
                source_root=arguments.source_root,
                output_root=arguments.output_root,
                collection_url=arguments.collection_url,
            )
            payload: Any = result.to_dict()
        elif arguments.command == "plan-publish":
            payload = build_publication_plan(
                arguments.recipe,
                stage_dir=arguments.stage_dir,
                collection_slug=arguments.collection_slug,
            ).to_dict()
        else:
            payload = {
                "commands": [
                    list(
                        fresh_process_train_command(
                            recipe_id,
                            source_root=arguments.source_root,
                            output_root=arguments.output_root,
                            collection_url=arguments.collection_url,
                        )
                    )
                    for recipe_id in RELEASE_RECIPES
                ]
            }
    except Exception as error:
        message = sanitize_text(
            f"{type(error).__name__}: {error}",
            environment_secrets(),
        )
        print(json.dumps({"ok": False, "error": message}), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(release_main())


__all__ = [
    "COLLECTION_SLUG",
    "DEFAULT_COLLECTION_URL",
    "DEFAULT_EXACT_SOURCE_ROOT",
    "DEFAULT_RELEASE_ROOT",
    "DINO_MODEL_ID",
    "EXPECTED_DINO_LICENSE_SHA256",
    "EXPECTED_DINO_REVISION",
    "EXPECTED_GOODFIRE_LICENSE_SHA256",
    "EXPECTED_SOURCE_BLOBS",
    "MINIMUM_RECONSTRUCTION_R2",
    "MAX_RELEASE_CHECKPOINT_BYTES",
    "PublicationPlan",
    "RELEASE_RECIPES",
    "ReleaseRecipe",
    "ReleaseResult",
    "STAGED_FILENAMES",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "build_publication_plan",
    "download_dino_license",
    "fresh_process_train_command",
    "release_main",
    "resolve_dino_revision",
    "train_exact_recipe",
    "validate_release_bundle",
    "validate_staging_directory",
    "verify_exact_source",
]
