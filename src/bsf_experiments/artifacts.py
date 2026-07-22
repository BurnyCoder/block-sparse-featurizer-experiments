"""Safe, reproducible files produced by one local BSF experiment run.

Global context
--------------
The outer application owns checkpoints and exports while the pinned research
submodule remains unchanged. Checkpoints store only a CPU ``state_dict`` and a
strictly allowlisted primitive configuration. Loading explicitly uses
``weights_only=True`` and ``map_location="cpu"`` as recommended by PyTorch:
https://docs.pytorch.org/docs/stable/generated/torch.load.html.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4
import zipfile

import numpy as np
import torch

from .logging_utils import sanitize_for_logging
from .model_phase import validate_model_config
from .types import FeaturizerKind, ModelConfig


CHECKPOINT_FORMAT_VERSION = 1
_CHECKPOINT_FIELDS = frozenset(
    {"format_version", "input_dim", "model_config", "state_dict"}
)
_MODEL_CONFIG_FIELDS = frozenset(
    {
        "kind",
        "n_groups",
        "group_size",
        "l0",
        "coef",
        "target_l0",
        "gain",
        "paper_version",
    }
)
_SAFE_RUN_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_ARRAY_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _safe_slug(value: str, *, fallback: str) -> str:
    """Convert a human label into one bounded filesystem-safe path component."""

    slug = _SAFE_RUN_NAME.sub("-", str(value)).strip("-.").lower()
    return (slug or fallback)[:80]


def _leaf_name(value: str, *, expected_suffix: str | None = None) -> str:
    """Reject traversal, hidden files, and unexpected extensions for an export."""

    name = str(value)
    if Path(name).name != name or not _SAFE_FILE_NAME.fullmatch(name):
        raise ValueError(f"Artifact filename must be one safe path component: {value}")
    if expected_suffix is not None and Path(name).suffix.lower() != expected_suffix:
        raise ValueError(f"Artifact filename must end with {expected_suffix}")
    return name


def _positive_integer(value: object, name: str) -> int:
    """Return a positive plain integer, rejecting booleans accepted by ``int``."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str, *, nonnegative: bool = False) -> float:
    """Return a finite primitive number with an optional nonnegative bound."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or (nonnegative and parsed < 0):
        qualifier = "finite nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be a {qualifier} number")
    return parsed


def _serialize_model_config(config: ModelConfig) -> dict[str, str | int | float | bool]:
    """Convert a typed config to primitives accepted by restricted unpickling."""

    try:
        kind = FeaturizerKind(config.kind)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported featurizer kind: {config.kind}") from error
    paper_version = config.paper_version
    if not isinstance(paper_version, bool):
        raise ValueError("paper_version must be a boolean")
    return {
        "kind": kind.value,
        "n_groups": _positive_integer(config.n_groups, "n_groups"),
        "group_size": _positive_integer(config.group_size, "group_size"),
        "l0": _positive_integer(config.l0, "l0"),
        "coef": _finite_number(config.coef, "coef", nonnegative=True),
        "target_l0": _positive_integer(config.target_l0, "target_l0"),
        "gain": _finite_number(config.gain, "gain"),
        "paper_version": paper_version,
    }


def _parse_model_config(value: object, *, input_dim: int) -> ModelConfig:
    """Validate the exact checkpoint config schema before constructing a dataclass."""

    if type(value) is not dict:
        raise ValueError("Checkpoint model_config must be a plain dictionary")
    fields = set(value)
    if fields != _MODEL_CONFIG_FIELDS:
        raise ValueError(
            "Checkpoint model_config fields do not match the allowlist: "
            f"expected {sorted(_MODEL_CONFIG_FIELDS)}, received {sorted(map(str, fields))}"
        )
    try:
        kind = FeaturizerKind(value["kind"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Checkpoint contains an unsupported featurizer kind"
        ) from error
    paper_version = value["paper_version"]
    if not isinstance(paper_version, bool):
        raise ValueError("Checkpoint paper_version must be a boolean")
    config = ModelConfig(
        kind=kind,
        n_groups=_positive_integer(value["n_groups"], "n_groups"),
        group_size=_positive_integer(value["group_size"], "group_size"),
        l0=_positive_integer(value["l0"], "l0"),
        coef=_finite_number(value["coef"], "coef", nonnegative=True),
        target_l0=_positive_integer(value["target_l0"], "target_l0"),
        gain=_finite_number(value["gain"], "gain"),
        paper_version=paper_version,
    )
    validate_model_config(config, input_dim=input_dim)
    return config


def _cpu_state_dict(model: Any) -> dict[str, torch.Tensor]:
    """Copy only named tensors from a module state dictionary onto CPU."""

    state_dict = model.state_dict()
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Model state_dict must be a nonempty mapping")
    copied: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Model state_dict keys must be nonempty strings")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Model state_dict value is not a tensor: {key}")
        copied[key] = value.detach().cpu().clone()
    return copied


def _expected_state_shapes(
    config: ModelConfig, *, input_dim: int
) -> dict[str, tuple[int, ...]]:
    """Describe the exact tensor schema without allocating a candidate model."""

    width = config.n_groups * config.group_size
    if config.kind is FeaturizerKind.GRASSMANNIAN:
        return {
            "B_raw": (config.n_groups, input_dim, config.group_size),
            "gamma": (config.n_groups,),
        }
    common = {
        "W_dec": (width, input_dim),
        "W_enc": (input_dim, width),
        "b_enc": (width,),
    }
    if config.kind is FeaturizerKind.VANILLA:
        return common
    if config.paper_version:
        return {**common, "log_theta": ()}
    return {
        **common,
        "raw_theta": (config.n_groups,),
        "bandwidth": (),
        "inited": (),
    }


def _validate_state_dict_schema(
    state_dict: Mapping[str, torch.Tensor],
    config: ModelConfig,
    *,
    input_dim: int,
) -> None:
    """Reject impossible tensor metadata before a model allocation occurs.

    PyTorch recommends validating untrusted checkpoint inputs even with
    ``weights_only=True`` because restricted unpickling is not a resource-usage
    sandbox: https://docs.pytorch.org/docs/stable/notes/serialization.html.
    """

    expected = _expected_state_shapes(config, input_dim=input_dim)
    if set(state_dict) != set(expected):
        raise ValueError(
            "Checkpoint state_dict keys do not match the selected model: "
            f"expected {sorted(expected)}, received {sorted(state_dict)}"
        )
    for name, expected_shape in expected.items():
        tensor = state_dict[name]
        if tensor.layout is not torch.strided:
            raise ValueError(f"Checkpoint tensor must use dense strided layout: {name}")
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Checkpoint tensor {name} has shape {actual_shape}; "
                f"expected {expected_shape}"
            )
        if name == "inited":
            if tensor.dtype is not torch.bool:
                raise ValueError("Checkpoint tensor inited must use bool dtype")
        elif not tensor.dtype.is_floating_point:
            raise ValueError(f"Checkpoint tensor must be floating point: {name}")


def _validate_model_matches_config(
    model: Any,
    config: ModelConfig,
    *,
    input_dim: int,
) -> None:
    """Reject metadata that would rebuild behavior different from ``model``.

    The attribute list mirrors the constructors in the pinned upstream sources:
    https://github.com/BurnyCoder/block-sparse-featurizer/tree/583bb538e4bec89cb046a3a8bd0b913f6245e594/bsf.
    """

    import bsf

    expected_classes = {
        FeaturizerKind.GRASSMANNIAN: bsf.GrassmannianBSF,
        FeaturizerKind.GROUP_LASSO: bsf.GroupLassoBSF,
        FeaturizerKind.VANILLA: bsf.VanillaBSF,
    }
    expected_class = expected_classes[config.kind]
    if type(model) is not expected_class:
        raise ValueError(
            "Checkpoint model kind does not match the config: "
            f"expected {expected_class.__name__}, received {type(model).__name__}"
        )

    expected_attributes: dict[str, int | float | bool] = {
        "d": input_dim,
        "n_groups": config.n_groups,
        "group_size": config.group_size,
    }
    if config.kind in (FeaturizerKind.GRASSMANNIAN, FeaturizerKind.VANILLA):
        expected_attributes["l0"] = config.l0
    else:
        expected_attributes.update(
            {
                "coef": config.coef,
                "target_l0": config.target_l0,
                "gain": config.gain,
                "paper_version": config.paper_version,
            }
        )

    for attribute, expected in expected_attributes.items():
        actual = getattr(model, attribute, None)
        if isinstance(expected, bool):
            matches = isinstance(actual, bool) and actual is expected
        elif isinstance(expected, int):
            matches = (
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and actual == expected
            )
        else:
            matches = (
                not isinstance(actual, bool)
                and isinstance(actual, (int, float))
                and actual == expected
            )
        if not matches:
            raise ValueError(
                f"Checkpoint {attribute} does not match the model: "
                f"expected {expected!r}, received {actual!r}"
            )


def _atomic_torch_save(payload: object, destination: Path) -> None:
    """Replace a checkpoint only after ``torch.save`` completes successfully."""

    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(
    destination: str | Path,
    model: Any,
    config: ModelConfig,
    *,
    input_dim: int | None = None,
) -> Path:
    """Save a portable tensor-only model checkpoint with validated config metadata."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    dimension = getattr(model, "d", None) if input_dim is None else input_dim
    dimension = _positive_integer(dimension, "input_dim")
    serialized_config = _serialize_model_config(config)
    validated_config = _parse_model_config(serialized_config, input_dim=dimension)
    _validate_model_matches_config(
        model,
        validated_config,
        input_dim=dimension,
    )

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "input_dim": dimension,
        "model_config": serialized_config,
        "state_dict": _cpu_state_dict(model),
    }
    _validate_state_dict_schema(
        payload["state_dict"], validated_config, input_dim=dimension
    )
    _atomic_torch_save(payload, path)
    return path


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Schema-checked checkpoint content ready for controlled model construction."""

    model_config: ModelConfig
    input_dim: int
    state_dict: dict[str, torch.Tensor]

    def build_model(self, *, strict: bool = True) -> Any:
        """Construct the allowlisted BSF class and apply its tensor state."""

        from .model_phase import create_model

        model = create_model(self.model_config, input_dim=self.input_dim)
        model.load_state_dict(self.state_dict, strict=strict)
        model.eval()
        return model


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Load and schema-check a checkpoint without allowing arbitrary Python globals.

    ``weights_only=True`` narrows unpickling to tensors and primitive containers.
    PyTorch notes that it is not a complete denial-of-service defense, so callers
    should still enforce upload-size limits before invoking this function:
    https://docs.pytorch.org/docs/stable/notes/serialization.html#torch-load-with-weights-only-true.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if type(payload) is not dict:
        raise ValueError("Checkpoint must contain a plain dictionary")
    fields = set(payload)
    if fields != _CHECKPOINT_FIELDS:
        raise ValueError(
            "Checkpoint fields do not match the allowlist: "
            f"expected {sorted(_CHECKPOINT_FIELDS)}, received {sorted(map(str, fields))}"
        )
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {payload['format_version']}")
    input_dim = _positive_integer(payload["input_dim"], "input_dim")
    config = _parse_model_config(payload["model_config"], input_dim=input_dim)
    raw_state = payload["state_dict"]
    if type(raw_state) is not dict or not raw_state:
        raise ValueError("Checkpoint state_dict must be a nonempty plain dictionary")
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Checkpoint state_dict keys must be nonempty strings")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Checkpoint state_dict value is not a tensor: {key}")
        state_dict[key] = value.detach().cpu()
    _validate_state_dict_schema(state_dict, config, input_dim=input_dim)
    return LoadedCheckpoint(config, input_dim, state_dict)


def restore_checkpoint(
    path: str | Path, *, strict: bool = True
) -> tuple[Any, ModelConfig]:
    """Load one safe checkpoint and return its instantiated model and config."""

    checkpoint = load_checkpoint(path)
    return checkpoint.build_model(strict=strict), checkpoint.model_config


def _json_default(value: object) -> object:
    """Convert common experiment records while rejecting opaque runtime objects."""

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Result value is not JSON serializable: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    """An isolated timestamped directory with bounded export helpers."""

    output_root: Path
    run_dir: Path

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        run_name: str = "experiment",
        *,
        now: datetime | None = None,
    ) -> ArtifactStore:
        """Atomically create a UTC timestamped run directory, suffixing collisions."""

        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        timestamp = now or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("Artifact timestamp must include a timezone")
        utc_time = timestamp.astimezone(UTC)
        prefix = utc_time.strftime("%Y%m%dT%H%M%S.%fZ")
        base = f"{prefix}-{_safe_slug(run_name, fallback='experiment')}"
        for collision in range(1_000):
            suffix = "" if collision == 0 else f"-{collision:02d}"
            candidate = root / f"{base}{suffix}"
            try:
                candidate.mkdir(mode=0o750)
            except FileExistsError:
                continue
            return cls(output_root=root, run_dir=candidate)
        raise FileExistsError(
            f"Could not allocate a unique artifact directory for {base}"
        )

    def artifact_path(self, filename: str, *, suffix: str | None = None) -> Path:
        """Return a validated leaf path inside this run without creating the file."""

        return self.run_dir / _leaf_name(filename, expected_suffix=suffix)

    def save_checkpoint(
        self,
        model: Any,
        config: ModelConfig,
        *,
        input_dim: int | None = None,
        filename: str = "checkpoint.pt",
    ) -> Path:
        """Save a validated model checkpoint inside this run directory."""

        return save_checkpoint(
            self.artifact_path(filename, suffix=".pt"),
            model,
            config,
            input_dim=input_dim,
        )

    def save_arrays(
        self,
        arrays: Mapping[str, Any],
        *,
        filename: str = "arrays.npz",
    ) -> Path:
        """Compress named numeric arrays after rejecting pickled object values.

        ``savez_compressed`` has no ``allow_pickle`` option; NumPy exposes that
        protection when loading instead. Rejecting object dtypes here ensures
        every value can later be loaded with ``allow_pickle=False``:
        https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html.
        """

        if not arrays:
            raise ValueError("Select at least one array to export")
        prepared: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            if not isinstance(name, str) or not _SAFE_ARRAY_NAME.fullmatch(name):
                raise ValueError(f"Array name is not a safe identifier: {name}")
            if name == "allow_pickle":
                raise ValueError("allow_pickle is reserved by NumPy")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError(f"Object arrays cannot be exported safely: {name}")
            prepared[name] = array

        destination = self.artifact_path(filename, suffix=".npz")
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **prepared)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def save_figure(
        self,
        figure: Any,
        *,
        stem: str = "concepts",
        dpi: int = 150,
    ) -> dict[str, Path]:
        """Export one Matplotlib-compatible figure as both PNG and vector PDF.

        ``Figure.savefig`` is Matplotlib's documented noninteractive export API:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html.
        """

        if not callable(getattr(figure, "savefig", None)):
            raise TypeError("figure must provide a savefig method")
        _positive_integer(dpi, "dpi")
        safe_stem = _safe_slug(stem, fallback="concepts")
        exports: dict[str, Path] = {}
        for format_name in ("png", "pdf"):
            destination = self.artifact_path(f"{safe_stem}.{format_name}")
            figure.savefig(
                destination, format=format_name, dpi=dpi, bbox_inches="tight"
            )
            if not destination.is_file() or destination.stat().st_size == 0:
                raise OSError(f"Figure export was not written: {destination}")
            exports[format_name] = destination
        return exports

    def save_result_bundle(
        self,
        result: Mapping[str, Any],
        *,
        filename: str = "result-bundle.zip",
    ) -> Path:
        """Write ``result.json`` and zip every regular artifact from this run."""

        if not isinstance(result, Mapping):
            raise TypeError("result must be a mapping")
        result_path = self.artifact_path("result.json", suffix=".json")
        sanitized_result = sanitize_for_logging(dict(result))
        serialized = json.dumps(
            sanitized_result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=_json_default,
        )
        temporary_json = result_path.with_name(f".{result_path.name}.{uuid4().hex}.tmp")
        try:
            temporary_json.write_text(serialized + "\n", encoding="utf-8")
            os.replace(temporary_json, result_path)
        finally:
            temporary_json.unlink(missing_ok=True)

        destination = self.artifact_path(filename, suffix=".zip")
        temporary_zip = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            with zipfile.ZipFile(
                temporary_zip,
                mode="x",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as bundle:
                for artifact in sorted(self.run_dir.rglob("*")):
                    if artifact in {destination, temporary_zip}:
                        continue
                    if artifact.is_file() and not artifact.is_symlink():
                        bundle.write(artifact, artifact.relative_to(self.run_dir))
            os.replace(temporary_zip, destination)
        finally:
            temporary_zip.unlink(missing_ok=True)
        return destination


__all__ = [
    "ArtifactStore",
    "CHECKPOINT_FORMAT_VERSION",
    "LoadedCheckpoint",
    "load_checkpoint",
    "restore_checkpoint",
    "save_checkpoint",
]
