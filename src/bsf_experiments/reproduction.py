"""Reproduce the vendored BSF README quickstart and starter notebooks.

Global context
--------------
The outer project owns orchestration and artifacts while the research code stays
unchanged in ``vendor/block-sparse-featurizer``.  This module therefore reads
the upstream examples verbatim, executes them from the upstream repository root,
and writes every generated file beneath the ignored ``outputs/runs`` directory.

Notebook execution follows nbclient's documented ``resources.metadata.path``
working-directory mechanism:
https://nbclient.readthedocs.io/en/latest/client.html
"""

from __future__ import annotations

import ast
import base64
import copy
import contextlib
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, TextIO


# Resolve paths from this installed source file instead of the caller's current
# directory so both ``uv run`` entry points and imported notebook launchers agree.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBMODULE_ROOT = PROJECT_ROOT / "vendor" / "block-sparse-featurizer"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "runs"

DINO_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_ACCESS_FILE = "config.json"
EXPECTED_SUBMODULE_COMMIT = "583bb538e4bec89cb046a3a8bd0b913f6245e594"

# These exact versions are resolved in ``uv.lock``. Checking the installed
# distributions follows importlib.metadata's documented distribution API:
# https://docs.python.org/3/library/importlib.metadata.html#distribution-versions
EXPECTED_DISTRIBUTION_VERSIONS: Mapping[str, str] = {
    "bsf": "0.1.0",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.14.1",
    "numpy": "2.5.1",
    "einops": "0.8.2",
    "matplotlib": "3.11.1",
    "nbclient": "0.10.4",
    "nbconvert": "7.17.1",
    "nbformat": "5.10.4",
    "ipykernel": "7.3.0",
    "jupyterlab": "4.6.2",
    "Pillow": "12.3.0",
    "python-dotenv": "1.2.2",
}

MINIMUM_RECONSTRUCTION_R2 = 0.70
EXPECTED_PATCH_GRID = 14
EXPECTED_NOTEBOOK_EPOCHS = 300
EXPECTED_NOTEBOOK_ACTIVATION_SHAPE = (58_800, 768)
EXPECTED_README_SHAPES: Mapping[str, tuple[int, ...]] = {
    "images_shape": (300, 224, 224, 3),
    "acts_shape": (300, 196, 768),
    "x_shape": (58_800, 768),
    "z_shape": (58_800, 256, 3),
    "atoms_shape": (256, 3, 768),
}

NOTEBOOK_RELATIVE_PATHS = (
    Path("starters/01_grassmannian.ipynb"),
    Path("starters/02_group_lasso.ipynb"),
    Path("starters/03_vanilla.ipynb"),
)

REQUIRED_ASSETS = (
    Path("README.md"),
    Path("rabbit.npz"),
    Path("bsf/__init__.py"),
    Path("bsf/pos_mean.npy"),
    *NOTEBOOK_RELATIVE_PATHS,
)

# These distributions cover every import used by the README and notebooks plus
# the programmatic execution/export layer in this module.
REQUIRED_DISTRIBUTIONS = (
    "bsf",
    "torch",
    "torchvision",
    "transformers",
    "numpy",
    "einops",
    "matplotlib",
    "nbclient",
    "nbconvert",
    "nbformat",
    "ipykernel",
    "jupyterlab",
    "Pillow",
    "python-dotenv",
)

ReproductionTarget = Literal["readme", "notebooks", "all"]

_SENSITIVE_KEY = re.compile(
    r"(?i)(?:token|password|passwd|secret|credential|authorization|api[_-]?key)"
)
_ASSIGNED_SECRET = re.compile(
    r"(?i)([\"']?(?:token|password|passwd|secret|credential|authorization|"
    r"api[_-]?key)[\"']?\s*[:=]\s*)([\"']?)([^\s,;\"']+)([\"']?)"
)
_BEARER_SECRET = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+\-/]+=*")
_HF_SECRET = re.compile(r"\bhf_[A-Za-z0-9]{10,}\b")
_TRAINING_LINE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s*/\s*(?P<epochs>\d+).*?"
    r"loss=(?P<loss>[-+0-9.eE]+).*?R2=(?P<r2>[-+0-9.eE]+).*?"
    r"L0=(?P<l0>[-+0-9.eE]+).*?dead=(?P<dead>\d+)\s*/\s*(?P<groups>\d+)"
)


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Serializable environment report that intentionally stores no secrets."""

    ok: bool
    checked_at: str
    submodule_root: str
    submodule_commit: str | None
    python_version: str
    package_versions: Mapping[str, str | None]
    assets: Mapping[str, bool]
    cuda_available: bool
    cuda_device: str | None
    hf_token_present: bool
    gated_model_access: bool | None
    submodule_clean: bool | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready copy for CLI and artifact consumers."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    """Describe one README or notebook execution and its generated artifacts."""

    name: str
    status: Literal["passed", "failed"]
    started_at: str
    finished_at: str
    output_dir: str
    log_path: str
    artifacts: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Expose success as a convenient boolean for command-line callers."""

        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready copy for suite summaries."""

        return asdict(self) | {"ok": self.ok}


@dataclass(frozen=True, slots=True)
class ReproductionSuiteResult:
    """Aggregate preflight and task results for ``readme``, ``notebooks``, or all."""

    target: ReproductionTarget
    status: Literal["passed", "failed"]
    started_at: str
    finished_at: str
    output_dir: str
    preflight: PreflightResult
    results: tuple[ReproductionResult, ...] = ()

    @property
    def ok(self) -> bool:
        """Return true only when preflight and every requested task succeeded."""

        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        """Return a nested JSON-ready suite summary."""

        return {
            "target": self.target,
            "status": self.status,
            "ok": self.ok,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output_dir": self.output_dir,
            "preflight": self.preflight.to_dict(),
            "results": [result.to_dict() for result in self.results],
        }


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for logs and JSON summaries."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sanitize_text(text: object, secrets: Iterable[str] = ()) -> str:
    """Redact known values and common credential assignments from arbitrary text.

    Structural redaction protects exception messages even if a library echoes an
    authorization header.  Exact-value replacement additionally protects tokens
    that appear without a key name.
    """

    sanitized = str(text)
    for secret in sorted({value for value in secrets if value}, key=len, reverse=True):
        sanitized = sanitized.replace(secret, "[REDACTED]")
    sanitized = _HF_SECRET.sub("[REDACTED]", sanitized)
    sanitized = _BEARER_SECRET.sub(r"\1[REDACTED]", sanitized)
    sanitized = _ASSIGNED_SECRET.sub(r"\1\2[REDACTED]\4", sanitized)
    return sanitized


def environment_secrets() -> tuple[str, ...]:
    """Collect sensitive environment values for redaction without exposing keys."""

    return tuple(
        value
        for key, value in os.environ.items()
        if value and _SENSITIVE_KEY.search(key)
    )


class TimestampedLogger:
    """Write complete sanitized messages to a run log and terminal in real time."""

    def __init__(
        self,
        path: Path | str,
        *,
        secrets: Iterable[str] = (),
        console: TextIO | None = None,
    ) -> None:
        """Open a line-buffered UTF-8 log while retaining only redaction values."""

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._secrets = tuple(value for value in secrets if value)
        self._console = console if console is not None else sys.__stdout__
        self._lock = threading.Lock()

    def log(self, level: str, message: object) -> None:
        """Timestamp and emit every line without truncating multiline output."""

        sanitized = sanitize_text(message, self._secrets)
        lines = sanitized.splitlines() or [""]
        with self._lock:
            for line in lines:
                rendered = f"{utc_timestamp()} {level.upper():<7} {line}\n"
                self._file.write(rendered)
                self._console.write(rendered)
            self._file.flush()
            self._console.flush()

    def info(self, message: object) -> None:
        """Log an informational event."""

        self.log("INFO", message)

    def warning(self, message: object) -> None:
        """Log a recoverable warning."""

        self.log("WARNING", message)

    def error(self, message: object) -> None:
        """Log a failed phase or exception."""

        self.log("ERROR", message)

    def close(self) -> None:
        """Flush and close the file handle owned by this logger."""

        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()

    def __enter__(self) -> TimestampedLogger:
        """Support deterministic logger cleanup with a context manager."""

        return self

    def __exit__(self, *_: object) -> None:
        """Close the logger when leaving its context."""

        self.close()


class _CapturedStream(io.TextIOBase):
    """Buffer partial writes so redirected output receives one timestamp per line."""

    def __init__(self, logger: TimestampedLogger, label: str) -> None:
        """Attach the stream to a logger and identify its stdout/stderr origin."""

        self._logger = logger
        self._label = label
        self._buffer = ""

    @property
    def encoding(self) -> str:
        """Advertise UTF-8 for libraries that inspect redirected stream encoding."""

        return "utf-8"

    def writable(self) -> bool:
        """Report that notebook and training code may write to this stream."""

        return True

    def write(self, value: str) -> int:
        """Emit complete lines immediately while preserving an unfinished suffix."""

        if not isinstance(value, str):
            value = str(value)
        self._buffer += value.replace("\r\n", "\n").replace("\r", "\n")
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            self._logger.info(f"[{self._label}] {line}")
        return len(value)

    def flush(self) -> None:
        """Emit the final partial line so no captured output is lost."""

        if self._buffer:
            self._logger.info(f"[{self._label}] {self._buffer}")
            self._buffer = ""

    def isatty(self) -> bool:
        """Disable terminal-control formatting in redirected progress libraries."""

        return False


@contextlib.contextmanager
def capture_output(logger: TimestampedLogger) -> Iterable[None]:
    """Redirect Python stdout/stderr into a timestamped logger for one phase."""

    stdout = _CapturedStream(logger, "stdout")
    stderr = _CapturedStream(logger, "stderr")
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            yield
        finally:
            stdout.flush()
            stderr.flush()


def find_submodule(start: Path | str | None = None) -> Path:
    """Locate and minimally validate the vendored BSF repository.

    ``start`` may be either the submodule itself or any directory inside the outer
    project.  Explicit discovery makes CLI use independent of the shell's CWD.
    """

    origin = Path(start).expanduser().resolve() if start is not None else PROJECT_ROOT
    for directory in (origin, *origin.parents):
        candidates = (
            directory,
            directory / "vendor" / "block-sparse-featurizer",
        )
        for candidate in candidates:
            if (candidate / "README.md").is_file() and (candidate / "bsf").is_dir():
                return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate vendor/block-sparse-featurizer; initialize the Git submodule first."
    )


def create_run_directory(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    label: str = "reproduction",
    now: datetime | None = None,
) -> Path:
    """Create a collision-safe timestamped directory beneath the ignored run root."""

    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-.") or "run"
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    timestamp = moment.strftime("%Y%m%dT%H%M%S.%fZ")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / f"{timestamp}-{safe_label}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}-{safe_label}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def extract_readme_quickstart(readme_path: Path | str) -> str:
    """Extract the first Python fence under the upstream ``Quickstart`` heading."""

    text = Path(readme_path).read_text(encoding="utf-8")
    heading = re.search(r"(?m)^##\s+Quickstart\s*$", text)
    if heading is None:
        raise ValueError("README.md does not contain a level-two Quickstart heading.")
    fence = re.search(
        r"(?ms)^```(?:python|py)\s*\n(?P<source>.*?)^```\s*$",
        text[heading.end() :],
    )
    if fence is None:
        raise ValueError("README Quickstart does not contain a Python code fence.")
    return fence.group("source")


def _load_hf_token(project_root: Path = PROJECT_ROOT) -> str | None:
    """Load the project ``.env`` without overriding an existing process setting."""

    dotenv_path = project_root / ".env"
    if dotenv_path.is_file():
        try:
            from dotenv import load_dotenv
        except ImportError:
            # Package-version preflight reports the missing dependency separately.
            pass
        else:
            load_dotenv(dotenv_path=dotenv_path, override=False)
    value = os.environ.get("HF_TOKEN", "").strip()
    return value or None


@contextlib.contextmanager
def _temporary_hf_token(token: str | None) -> Iterable[None]:
    """Expose an explicit token only while trusted upstream code is executing.

    Hugging Face libraries discover ``HF_TOKEN`` from the process environment.
    Restoring the exact prior state prevents one run from changing later sessions;
    the mapping behavior is documented at https://docs.python.org/3/library/os.html#os.environ.
    """

    if not token:
        yield
        return
    was_present = "HF_TOKEN" in os.environ
    previous = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = token
    try:
        yield
    finally:
        if was_present and previous is not None:
            os.environ["HF_TOKEN"] = previous
        else:
            os.environ.pop("HF_TOKEN", None)


def _submodule_commit(submodule_root: Path) -> str | None:
    """Read the pinned Git commit without changing submodule state."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(submodule_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _submodule_clean(submodule_root: Path) -> bool | None:
    """Return whether the vendored checkout has no tracked or untracked changes."""

    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(submodule_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return not completed.stdout.strip()


def _check_hf_model_access(model_id: str, token: str) -> None:
    """Perform an authenticated metadata HEAD request for a gated model file.

    Hugging Face documents ``get_hf_file_metadata`` as a metadata-only request
    accepting a token, avoiding a model download during preflight:
    https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download
    """

    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    url = hf_hub_url(repo_id=model_id, filename=DINO_ACCESS_FILE)
    get_hf_file_metadata(url, token=token)


def preflight_environment(
    submodule_root: Path | str | None = None,
    *,
    token: str | None = None,
    require_cuda: bool = True,
    check_hf: bool = True,
    required_python: tuple[int, int] = (3, 12),
    required_distributions: Sequence[str] = REQUIRED_DISTRIBUTIONS,
    expected_versions: Mapping[str, str] | None = EXPECTED_DISTRIBUTION_VERSIONS,
    expected_commit: str | None = EXPECTED_SUBMODULE_COMMIT,
    require_clean_submodule: bool = True,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    torch_module: Any | None = None,
    hf_access_checker: Callable[[str, str], None] | None = None,
    logger: TimestampedLogger | None = None,
) -> PreflightResult:
    """Validate assets, versions, CUDA, and authenticated DINOv3 access safely."""

    root = find_submodule(submodule_root)
    resolved_token = token if token is not None else _load_hf_token()
    secrets = (*environment_secrets(), *(value for value in (resolved_token,) if value))
    errors: list[str] = []
    warnings: list[str] = []

    assets = {
        str(relative): (root / relative).is_file() for relative in REQUIRED_ASSETS
    }
    for relative, present in assets.items():
        if not present:
            errors.append(f"Missing required upstream asset: {relative}")

    if sys.version_info < required_python:
        errors.append(
            f"Python {required_python[0]}.{required_python[1]} or newer is required; "
            f"found {sys.version_info.major}.{sys.version_info.minor}."
        )

    package_versions: dict[str, str | None] = {}
    for distribution in required_distributions:
        try:
            installed_version = version_getter(distribution)
            package_versions[distribution] = installed_version
            if expected_versions is not None:
                expected_version = expected_versions.get(distribution)
                if expected_version is None:
                    errors.append(
                        f"No locked version is configured for distribution: {distribution}"
                    )
                elif installed_version != expected_version:
                    errors.append(
                        f"{distribution} version {installed_version} does not match "
                        f"expected {expected_version}."
                    )
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
            errors.append(f"Required distribution is not installed: {distribution}")
        except Exception as exc:  # pragma: no cover - defensive metadata backend guard.
            package_versions[distribution] = None
            errors.append(
                sanitize_text(
                    f"Could not inspect distribution {distribution}: {type(exc).__name__}: {exc}",
                    secrets,
                )
            )

    cuda_available = False
    cuda_device: str | None = None
    try:
        if torch_module is None:
            import torch as imported_torch

            torch_module = imported_torch
        cuda_available = bool(torch_module.cuda.is_available())
        if cuda_available:
            cuda_device = str(torch_module.cuda.get_device_name(0))
    except Exception as exc:
        errors.append(
            sanitize_text(
                f"Could not inspect CUDA: {type(exc).__name__}: {exc}", secrets
            )
        )
    if require_cuda and not cuda_available:
        errors.append("CUDA is required to reproduce DINOv3 activation extraction.")
    elif not cuda_available:
        warnings.append("CUDA is unavailable; only offline/CPU checks can run.")

    gated_model_access: bool | None = None
    if check_hf:
        if not resolved_token:
            gated_model_access = False
            errors.append(
                "HF_TOKEN is missing; authenticated DINOv3 access cannot be checked."
            )
        else:
            checker = hf_access_checker or _check_hf_model_access
            try:
                checker(DINO_MODEL_ID, resolved_token)
            except Exception as exc:
                gated_model_access = False
                errors.append(
                    sanitize_text(
                        f"DINOv3 gated access failed: {type(exc).__name__}: {exc}",
                        secrets,
                    )
                )
            else:
                gated_model_access = True
    else:
        warnings.append("Hugging Face gated-model access check was skipped.")

    commit = _submodule_commit(root)
    if commit is None:
        message = "Could not determine the submodule Git commit."
        (errors if expected_commit is not None else warnings).append(message)
    elif expected_commit is not None and commit != expected_commit:
        errors.append(
            f"Found submodule commit {commit}; expected submodule commit {expected_commit}."
        )

    submodule_clean = _submodule_clean(root)
    if submodule_clean is None:
        message = "Could not determine whether the submodule checkout is clean."
        (errors if require_clean_submodule else warnings).append(message)
    elif require_clean_submodule and not submodule_clean:
        errors.append(
            "The submodule contains uncommitted changes; restore the pinned checkout "
            "before reproduction."
        )

    result = PreflightResult(
        ok=not errors,
        checked_at=utc_timestamp(),
        submodule_root=str(root),
        submodule_commit=commit,
        python_version=sys.version.split()[0],
        package_versions=package_versions,
        assets=assets,
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        hf_token_present=bool(resolved_token),
        gated_model_access=gated_model_access,
        submodule_clean=submodule_clean,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
    if logger is not None:
        logger.info(
            "Preflight completed: "
            f"ok={result.ok}, cuda={result.cuda_available}, "
            f"hf_token_present={result.hf_token_present}, "
            f"gated_model_access={result.gated_model_access}, "
            f"submodule_clean={result.submodule_clean}"
        )
        for warning in warnings:
            logger.warning(warning)
        for error in errors:
            logger.error(error)
    return result


@contextlib.contextmanager
def _upstream_execution_context(
    submodule_root: Path, logger: TimestampedLogger
) -> Iterable[None]:
    """Execute imports and relative files as if launched from the upstream root."""

    old_cwd = Path.cwd()
    inserted = str(submodule_root)
    previous_backend = os.environ.get("MPLBACKEND")
    sys.path.insert(0, inserted)
    os.chdir(submodule_root)
    # A non-interactive backend makes the README's ``plt.show()`` return while
    # retaining figures for PNG/PDF artifact export.
    os.environ["MPLBACKEND"] = "Agg"
    try:
        with capture_output(logger):
            yield
    finally:
        os.chdir(old_cwd)
        if previous_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous_backend
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)
        else:  # Preserve unrelated path mutations made by the executed example.
            with contextlib.suppress(ValueError):
                sys.path.remove(inserted)


def _sha256(path: Path) -> str:
    """Hash a source notebook/README to prove it was not modified by execution."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(
    path: Path, payload: Mapping[str, Any], secrets: Iterable[str] = ()
) -> None:
    """Write stable, recursively sanitized JSON for human and machine inspection."""

    def sanitized(value: Any) -> Any:
        if isinstance(value, str):
            return sanitize_text(value, secrets)
        if isinstance(value, Mapping):
            return {str(key): sanitized(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitized(item) for item in value]
        return value

    path.write_text(
        json.dumps(sanitized(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _sanitized_notebook(notebook: Any, secrets: Iterable[str]) -> Any:
    """Deep-copy and redact notebook strings before saving executed artifacts.

    HTTP exceptions can echo authorization headers into cell outputs. Redacting a
    copy protects IPYNB and derived HTML artifacts without changing execution state.
    """

    sanitized = copy.deepcopy(notebook)

    def visit(value: Any) -> Any:
        """Recursively redact NotebookNode, dictionary, and list values in place."""

        if isinstance(value, str):
            return sanitize_text(value, secrets)
        if isinstance(value, Mapping):
            for key in tuple(value):
                value[key] = visit(value[key])
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = visit(item)
            return value
        return value

    return visit(sanitized)


def _clear_notebook_outputs(notebook: Any) -> Any:
    """Remove embedded reference outputs from the in-memory execution copy.

    Starter notebooks ship with prior successful outputs. Clearing only the copy
    prevents a failed fresh run from reporting those historical metrics as current.
    """

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    return notebook


def _save_open_figures(namespace: Mapping[str, Any], output_dir: Path) -> list[Path]:
    """Export every Matplotlib figure left open by the README quickstart."""

    pyplot = namespace.get("plt")
    if pyplot is None:
        # The README calls ``bsf.viz.plot_concepts`` without importing pyplot
        # itself. Matplotlib's public pyplot registry still owns that unassigned
        # figure, so consult it directly before deciding there is nothing to save.
        import matplotlib.pyplot as pyplot
    artifacts: list[Path] = []
    for sequence, figure_number in enumerate(pyplot.get_fignums(), start=1):
        figure = pyplot.figure(figure_number)
        for suffix in ("png", "pdf"):
            path = output_dir / f"quickstart-figure-{sequence:03d}.{suffix}"
            figure.savefig(path, bbox_inches="tight")
            artifacts.append(path)
        pyplot.close(figure)
    return artifacts


def _release_readme_resources(
    namespace: dict[str, Any], logger: TimestampedLogger
) -> None:
    """Release README figures, Python references, and cached CUDA allocations.

    A full ``gc.collect`` clears unreachable cycles after the execution namespace
    is emptied (https://docs.python.org/3/library/gc.html#gc.collect). PyTorch's
    public ``empty_cache`` then returns unoccupied allocator blocks before a
    notebook kernel starts:
    https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.empty_cache.html.
    """

    try:
        import matplotlib.pyplot as pyplot

        pyplot.close("all")
    except Exception as error:  # pragma: no cover - defensive backend cleanup.
        logger.warning(
            f"Could not close README figures: {type(error).__name__}: {error}"
        )
    namespace.clear()
    try:
        gc.collect()
    except Exception as error:  # pragma: no cover - interpreter-level defense.
        logger.warning(
            f"Could not collect README objects: {type(error).__name__}: {error}"
        )
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception as error:  # pragma: no cover - defensive runtime cleanup.
        logger.warning(f"Could not empty CUDA cache: {type(error).__name__}: {error}")
    logger.info("Released README execution resources and unoccupied CUDA cache")


def _has_nonblank_raster_plot(artifacts: Sequence[Path]) -> bool:
    """Return true when at least one readable raster contains pixel variation."""

    from PIL import Image

    for path in artifacts:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            if path.stat().st_size <= 0:
                continue
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                if rgb.width < 2 or rgb.height < 2:
                    continue
                if any(low != high for low, high in rgb.getextrema()):
                    return True
        except (OSError, ValueError):
            continue
    return False


def _finite_number(value: Any) -> bool:
    """Recognize a real finite metric while rejecting booleans and placeholders."""

    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _shape_matches(value: Any, expected: tuple[int, ...]) -> bool:
    """Compare a JSON-style shape with one exact expected tuple."""

    if not isinstance(value, (list, tuple)):
        return False
    try:
        return tuple(int(size) for size in value) == expected
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_readme_acceptance(
    metrics: Mapping[str, Any], artifacts: Sequence[Path]
) -> tuple[str, ...]:
    """Validate exact quickstart shapes, quality, concepts, and visual evidence."""

    errors: list[str] = []
    for key, expected in EXPECTED_README_SHAPES.items():
        if not _shape_matches(metrics.get(key), expected):
            errors.append(
                f"README {key} must be {list(expected)}; found {metrics.get(key)!r}."
            )
    if metrics.get("patch_grid") != EXPECTED_PATCH_GRID:
        errors.append(
            f"README patch grid must be {EXPECTED_PATCH_GRID}; "
            f"found {metrics.get('patch_grid')!r}."
        )
    if metrics.get("finite_codes") is not True:
        errors.append("README codes must all be finite.")
    concepts = metrics.get("top_concepts")
    if not isinstance(concepts, (list, tuple)) or not concepts:
        errors.append("README must produce at least one ranked top concept.")
    r2 = metrics.get("r2")
    if not _finite_number(r2) or float(r2) < MINIMUM_RECONSTRUCTION_R2:
        errors.append(
            f"README R² must be finite and >= {MINIMUM_RECONSTRUCTION_R2:.2f}; "
            f"found {r2!r}."
        )
    if not _has_nonblank_raster_plot(artifacts):
        errors.append("README must produce at least one nonblank raster plot artifact.")
    return tuple(errors)


def _validate_notebook_acceptance(
    metrics: Mapping[str, Any], artifacts: Sequence[Path]
) -> tuple[str, ...]:
    """Validate one exact 300-epoch notebook and its saved plot evidence."""

    errors: list[str] = []
    if not _shape_matches(
        metrics.get("activation_shape"), EXPECTED_NOTEBOOK_ACTIVATION_SHAPE
    ):
        errors.append(
            "Notebook activation shape must be "
            f"{list(EXPECTED_NOTEBOOK_ACTIVATION_SHAPE)}; "
            f"found {metrics.get('activation_shape')!r}."
        )
    if metrics.get("patch_grid") != EXPECTED_PATCH_GRID:
        errors.append(
            f"Notebook patch grid must be {EXPECTED_PATCH_GRID}; "
            f"found {metrics.get('patch_grid')!r}."
        )
    if (
        not metrics.get("training")
        or metrics.get("finite_losses") is not True
        or not _finite_number(metrics.get("loss"))
    ):
        errors.append(
            "Notebook training metrics are missing or contain non-finite losses."
        )
    if (
        metrics.get("epoch") != EXPECTED_NOTEBOOK_EPOCHS
        or metrics.get("epochs") != EXPECTED_NOTEBOOK_EPOCHS
    ):
        errors.append(
            f"Notebook training must finish epoch {EXPECTED_NOTEBOOK_EPOCHS}."
        )
    r2 = metrics.get("r2")
    if not _finite_number(r2) or float(r2) < MINIMUM_RECONSTRUCTION_R2:
        errors.append(
            f"Notebook R² must be finite and >= {MINIMUM_RECONSTRUCTION_R2:.2f}; "
            f"found {r2!r}."
        )
    concepts = metrics.get("top_concepts")
    if not isinstance(concepts, (list, tuple)) or not concepts:
        errors.append("Notebook must produce at least one ranked top concept.")
    if metrics.get("error_count") != 0:
        errors.append("Notebook outputs must contain no execution errors.")
    if not _has_nonblank_raster_plot(artifacts):
        errors.append(
            "Notebook must produce at least one nonblank raster plot artifact."
        )
    return tuple(errors)


def _batched_reconstruction_r2(model: Any, x: Any, *, batch_size: int = 2048) -> float:
    """Measure reconstruction R² in bounded batches to avoid a second GPU OOM."""

    import numpy as np
    import torch

    values = np.asarray(x, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    total = 0.0
    for start in range(0, len(values), batch_size):
        centered = values[start : start + batch_size] - mean
        total += float(np.square(centered, dtype=np.float64).sum())
    if total <= 0:
        return 0.0
    device = next(model.parameters()).device
    residual = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start : start + batch_size], device=device)
            reconstructed = model(batch)[0]
            residual += float((batch - reconstructed).double().pow(2).sum().item())
    return 1.0 - residual / total


def _quickstart_metrics(namespace: Mapping[str, Any]) -> dict[str, Any]:
    """Collect acceptance metrics from variables created by the verbatim quickstart."""

    import numpy as np

    metrics: dict[str, Any] = {}
    for name in ("images", "acts", "x", "z", "atoms"):
        value = namespace.get(name)
        if value is not None and hasattr(value, "shape"):
            metrics[f"{name}_shape"] = [int(size) for size in value.shape]
    if "grid" in namespace:
        metrics["patch_grid"] = int(namespace["grid"])
    if "top" in namespace:
        metrics["top_concepts"] = [int(index) for index in namespace["top"]]

    z = namespace.get("z")
    if z is not None:
        heat = np.linalg.norm(np.asarray(z), axis=-1)
        active = heat > 1e-6
        metrics["finite_codes"] = bool(np.isfinite(z).all())
        metrics["l0"] = float(active.sum(axis=1).mean())
        metrics["dead_groups"] = int((~active.any(axis=0)).sum())
        metrics["firing_counts"] = [int(value) for value in active.sum(axis=0)]

    model = namespace.get("model")
    x = namespace.get("x")
    if model is not None and x is not None:
        metrics["r2"] = float(_batched_reconstruction_r2(model, x))
    return metrics


def run_readme_quickstart(
    submodule_root: Path | str | None = None,
    *,
    output_dir: Path | str | None = None,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    token: str | None = None,
    executor: Callable[[str, dict[str, Any]], None] | None = None,
) -> ReproductionResult:
    """Execute the exact Python block in upstream README.md and save artifacts.

    ``executor`` exists for fast offline tests; production callers omit it and the
    extracted source is compiled and executed without edits or parameter changes.
    """

    root = find_submodule(submodule_root)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else create_run_directory(output_root, label="readme")
    )
    destination.mkdir(parents=True, exist_ok=True)
    resolved_token = token if token is not None else _load_hf_token()
    secrets = tuple(
        dict.fromkeys(
            (*environment_secrets(), *(value for value in (resolved_token,) if value))
        )
    )
    log_path = destination / "readme-quickstart.log"
    started_at = utc_timestamp()
    artifacts: list[Path] = []
    metrics: dict[str, Any] = {}
    error: str | None = None
    status: Literal["passed", "failed"] = "passed"
    readme = root / "README.md"
    source_hash = _sha256(readme)
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(readme),
    }

    with TimestampedLogger(log_path, secrets=secrets) as logger:
        try:
            source = extract_readme_quickstart(readme)
            source_artifact = destination / "README-quickstart.py"
            source_artifact.write_text(source, encoding="utf-8")
            artifacts.append(source_artifact)
            logger.info(
                f"Executing README quickstart from {readme}; source_sha256={source_hash}"
            )
            logger.info(f"Quickstart input source:\n{source}")
            with (
                _temporary_hf_token(resolved_token),
                _upstream_execution_context(root, logger),
            ):
                if executor is None:
                    exec(compile(source, str(readme), "exec"), namespace)
                else:
                    executor(source, namespace)
            artifacts.extend(_save_open_figures(namespace, destination))
            metrics = _quickstart_metrics(namespace)
            metrics_path = destination / "readme-metrics.json"
            _write_json(metrics_path, metrics, secrets)
            artifacts.append(metrics_path)
            logger.info(f"README quickstart metrics:\n{json.dumps(metrics, indent=2)}")
            acceptance_errors = _validate_readme_acceptance(metrics, artifacts)
            if acceptance_errors:
                status = "failed"
                error = "Acceptance validation failed: " + "; ".join(acceptance_errors)
                for acceptance_error in acceptance_errors:
                    logger.error(f"Acceptance: {acceptance_error}")
        except Exception as exc:
            status = "failed"
            error = sanitize_text(f"{type(exc).__name__}: {exc}", secrets)
            logger.error(f"README quickstart failed: {error}")
            logger.error(sanitize_text(traceback.format_exc(), secrets))
        finally:
            _release_readme_resources(namespace, logger)

        if _sha256(readme) != source_hash:
            status = "failed"
            error = (
                "README.md changed during reproduction; restore the upstream source."
            )
            logger.error(error)
        logger.info(f"README quickstart finished with status={status}")

    finished_at = utc_timestamp()
    summary_path = destination / "readme-summary.json"
    artifacts.append(summary_path)
    result = ReproductionResult(
        name="readme-quickstart",
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        output_dir=str(destination),
        log_path=str(log_path),
        artifacts=tuple(str(path) for path in artifacts),
        metrics=metrics,
        error=error,
    )
    _write_json(summary_path, result.to_dict(), secrets)
    return result


def _output_text(output: Mapping[str, Any]) -> str | None:
    """Return complete human-readable text from one nbformat output object."""

    output_type = output.get("output_type")
    if output_type == "stream":
        value = output.get("text", "")
        return "".join(value) if isinstance(value, list) else str(value)
    if output_type == "error":
        traceback_lines = output.get("traceback", ())
        if traceback_lines:
            return "\n".join(str(line) for line in traceback_lines)
        return f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
    data = output.get("data", {})
    if isinstance(data, Mapping) and "text/plain" in data:
        value = data["text/plain"]
        return "".join(value) if isinstance(value, list) else str(value)
    return None


def _log_notebook_outputs(
    cell: Mapping[str, Any], cell_index: int, logger: TimestampedLogger
) -> None:
    """Stream every textual notebook output to the timestamped run log."""

    for output_index, output in enumerate(cell.get("outputs", ()), start=1):
        text = _output_text(output)
        if text is not None:
            logger.info(f"Notebook cell {cell_index} output {output_index}:\n{text}")
        else:
            mime_types = sorted(output.get("data", {}).keys())
            logger.info(
                f"Notebook cell {cell_index} output {output_index}: "
                f"binary/rich output {mime_types} retained in executed notebook"
            )


def _log_kernel_message(
    message: Mapping[str, Any], cell_index: int, logger: TimestampedLogger
) -> None:
    """Mirror a kernel output message immediately for real-time notebook logs."""

    message_type = message.get("msg_type")
    content = message.get("content", {})
    if message_type == "stream":
        logger.info(
            f"Notebook cell {cell_index} live stream:\n{content.get('text', '')}"
        )
        return
    if message_type == "error":
        trace = content.get("traceback", ())
        text = "\n".join(str(line) for line in trace) or (
            f"{content.get('ename', 'Error')}: {content.get('evalue', '')}"
        )
        logger.error(f"Notebook cell {cell_index} live error:\n{text}")
        return
    if message_type not in {"execute_result", "display_data", "update_display_data"}:
        return
    data = content.get("data", {})
    if "text/plain" in data:
        value = data["text/plain"]
        text = "".join(value) if isinstance(value, list) else str(value)
        logger.info(f"Notebook cell {cell_index} live result:\n{text}")
    else:
        logger.info(
            f"Notebook cell {cell_index} live rich output {sorted(data)} retained in artifact"
        )


def _decode_base64(value: Any) -> bytes:
    """Decode nbformat's string-or-lines representation for binary rich outputs."""

    encoded = "".join(value) if isinstance(value, list) else str(value)
    return base64.b64decode(encoded)


def _extract_notebook_figures(notebook: Any, output_dir: Path) -> list[Path]:
    """Write embedded PNG, JPEG, SVG, and PDF outputs as standalone artifacts."""

    mime_suffixes = {
        "image/png": ("png", True),
        "image/jpeg": ("jpg", True),
        "image/svg+xml": ("svg", False),
        "application/pdf": ("pdf", True),
    }
    artifacts: list[Path] = []
    sequence = 0
    for cell in notebook.cells:
        for output in cell.get("outputs", ()):
            data = output.get("data", {})
            for mime_type, (suffix, binary) in mime_suffixes.items():
                if mime_type not in data:
                    continue
                sequence += 1
                path = output_dir / f"figure-{sequence:03d}.{suffix}"
                if binary:
                    path.write_bytes(_decode_base64(data[mime_type]))
                else:
                    value = data[mime_type]
                    path.write_text(
                        "".join(value) if isinstance(value, list) else str(value),
                        encoding="utf-8",
                    )
                artifacts.append(path)
    return artifacts


def _notebook_metrics(notebook: Any) -> dict[str, Any]:
    """Parse upstream training/status prints into an acceptance-oriented summary."""

    texts: list[str] = []
    output_count = 0
    error_count = 0
    for cell in notebook.cells:
        for output in cell.get("outputs", ()):
            output_count += 1
            error_count += int(output.get("output_type") == "error")
            text = _output_text(output)
            if text is not None:
                texts.append(text)
    complete_text = "\n".join(texts)

    training = []
    for match in _TRAINING_LINE.finditer(complete_text):
        training.append(
            {
                "epoch": int(match.group("epoch")),
                "epochs": int(match.group("epochs")),
                "loss": float(match.group("loss")),
                "r2": float(match.group("r2")),
                "l0": float(match.group("l0")),
                "dead_groups": int(match.group("dead")),
                "groups": int(match.group("groups")),
            }
        )

    metrics: dict[str, Any] = {
        "cell_count": len(notebook.cells),
        "executed_code_cells": sum(
            int(
                cell.get("cell_type") == "code"
                and cell.get("execution_count") is not None
            )
            for cell in notebook.cells
        ),
        "output_count": output_count,
        "error_count": error_count,
        "training": training,
        "finite_losses": bool(training)
        and all(math.isfinite(item["loss"]) for item in training),
    }
    if training:
        metrics.update(training[-1])

    top_match = re.search(r"top concepts:\s*(\[[^\n]*\])", complete_text)
    if top_match:
        # NumPy 2.x represents scalar list elements as ``np.int64(4)``. Normalize
        # only that integer constructor form before using the safe literal parser.
        top_literal = re.sub(
            r"np\.int(?:8|16|32|64)?\(\s*([-+]?\d+)\s*\)",
            r"\1",
            top_match.group(1),
        )
        try:
            top = ast.literal_eval(top_literal)
        except (SyntaxError, ValueError):
            pass
        else:
            if isinstance(top, list):
                metrics["top_concepts"] = [int(value) for value in top]

    activation_match = re.search(
        r"activations:\s*\((?P<shape>[^)]*)\)\s*patch grid:\s*(?P<grid>\d+)",
        complete_text,
    )
    if activation_match:
        metrics["activation_shape"] = [
            int(value.strip())
            for value in activation_match.group("shape").split(",")
            if value.strip()
        ]
        metrics["patch_grid"] = int(activation_match.group("grid"))
    return metrics


def execute_notebook(
    notebook_path: Path | str,
    *,
    submodule_root: Path | str | None = None,
    output_dir: Path | str | None = None,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    token: str | None = None,
    timeout: int = 7_200,
    kernel_name: str | None = None,
    client_factory: Callable[..., Any] | None = None,
    html_exporter_factory: Callable[[], Any] | None = None,
) -> ReproductionResult:
    """Execute one unchanged notebook with its kernel CWD set to the submodule root.

    The callback hooks are the nbclient-supported mechanism for streaming cell
    inputs and outputs: https://nbclient.readthedocs.io/en/latest/client.html#hooks-before-and-after-notebook-or-cell-execution
    """

    root = find_submodule(submodule_root)
    source_path = Path(notebook_path)
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve()
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else create_run_directory(output_root, label=source_path.stem)
    )
    destination.mkdir(parents=True, exist_ok=True)
    resolved_token = token if token is not None else _load_hf_token()
    secrets = tuple(
        dict.fromkeys(
            (*environment_secrets(), *(value for value in (resolved_token,) if value))
        )
    )
    log_path = destination / f"{source_path.stem}.log"
    executed_path = destination / f"{source_path.stem}.executed.ipynb"
    html_path = destination / f"{source_path.stem}.html"
    metrics_path = destination / f"{source_path.stem}.metrics.json"
    source_hash = _sha256(source_path)
    started_at = utc_timestamp()
    artifacts: list[Path] = []
    metrics: dict[str, Any] = {}
    error: str | None = None
    status: Literal["passed", "failed"] = "passed"

    # Lazy imports keep inexpensive preflight and CLI help usable before the
    # notebook extras are installed into the local virtual environment.
    import nbformat

    notebook = _clear_notebook_outputs(nbformat.read(source_path, as_version=4))
    executed = notebook
    stream_kernel_messages = client_factory is None
    with TimestampedLogger(log_path, secrets=secrets) as logger:
        logger.info(
            f"Executing unchanged notebook {source_path}; source_sha256={source_hash}; "
            f"kernel_cwd={root}"
        )

        def on_cell_execute(
            *, cell: Mapping[str, Any], cell_index: int, **_: Any
        ) -> None:
            """Log each complete code input immediately before kernel execution."""

            logger.info(f"Notebook cell {cell_index} input:\n{cell.get('source', '')}")

        def on_cell_complete(
            *, cell: Mapping[str, Any], cell_index: int, **_: Any
        ) -> None:
            """Log every completed textual output while rich data stays in nbformat."""

            if not stream_kernel_messages:
                _log_notebook_outputs(cell, cell_index, logger)

        def on_cell_error(
            *, cell: Mapping[str, Any], cell_index: int, **_: Any
        ) -> None:
            """Ensure failing-cell output reaches logs before nbclient propagates it."""

            if not stream_kernel_messages:
                _log_notebook_outputs(cell, cell_index, logger)

        try:
            if client_factory is None:
                from nbclient import NotebookClient

                class StreamingNotebookClient(NotebookClient):
                    """NotebookClient variant that mirrors IOPub output as it arrives."""

                    def process_message(
                        self,
                        msg: dict[str, Any],
                        cell: Any,
                        cell_index: int,
                    ) -> Any:
                        """Log a kernel message before normal nbclient persistence."""

                        _log_kernel_message(msg, cell_index, logger)
                        return super().process_message(msg, cell, cell_index)

                client_factory = StreamingNotebookClient
            selected_kernel = kernel_name or notebook.metadata.get(
                "kernelspec", {}
            ).get("name", "python3")
            # nbclient explicitly documents resources.metadata.path as the kernel
            # working directory; this lets each upstream notebook find ``bsf/``.
            with _temporary_hf_token(resolved_token):
                client = client_factory(
                    notebook,
                    timeout=timeout,
                    kernel_name=selected_kernel,
                    resources={"metadata": {"path": str(root)}},
                    allow_errors=False,
                    record_timing=True,
                    on_cell_execute=on_cell_execute,
                    on_cell_complete=on_cell_complete,
                    on_cell_error=on_cell_error,
                )
                with capture_output(logger):
                    returned = client.execute()
            if returned is not None:
                executed = returned
        except Exception as exc:
            status = "failed"
            error = sanitize_text(f"{type(exc).__name__}: {exc}", secrets)
            logger.error(f"Notebook execution failed: {error}")
            logger.error(sanitize_text(traceback.format_exc(), secrets))

        # nbclient recommends saving the mutated in-memory notebook even after a
        # cell exception so the successful prefix and failing output are inspectable.
        # Persist a redacted copy because an HTTP error may echo its credential.
        persisted = _sanitized_notebook(executed, secrets)
        nbformat.write(persisted, executed_path)
        artifacts.append(executed_path)
        try:
            if html_exporter_factory is None:
                from nbconvert.exporters import HTMLExporter

                html_exporter_factory = HTMLExporter
            exporter = html_exporter_factory()
            html, _resources = exporter.from_notebook_node(persisted)
            html_path.write_text(html, encoding="utf-8")
            artifacts.append(html_path)
        except Exception as exc:
            logger.warning(
                sanitize_text(
                    f"HTML export failed: {type(exc).__name__}: {exc}", secrets
                )
            )

        try:
            artifacts.extend(_extract_notebook_figures(persisted, destination))
        except Exception as exc:
            logger.warning(
                sanitize_text(
                    f"Figure extraction failed: {type(exc).__name__}: {exc}", secrets
                )
            )
        metrics = _notebook_metrics(persisted)
        _write_json(metrics_path, metrics, secrets)
        artifacts.append(metrics_path)
        logger.info(f"Notebook metrics:\n{json.dumps(metrics, indent=2)}")
        if status == "passed":
            acceptance_errors = _validate_notebook_acceptance(metrics, artifacts)
            if acceptance_errors:
                status = "failed"
                error = "Acceptance validation failed: " + "; ".join(acceptance_errors)
                for acceptance_error in acceptance_errors:
                    logger.error(f"Acceptance: {acceptance_error}")

        if _sha256(source_path) != source_hash:
            status = "failed"
            error = (
                "Source notebook changed during execution; restore the upstream file."
            )
            logger.error(error)
        logger.info(f"Notebook finished with status={status}")

    finished_at = utc_timestamp()
    summary_path = destination / f"{source_path.stem}.summary.json"
    artifacts.append(summary_path)
    result = ReproductionResult(
        name=source_path.stem,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        output_dir=str(destination),
        log_path=str(log_path),
        artifacts=tuple(str(path) for path in artifacts),
        metrics=metrics,
        error=error,
    )
    _write_json(summary_path, result.to_dict(), secrets)
    return result


def run_notebooks(
    submodule_root: Path | str | None = None,
    *,
    output_dir: Path | str | None = None,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    token: str | None = None,
    timeout: int = 7_200,
    notebook_paths: Sequence[Path | str] = NOTEBOOK_RELATIVE_PATHS,
    client_factory: Callable[..., Any] | None = None,
    html_exporter_factory: Callable[[], Any] | None = None,
) -> tuple[ReproductionResult, ...]:
    """Execute all three upstream starter notebooks sequentially and unchanged."""

    root = find_submodule(submodule_root)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else create_run_directory(output_root, label="notebooks")
    )
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for notebook_path in notebook_paths:
        source = Path(notebook_path)
        child = destination / source.stem
        results.append(
            execute_notebook(
                source,
                submodule_root=root,
                output_dir=child,
                token=token,
                timeout=timeout,
                client_factory=client_factory,
                html_exporter_factory=html_exporter_factory,
            )
        )
    return tuple(results)


def run_reproduction(
    target: ReproductionTarget = "all",
    *,
    submodule_root: Path | str | None = None,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    token: str | None = None,
    timeout: int = 7_200,
    require_cuda: bool = True,
    check_hf: bool = True,
    enforce_preflight: bool = True,
) -> ReproductionSuiteResult:
    """Run preflight followed by the requested exact upstream workflows.

    This function is the CLI-friendly orchestration boundary.  Failed preflight
    returns a structured failed suite by default rather than starting hours of
    work in an invalid environment.
    """

    if target not in {"readme", "notebooks", "all"}:
        raise ValueError("target must be one of: readme, notebooks, all")
    root = find_submodule(submodule_root)
    resolved_token = token if token is not None else _load_hf_token()
    secrets = tuple(
        dict.fromkeys(
            (*environment_secrets(), *(value for value in (resolved_token,) if value))
        )
    )
    destination = create_run_directory(output_root, label=target)
    started_at = utc_timestamp()
    results: list[ReproductionResult] = []

    with TimestampedLogger(destination / "reproduction.log", secrets=secrets) as logger:
        logger.info(f"Starting reproduction target={target}; submodule={root}")
        preflight = preflight_environment(
            root,
            token=resolved_token,
            require_cuda=require_cuda,
            check_hf=check_hf,
            logger=logger,
        )
        _write_json(destination / "preflight.json", preflight.to_dict(), secrets)
        if not preflight.ok and enforce_preflight:
            logger.error("Preflight failed; requested workflows were not started.")
        else:
            if target in {"readme", "all"}:
                results.append(
                    run_readme_quickstart(
                        root,
                        output_dir=destination / "readme",
                        token=resolved_token,
                    )
                )
            if target in {"notebooks", "all"}:
                results.extend(
                    run_notebooks(
                        root,
                        output_dir=destination / "notebooks",
                        token=resolved_token,
                        timeout=timeout,
                    )
                )

    expected_results = 1 if target == "readme" else 3 if target == "notebooks" else 4
    suite_ok = (
        preflight.ok
        and len(results) == expected_results
        and all(result.ok for result in results)
    )
    suite = ReproductionSuiteResult(
        target=target,
        status="passed" if suite_ok else "failed",
        started_at=started_at,
        finished_at=utc_timestamp(),
        output_dir=str(destination),
        preflight=preflight,
        results=tuple(results),
    )
    _write_json(destination / "summary.json", suite.to_dict(), secrets)
    return suite


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SUBMODULE_ROOT",
    "DINO_MODEL_ID",
    "EXPECTED_DISTRIBUTION_VERSIONS",
    "EXPECTED_SUBMODULE_COMMIT",
    "NOTEBOOK_RELATIVE_PATHS",
    "PreflightResult",
    "ReproductionResult",
    "ReproductionSuiteResult",
    "TimestampedLogger",
    "create_run_directory",
    "execute_notebook",
    "extract_readme_quickstart",
    "find_submodule",
    "preflight_environment",
    "run_notebooks",
    "run_readme_quickstart",
    "run_reproduction",
    "sanitize_text",
]
