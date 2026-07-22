"""Load local workbench settings while keeping credentials out of application state.

``python-dotenv`` deliberately defaults to ``override=False`` so shell-provided
configuration remains authoritative; see
https://saurabh-kumar.com/python-dotenv/reference/#dotenv.main.load_dotenv.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv


_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_DEVICE_PATTERN = re.compile(r"^(?:auto|cpu|cuda(?::[0-9]+)?)$")


def validate_device_setting(device: str) -> str:
    """Return one exact supported device string or raise an actionable error."""

    if not isinstance(device, str) or not _DEVICE_PATTERN.fullmatch(device):
        raise ValueError("Device must be auto, cpu, cuda, or cuda:<index>")
    return device


def resolve_torch_device(device: str, *, torch_module: Any | None = None) -> str:
    """Resolve ``auto`` and validate CUDA availability plus any explicit ordinal.

    ``torch.cuda.device_count`` is PyTorch's public count of visible GPUs, so an
    explicit index is checked here before model transfer or allocation begins:
    https://docs.pytorch.org/docs/stable/generated/torch.cuda.device_count.html.
    """

    requested = validate_device_setting(device)
    if torch_module is None:
        # The lazy import keeps dotenv/config-only commands inexpensive.
        import torch as torch_module

    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cpu":
        return requested
    if not torch_module.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if ":" in requested:
        index = int(requested.partition(":")[2])
        visible_devices = int(torch_module.cuda.device_count())
        if index >= visible_devices:
            raise ValueError(
                f"CUDA device index {index} is out of range; "
                f"PyTorch reports {visible_devices} visible CUDA device(s)"
            )
    return requested


def default_project_root() -> Path:
    """Return the checkout root from this package's ``src`` layout."""

    return Path(__file__).resolve().parents[2]


def _resolved_output_paths(
    project_root: str | Path, output_dir: str | Path
) -> tuple[Path, Path]:
    """Resolve and contain artifacts beneath the physical project outputs tree.

    Python documents that ``Path.resolve`` follows symlinks and eliminates
    ``..`` components before the subsequent containment comparison:
    https://docs.python.org/3.12/library/pathlib.html#pathlib.Path.resolve.
    """

    resolved_project_root = Path(project_root).expanduser().resolve()
    dedicated_output_root = resolved_project_root / "outputs"
    physical_output_root = dedicated_output_root.resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    if (
        physical_output_root != dedicated_output_root
        or not resolved_output_dir.is_relative_to(dedicated_output_root)
    ):
        raise ValueError(
            "BSF_OUTPUT_DIR must resolve to the dedicated project outputs tree "
            f"({dedicated_output_root}) or one of its descendants; "
            f"received {resolved_output_dir}"
        )
    return resolved_project_root, resolved_output_dir


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated non-secret process configuration for the local application."""

    project_root: Path
    env_file: Path
    host: str
    port: int
    output_dir: Path
    log_level: str
    max_upload_mb: int
    session_ttl_seconds: int
    device: str
    hf_token_available: bool

    def __post_init__(self) -> None:
        """Reject unsafe networking and malformed numeric/device settings early."""

        resolved_root, resolved_output = _resolved_output_paths(
            self.project_root, self.output_dir
        )
        # Frozen dataclasses permit validation-time normalization via object.__setattr__.
        object.__setattr__(self, "project_root", resolved_root)
        object.__setattr__(self, "output_dir", resolved_output)
        if self.host not in _LOCAL_HOSTS:
            raise ValueError(
                "BSF_HOST must be a local address: 127.0.0.1, localhost, or ::1"
            )
        if not 1 <= self.port <= 65_535:
            raise ValueError("BSF_PORT must be between 1 and 65535")
        if self.max_upload_mb <= 0:
            raise ValueError("BSF_MAX_UPLOAD_MB must be greater than zero")
        if self.session_ttl_seconds <= 0:
            raise ValueError("BSF_SESSION_TTL_SECONDS must be greater than zero")
        try:
            validate_device_setting(self.device)
        except ValueError as error:
            raise ValueError(
                "BSF_DEVICE must be auto, cpu, cuda, or cuda:<index>"
            ) from error
        if self.log_level not in logging.getLevelNamesMapping():
            raise ValueError(f"Unsupported BSF_LOG_LEVEL: {self.log_level}")


def _environment_int(name: str, default: int) -> int:
    """Parse one integer setting and attach its name to conversion failures."""

    raw_value = os.getenv(name, str(default))
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def load_app_config(
    env_file: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> AppConfig:
    """Load ``.env`` and return only validated, non-secret settings.

    The token is intentionally represented by a boolean. Hugging Face clients
    read ``HF_TOKEN`` from the environment directly, so copying its value into a
    printable dataclass would only increase its exposure surface.
    """

    root = Path(project_root).resolve() if project_root else default_project_root()
    dotenv_path = Path(env_file).resolve() if env_file else root / ".env"
    if env_file is not None and not dotenv_path.is_file():
        raise FileNotFoundError(f"Environment file does not exist: {dotenv_path}")

    # Existing shell variables win, matching python-dotenv's documented default.
    load_dotenv(dotenv_path=dotenv_path, override=False, verbose=False)
    output_setting = Path(os.getenv("BSF_OUTPUT_DIR", "outputs/runs"))
    output_dir = (
        output_setting if output_setting.is_absolute() else root / output_setting
    )

    return AppConfig(
        project_root=root,
        env_file=dotenv_path,
        host=os.getenv("BSF_HOST", "127.0.0.1"),
        port=_environment_int("BSF_PORT", 7860),
        output_dir=output_dir.resolve(),
        log_level=os.getenv("BSF_LOG_LEVEL", "INFO").upper(),
        max_upload_mb=_environment_int("BSF_MAX_UPLOAD_MB", 512),
        session_ttl_seconds=_environment_int("BSF_SESSION_TTL_SECONDS", 3600),
        device=os.getenv("BSF_DEVICE", "auto"),
        hf_token_available=bool(os.getenv("HF_TOKEN", "").strip()),
    )


__all__ = [
    "AppConfig",
    "default_project_root",
    "load_app_config",
    "resolve_torch_device",
    "validate_device_setting",
]
