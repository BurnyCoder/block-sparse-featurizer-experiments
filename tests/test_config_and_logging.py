"""Tests for safe environment configuration and complete redacted logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from bsf_experiments.config import load_app_config
from bsf_experiments.logging_utils import (
    create_run_logger,
    log_event,
    sanitize_for_logging,
)


def test_load_app_config_reads_env_without_storing_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Hugging Face token reaches the process environment but not AppConfig."""

    secret = "hf_test_value_that_must_never_be_logged"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                f"HF_TOKEN={secret}",
                "BSF_PORT=8123",
                "BSF_OUTPUT_DIR=outputs/artifacts",
                "BSF_MAX_UPLOAD_MB=32",
                "BSF_SESSION_TTL_SECONDS=90",
                "BSF_DEVICE=cpu",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)

    config = load_app_config(env_file=env_file, project_root=tmp_path)

    assert config.hf_token_available is True
    assert config.port == 8123
    assert config.output_dir == (tmp_path / "outputs" / "artifacts").resolve()
    assert secret not in repr(config)
    assert not hasattr(config, "hf_token")


def test_load_app_config_does_not_override_existing_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing process configuration has precedence over local dotenv values."""

    env_file = tmp_path / ".env"
    env_file.write_text("BSF_PORT=8123\nHF_TOKEN=hf_file_value\n", encoding="utf-8")
    monkeypatch.setenv("BSF_PORT", "9000")
    monkeypatch.setenv("HF_TOKEN", "hf_process_value")

    config = load_app_config(env_file=env_file, project_root=tmp_path)

    assert config.port == 9000
    assert config.hf_token_available is True


def test_load_app_config_accepts_only_the_dedicated_outputs_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output root itself and its descendants remain available for artifacts."""

    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")

    for setting, expected in (
        ("outputs", tmp_path / "outputs"),
        ("outputs/custom", tmp_path / "outputs" / "custom"),
    ):
        monkeypatch.setenv("BSF_OUTPUT_DIR", setting)
        config = load_app_config(env_file=env_file, project_root=tmp_path)
        assert config.output_dir == expected.resolve()


def test_load_app_config_defaults_to_outputs_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting the setting retains the documented safe ``outputs/runs`` default."""

    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.delenv("BSF_OUTPUT_DIR", raising=False)

    config = load_app_config(env_file=env_file, project_root=tmp_path)

    assert config.output_dir == (tmp_path / "outputs" / "runs").resolve()


def test_load_app_config_rejects_broad_or_external_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allowed downloads cannot expose broad, source, ancestor, or external paths."""

    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    unsafe_paths = (
        Path("/"),
        Path("/tmp"),
        Path.home(),
        tmp_path,
        tmp_path / "src",
        tmp_path / "vendor",
        tmp_path.parent,
        tmp_path.parent / "external-artifacts",
    )

    for unsafe_path in unsafe_paths:
        monkeypatch.setenv("BSF_OUTPUT_DIR", str(unsafe_path))
        with pytest.raises(ValueError, match="dedicated .*outputs"):
            load_app_config(env_file=env_file, project_root=tmp_path)


def test_load_app_config_rejects_output_traversal_and_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved containment prevents ``..`` and symlinks escaping outputs."""

    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.mkdir()
    (outputs / "escape").symlink_to(external, target_is_directory=True)

    for setting in ("outputs/../src", "outputs/escape/results"):
        monkeypatch.setenv("BSF_OUTPUT_DIR", setting)
        with pytest.raises(ValueError, match="dedicated .*outputs"):
            load_app_config(env_file=env_file, project_root=tmp_path)


def test_load_app_config_rejects_a_symlinked_outputs_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dedicated outputs directory itself cannot redirect to an external tree."""

    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    external = tmp_path.parent / f"{tmp_path.name}-external-root"
    external.mkdir()
    (tmp_path / "outputs").symlink_to(external, target_is_directory=True)
    monkeypatch.setenv("BSF_OUTPUT_DIR", "outputs/custom")

    with pytest.raises(ValueError, match="dedicated .*outputs"):
        load_app_config(env_file=env_file, project_root=tmp_path)


@pytest.mark.parametrize(
    ("setting", "value"),
    (
        ("BSF_HOST", "0.0.0.0"),
        ("BSF_PORT", "70000"),
        ("BSF_DEVICE", "tpu"),
        ("BSF_DEVICE", "CUDA"),
        ("BSF_DEVICE", "cuda:"),
        ("BSF_DEVICE", "cuda:-1"),
        ("BSF_DEVICE", "cuda:+1"),
        ("BSF_DEVICE", "cuda:١"),
        ("BSF_DEVICE", "cuda:1:2"),
        ("BSF_DEVICE", "cuda0"),
        ("BSF_DEVICE", " cuda"),
    ),
)
def test_load_app_config_rejects_unsafe_or_invalid_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: str,
) -> None:
    """Configuration validation keeps the workbench local and actionable."""

    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv(setting, value)

    with pytest.raises(ValueError):
        load_app_config(env_file=env_file, project_root=tmp_path)


def test_sanitize_for_logging_redacts_keys_nested_values_and_token_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured and embedded secrets are replaced without truncating safe text."""

    secret = "arbitrary-value-known-through-a-secret-environment-key"
    monkeypatch.setenv("SERVICE_PASSWORD", secret)
    payload = {
        "authorization": "Bearer some-credential",
        "nested": {
            "api_token": "hf_explicit_pattern",
            "safe": f"before {secret} after",
        },
        "long_output": "x" * 20_000,
    }

    sanitized = sanitize_for_logging(payload)
    encoded = json.dumps(sanitized)

    assert "some-credential" not in encoded
    assert "hf_explicit_pattern" not in encoded
    assert secret not in encoded
    assert sanitized["authorization"] == "[REDACTED]"
    assert sanitized["nested"]["api_token"] == "[REDACTED]"
    assert sanitized["long_output"] == "x" * 20_000


def test_sanitizer_recognizes_compound_secret_keys_and_assignments() -> None:
    """Camel-case credentials redact while innocent substring lookalikes survive."""

    leaked_values = (
        "database-password-fabricated-leak",
        "access-token-fabricated-leak",
        "client-secret-fabricated-leak",
        "authorization-header-fabricated-leak",
    )
    payload = {
        "databasePassword": leaked_values[0],
        "nested": {
            "accessToken": leaked_values[1],
            "clientSecret": leaked_values[2],
            "authorizationHeader": leaked_values[3],
        },
        "assignments": " ".join(
            (
                f"databasePassword={leaked_values[0]}",
                f"accessToken:{leaked_values[1]}",
                f"clientSecret='{leaked_values[2]}'",
                f'authorizationHeader="{leaked_values[3]}"',
            )
        ),
        "safe": {
            "passwordlessMode": "enabled",
            "tokenizerName": "sentencepiece",
            "secretaryName": "Ada",
        },
        "unseparated": {
            "databasepassword": leaked_values[0],
            "accesstoken": leaked_values[1],
            "clientsecret": leaked_values[2],
            "authorizationheader": leaked_values[3],
        },
        "safe_assignments": (
            "passwordlessMode=true tokenizerName=sentencepiece secretaryName=Ada"
        ),
    }

    sanitized = sanitize_for_logging(payload)
    encoded = json.dumps(sanitized)

    for leaked_value in leaked_values:
        assert leaked_value not in encoded
    assert sanitized["databasePassword"] == "[REDACTED]"
    assert sanitized["nested"] == {
        "accessToken": "[REDACTED]",
        "clientSecret": "[REDACTED]",
        "authorizationHeader": "[REDACTED]",
    }
    assert sanitized["safe"] == {
        "passwordlessMode": "enabled",
        "tokenizerName": "sentencepiece",
        "secretaryName": "Ada",
    }
    assert set(sanitized["unseparated"].values()) == {"[REDACTED]"}
    assert sanitized["safe_assignments"] == (
        "passwordlessMode=true tokenizerName=sentencepiece secretaryName=Ada"
    )


def test_run_logger_writes_full_sanitized_events_to_file_and_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both logging destinations receive complete content with secrets removed."""

    secret = "hf_logger_secret_value"
    monkeypatch.setenv("HF_TOKEN", secret)
    logger, log_path = create_run_logger("unit", tmp_path, level=logging.INFO)
    long_output = "result:" + "z" * 20_000

    log_event(logger, "model_output", {"text": long_output, "token": secret})
    logger.info("embedded=%s", secret)
    for handler in logger.handlers:
        handler.flush()

    file_text = log_path.read_text(encoding="utf-8")
    terminal_text = capsys.readouterr().out
    for emitted in (file_text, terminal_text):
        assert secret not in emitted
        assert "[REDACTED]" in emitted
        assert long_output in emitted
