"""Fast, offline tests for secure upstream reproduction orchestration.

The real examples require a gated model and CUDA.  These tests instead exercise
the same filesystem, logging, callback, and summary boundaries with mocked
network/GPU/kernel collaborators.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nbformat
import pytest
from PIL import Image

from bsf_experiments import reproduction


_QUICKSTART_SOURCE = "print('exact upstream source')\nanswer = 42\n"


def _valid_readme_metrics() -> dict[str, Any]:
    """Return the exact quickstart evidence required by production validation."""

    return {
        "images_shape": [300, 224, 224, 3],
        "acts_shape": [300, 196, 768],
        "x_shape": [58_800, 768],
        "z_shape": [58_800, 256, 3],
        "atoms_shape": [256, 3, 768],
        "patch_grid": 14,
        "top_concepts": [3, 7],
        "finite_codes": True,
        "r2": 0.91,
    }


def _valid_notebook_output() -> str:
    """Return the status lines emitted by one successful exact starter notebook."""

    return (
        "activations: (58800, 768) patch grid: 14\n"
        "epoch 300/300 loss=0.1000 R2=0.9000 L0=8.0 dead=2/256\n"
        "top concepts: [np.int64(4), np.int64(8)]\n"
    )


def _nonblank_png_data() -> str:
    """Encode a tiny two-color PNG as a realistic nonblank notebook output."""

    image = Image.new("RGB", (2, 2), color="white")
    image.putpixel((0, 0), (0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _make_upstream(root: Path, *, notebook_source: str = "print('hello')") -> Path:
    """Create the minimal upstream layout expected by discovery and preflight."""

    upstream = root / "vendor" / "block-sparse-featurizer"
    (upstream / "bsf").mkdir(parents=True)
    (upstream / "starters").mkdir()
    (upstream / "README.md").write_text(
        f"# BSF\n\n## Quickstart\n\n```python\n{_QUICKSTART_SOURCE}```\n",
        encoding="utf-8",
    )
    (upstream / "rabbit.npz").write_bytes(b"fixture")
    (upstream / "bsf/__init__.py").write_text("", encoding="utf-8")
    (upstream / "bsf/pos_mean.npy").write_bytes(b"fixture")
    for relative in reproduction.NOTEBOOK_RELATIVE_PATHS:
        notebook = nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell(notebook_source)],
            metadata={
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            },
        )
        nbformat.write(notebook, upstream / relative)
    return upstream


class _FakeCuda:
    """Model the two torch.cuda calls used by preflight."""

    @staticmethod
    def is_available() -> bool:
        """Report the mocked GPU as available."""

        return True

    @staticmethod
    def get_device_name(index: int) -> str:
        """Return a deterministic device label for JSON assertions."""

        assert index == 0
        return "Mock RTX"


class _FakeHtmlExporter:
    """Avoid nbconvert template loading while preserving its public call shape."""

    def from_notebook_node(self, notebook: Any) -> tuple[str, dict[str, Any]]:
        """Return a tiny deterministic HTML artifact."""

        assert notebook.cells
        return "<!doctype html><title>executed</title>", {}


def _fake_client_factory(
    record: dict[str, Any], output: str, *, include_image: bool = True
):
    """Build an nbclient-compatible fake that fires documented callback hooks."""

    class FakeClient:
        """Execute one in-memory cell without starting a Jupyter kernel."""

        def __init__(self, notebook: Any, **kwargs: Any) -> None:
            """Retain the notebook and runner options for later assertions."""

            self.notebook = notebook
            self.kwargs = kwargs
            record.update(kwargs)

        def execute(self) -> Any:
            """Populate one stream output and invoke before/after callbacks."""

            record["hf_token_during_execute"] = os.environ.get("HF_TOKEN")
            cell = self.notebook.cells[0]
            self.kwargs["on_cell_execute"](cell=cell, cell_index=0)
            cell.execution_count = 1
            cell.outputs = [
                nbformat.v4.new_output("stream", name="stdout", text=output)
            ]
            if include_image:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "display_data", data={"image/png": _nonblank_png_data()}
                    )
                )
            self.kwargs["on_cell_complete"](cell=cell, cell_index=0)
            return self.notebook

    return FakeClient


def test_sanitize_text_redacts_values_assignments_and_bearer_headers() -> None:
    """Known and structurally recognizable credentials never survive logging."""

    secret = "plain-secret-value"
    synthetic_hf_credential = "hf_" + "abcdefghijklmnopqrstuvwxyz"
    text = (
        f"raw={secret} HF_TOKEN={synthetic_hf_credential} "
        "Authorization: Bearer abc.def-123 password='hunter2'"
    )

    sanitized = reproduction.sanitize_text(text, (secret,))

    assert secret not in sanitized
    assert synthetic_hf_credential not in sanitized
    assert "abc.def-123" not in sanitized
    assert "hunter2" not in sanitized
    assert sanitized.count("[REDACTED]") >= 4


def test_timestamped_logger_preserves_full_multiline_output(tmp_path: Path) -> None:
    """Every untruncated line reaches both the file and selected terminal stream."""

    terminal = io.StringIO()
    log_path = tmp_path / "run.log"
    long_line = "x" * 20_000

    with reproduction.TimestampedLogger(log_path, console=terminal) as logger:
        logger.info(f"first\n{long_line}\nlast")

    file_text = log_path.read_text(encoding="utf-8")
    assert long_line in file_text
    assert terminal.getvalue() == file_text
    assert len(file_text.splitlines()) == 3


def test_live_kernel_stream_is_logged_and_redacted(tmp_path: Path) -> None:
    """IOPub stream chunks reach the timestamped terminal without credential leaks."""

    token = "hf_mocked_token_that_must_never_be_serialized"
    terminal = io.StringIO()
    log_path = tmp_path / "kernel.log"
    with reproduction.TimestampedLogger(
        log_path, secrets=(token,), console=terminal
    ) as logger:
        reproduction._log_kernel_message(
            {
                "msg_type": "stream",
                "content": {"text": f"epoch 5/10 Authorization: Bearer {token}\n"},
            },
            2,
            logger,
        )

    text = log_path.read_text(encoding="utf-8")
    assert "epoch 5/10" in text
    assert token not in text
    assert "[REDACTED]" in text


def test_discovery_and_quickstart_extraction_are_cwd_independent(
    tmp_path: Path,
) -> None:
    """Discovery finds the vendor child and extraction returns fence bytes exactly."""

    upstream = _make_upstream(tmp_path)

    assert reproduction.find_submodule(tmp_path) == upstream.resolve()
    assert reproduction.find_submodule(upstream / "starters") == upstream.resolve()
    assert (
        reproduction.extract_readme_quickstart(upstream / "README.md")
        == _QUICKSTART_SOURCE
    )


def test_create_run_directory_is_timestamped_and_collision_safe(tmp_path: Path) -> None:
    """Two runs started at the same instant receive separate safe directories."""

    moment = datetime(2026, 7, 22, 18, 30, 45, 123456, tzinfo=timezone.utc)

    first = reproduction.create_run_directory(
        tmp_path, label="README quickstart", now=moment
    )
    second = reproduction.create_run_directory(
        tmp_path, label="README quickstart", now=moment
    )

    assert first.name == "20260722T183045.123456Z-README-quickstart"
    assert second.name == "20260722T183045.123456Z-README-quickstart-1"


def test_preflight_checks_assets_cuda_and_gated_access_without_storing_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful mocked preflight records booleans and versions, never the token."""

    upstream = _make_upstream(tmp_path)
    token = "hf_mocked_token_that_must_never_be_serialized"
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        reproduction,
        "_submodule_commit",
        lambda _root: reproduction.EXPECTED_SUBMODULE_COMMIT,
    )
    monkeypatch.setattr(reproduction, "_submodule_clean", lambda _root: True)

    result = reproduction.preflight_environment(
        upstream,
        token=token,
        required_python=(3, 0),
        required_distributions=("torch", "nbclient"),
        version_getter=lambda _name: "1.2.3",
        expected_versions={"torch": "1.2.3", "nbclient": "1.2.3"},
        torch_module=SimpleNamespace(cuda=_FakeCuda()),
        hf_access_checker=lambda model_id, received: calls.append((model_id, received)),
    )

    serialized = json.dumps(result.to_dict())
    assert result.ok
    assert result.cuda_available
    assert result.cuda_device == "Mock RTX"
    assert result.submodule_commit == reproduction.EXPECTED_SUBMODULE_COMMIT
    assert result.submodule_clean is True
    assert result.gated_model_access
    assert result.hf_token_present
    assert calls == [(reproduction.DINO_MODEL_ID, token)]
    assert token not in serialized


def test_hf_access_probe_targets_the_pinned_dino_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lightweight gated-access check probes the same immutable runtime model."""

    import huggingface_hub

    calls: dict[str, object] = {}

    def fake_url(*, repo_id: str, filename: str, revision: str) -> str:
        """Record immutable URL inputs without contacting the Hub."""

        calls.update(repo_id=repo_id, filename=filename, revision=revision)
        return "https://example.invalid/pinned-config"

    def fake_metadata(url: str, *, token: str) -> None:
        """Record the metadata request without persisting its synthetic token."""

        calls.update(url=url, token=token)

    monkeypatch.setattr(huggingface_hub, "hf_hub_url", fake_url)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", fake_metadata)

    reproduction._check_hf_model_access(reproduction.DINO_MODEL_ID, "synthetic")

    assert calls == {
        "repo_id": reproduction.DINO_MODEL_ID,
        "filename": reproduction.DINO_ACCESS_FILE,
        "revision": reproduction.DINO_REVISION,
        "url": "https://example.invalid/pinned-config",
        "token": "synthetic",
    }


def test_preflight_sanitizes_token_echoed_by_hub_exception(tmp_path: Path) -> None:
    """A dependency echoing an authorization value cannot leak it into reports."""

    upstream = _make_upstream(tmp_path)
    token = "hf_mocked_token_that_must_never_be_serialized"

    def fail_access(_model_id: str, received: str) -> None:
        """Simulate a poorly behaved HTTP client error containing its credential."""

        raise RuntimeError(f"Authorization: Bearer {received}")

    result = reproduction.preflight_environment(
        upstream,
        token=token,
        required_python=(3, 0),
        required_distributions=(),
        expected_commit=None,
        require_clean_submodule=False,
        version_getter=lambda _name: "unused",
        torch_module=SimpleNamespace(cuda=_FakeCuda()),
        hf_access_checker=fail_access,
    )

    serialized = json.dumps(result.to_dict())
    assert not result.ok
    assert not result.gated_model_access
    assert token not in serialized
    assert "[REDACTED]" in serialized


def test_preflight_rejects_wrong_versions_commit_and_dirty_submodule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduction cannot certify dependency or upstream source drift."""

    upstream = _make_upstream(tmp_path)
    monkeypatch.setattr(reproduction, "_submodule_commit", lambda _root: "bad-commit")
    monkeypatch.setattr(reproduction, "_submodule_clean", lambda _root: False)

    result = reproduction.preflight_environment(
        upstream,
        token="mock-token",
        required_python=(3, 0),
        required_distributions=("torch",),
        version_getter=lambda _name: "0.0.0",
        expected_versions={"torch": "2.13.0"},
        torch_module=SimpleNamespace(cuda=_FakeCuda()),
        hf_access_checker=lambda _model_id, _token: None,
    )

    assert not result.ok
    assert any("expected submodule commit" in error for error in result.errors)
    assert any("uncommitted changes" in error for error in result.errors)
    assert any("expected 2.13.0" in error for error in result.errors)


def test_temporary_hf_token_restores_an_absent_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit token leaves no process credential behind after execution."""

    monkeypatch.delenv("HF_TOKEN", raising=False)

    with reproduction._temporary_hf_token("mock-token"):
        assert os.environ["HF_TOKEN"] == "mock-token"

    assert "HF_TOKEN" not in os.environ


def test_readme_runner_executes_extracted_source_and_redacts_captured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README runner snapshots exact code and sanitizes runtime output."""

    upstream = _make_upstream(tmp_path)
    output_dir = tmp_path / "runs" / "readme"
    token = "hf_mocked_token_that_must_never_be_serialized"
    received: list[str] = []
    monkeypatch.setenv("HF_TOKEN", "prior-token")
    monkeypatch.setattr(
        reproduction, "_quickstart_metrics", lambda _namespace: _valid_readme_metrics()
    )

    def fake_executor(source: str, namespace: dict[str, Any]) -> None:
        """Record the exact source and provide lightweight metric variables."""

        import matplotlib.pyplot as plt

        received.append(source)
        assert os.environ["HF_TOKEN"] == token
        print(f"HF_TOKEN={token}")
        namespace["grid"] = 14
        namespace["top"] = [3, 7]
        plt.figure().add_subplot().plot([0, 1], [0, 1])

    result = reproduction.run_readme_quickstart(
        upstream,
        output_dir=output_dir,
        token=token,
        executor=fake_executor,
    )

    assert result.ok
    assert os.environ["HF_TOKEN"] == "prior-token"
    assert received == [_QUICKSTART_SOURCE]
    assert (output_dir / "README-quickstart.py").read_text(
        encoding="utf-8"
    ) == _QUICKSTART_SOURCE
    assert result.metrics == _valid_readme_metrics()
    assert (output_dir / "quickstart-figure-001.png").stat().st_size > 0
    assert (output_dir / "quickstart-figure-001.pdf").stat().st_size > 0
    log_text = Path(result.log_path).read_text(encoding="utf-8")
    assert token not in log_text
    assert "[REDACTED]" in log_text


def test_failed_readme_releases_namespace_figures_and_cuda_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """README cleanup runs even on failure before a notebook kernel can start."""

    import matplotlib.pyplot as plt
    import torch

    upstream = _make_upstream(tmp_path)
    retained_namespaces: list[dict[str, Any]] = []
    cleanup_calls: list[str] = []

    def failing_executor(_source: str, namespace: dict[str, Any]) -> None:
        """Allocate representative README resources and fail before artifact export."""

        retained_namespaces.append(namespace)
        namespace["large_array"] = bytearray(1024)
        namespace["plt"] = plt
        plt.figure().add_subplot().plot([0, 1], [0, 1])
        raise RuntimeError("synthetic README failure")

    monkeypatch.setattr(
        reproduction.gc, "collect", lambda: cleanup_calls.append("gc") or 0
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: cleanup_calls.append("cuda-cache"),
    )

    result = reproduction.run_readme_quickstart(
        upstream,
        output_dir=tmp_path / "runs" / "failed-readme",
        token="mock-token",
        executor=failing_executor,
    )

    assert not result.ok
    assert cleanup_calls == ["gc", "cuda-cache"]
    assert retained_namespaces == [{}]
    assert plt.get_fignums() == []


def test_all_target_finishes_failed_readme_cleanup_before_notebooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a failed README run releases parent GPU state before notebook launch."""

    import torch

    upstream = _make_upstream(tmp_path)
    cleanup_calls: list[str] = []
    namespaces: list[dict[str, Any]] = []
    real_readme_runner = reproduction.run_readme_quickstart

    preflight = reproduction.PreflightResult(
        ok=True,
        checked_at="2026-07-22T18:00:00+00:00",
        submodule_root=str(upstream),
        submodule_commit=reproduction.EXPECTED_SUBMODULE_COMMIT,
        python_version="3.12.0",
        package_versions={},
        assets={},
        cuda_available=True,
        cuda_device="Mock GPU",
        hf_token_present=True,
        gated_model_access=True,
        submodule_clean=True,
    )

    def failed_readme(*args: Any, **kwargs: Any) -> reproduction.ReproductionResult:
        """Run the production finally path with a deliberately failing executor."""

        def fail(_source: str, namespace: dict[str, Any]) -> None:
            namespaces.append(namespace)
            namespace["resource"] = bytearray(1024)
            raise RuntimeError("synthetic failure before notebooks")

        return real_readme_runner(*args, **kwargs, executor=fail)

    def notebooks_after_cleanup(
        *_args: Any, **_kwargs: Any
    ) -> tuple[reproduction.ReproductionResult, ...]:
        """Stand in for kernel creation and assert parent cleanup already finished."""

        assert cleanup_calls == ["gc", "cuda-cache"]
        assert namespaces == [{}]
        return ()

    monkeypatch.setattr(
        reproduction, "preflight_environment", lambda *_args, **_kwargs: preflight
    )
    monkeypatch.setattr(reproduction, "run_readme_quickstart", failed_readme)
    monkeypatch.setattr(reproduction, "run_notebooks", notebooks_after_cleanup)
    monkeypatch.setattr(
        reproduction.gc, "collect", lambda: cleanup_calls.append("gc") or 0
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: cleanup_calls.append("cuda-cache")
    )

    suite = reproduction.run_reproduction(
        "all",
        submodule_root=upstream,
        output_root=tmp_path / "runs",
        token="mock-token",
    )

    assert not suite.ok


def test_notebook_runner_sets_upstream_cwd_preserves_source_and_exports_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mocked nbclient execution receives the correct resource path and hooks."""

    upstream = _make_upstream(tmp_path)
    source = upstream / reproduction.NOTEBOOK_RELATIVE_PATHS[0]
    before = source.read_bytes()
    output_dir = tmp_path / "runs" / "notebook"
    token = "hf_mocked_token_that_must_never_be_serialized"
    monkeypatch.setenv("HF_TOKEN", "prior-token")
    record: dict[str, Any] = {}
    client_factory = _fake_client_factory(
        record,
        f"activations: (58800, 768) patch grid: 14\n"
        "epoch 300/300 loss=0.1000 R2=0.9000 L0=8.0 dead=2/256\n"
        f"top concepts: [np.int64(4), np.int64(8)]\n"
        f"Authorization: Bearer {token}\n",
    )

    result = reproduction.execute_notebook(
        source,
        submodule_root=upstream,
        output_dir=output_dir,
        token=token,
        client_factory=client_factory,
        html_exporter_factory=_FakeHtmlExporter,
    )

    assert result.ok
    assert record["hf_token_during_execute"] == token
    assert os.environ["HF_TOKEN"] == "prior-token"
    assert source.read_bytes() == before
    assert record["resources"] == {"metadata": {"path": str(upstream.resolve())}}
    assert record["kernel_name"] == "python3"
    assert result.metrics["r2"] == pytest.approx(0.9)
    assert result.metrics["top_concepts"] == [4, 8]
    assert result.metrics["activation_shape"] == [58_800, 768]
    assert (output_dir / "01_grassmannian.executed.ipynb").is_file()
    assert (output_dir / "01_grassmannian.html").is_file()
    assert token not in (output_dir / "01_grassmannian.executed.ipynb").read_text(
        encoding="utf-8"
    )
    log_text = Path(result.log_path).read_text(encoding="utf-8")
    assert token not in log_text
    assert "[REDACTED]" in log_text


def test_failed_notebook_run_does_not_reuse_embedded_reference_metrics(
    tmp_path: Path,
) -> None:
    """Historical outputs are cleared before execution and cannot mask a failure."""

    upstream = _make_upstream(tmp_path)
    source = upstream / reproduction.NOTEBOOK_RELATIVE_PATHS[0]
    notebook = nbformat.read(source, as_version=4)
    notebook.cells[0].execution_count = 9
    notebook.cells[0].outputs = [
        nbformat.v4.new_output(
            "stream",
            name="stdout",
            text="epoch 300/300 loss=0.1 R2=0.99 L0=8.0 dead=0/256\n",
        )
    ]
    nbformat.write(notebook, source)
    before = source.read_bytes()

    class FailingClient:
        """Fail before any fresh output is produced."""

        def __init__(self, _notebook: Any, **_kwargs: Any) -> None:
            """Accept the normal nbclient construction interface."""

        def execute(self) -> Any:
            """Simulate a kernel startup or first-cell failure."""

            raise RuntimeError("mock kernel failure")

    result = reproduction.execute_notebook(
        source,
        submodule_root=upstream,
        output_dir=tmp_path / "runs" / "failed-notebook",
        token="mock-token",
        client_factory=FailingClient,
        html_exporter_factory=_FakeHtmlExporter,
    )

    assert not result.ok
    assert result.metrics["training"] == []
    assert result.metrics["executed_code_cells"] == 0
    assert source.read_bytes() == before


def test_readme_runner_fails_acceptance_for_low_r2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed quickstart with a poor reconstruction is still a failed run."""

    upstream = _make_upstream(tmp_path)
    metrics = _valid_readme_metrics() | {"r2": 0.69}
    monkeypatch.setattr(reproduction, "_quickstart_metrics", lambda _namespace: metrics)

    def draw_plot(_source: str, _namespace: dict[str, Any]) -> None:
        """Provide the required nonblank artifact while isolating the R² failure."""

        import matplotlib.pyplot as plt

        plt.figure().add_subplot().plot([0, 1], [0, 1])

    result = reproduction.run_readme_quickstart(
        upstream,
        output_dir=tmp_path / "runs" / "low-r2",
        token="mock-token",
        executor=draw_plot,
    )

    assert not result.ok
    assert result.error is not None
    assert "R²" in result.error


def test_notebook_runner_fails_acceptance_without_training_or_plot(
    tmp_path: Path,
) -> None:
    """Missing quality metrics and visual evidence cannot yield a passing notebook."""

    upstream = _make_upstream(tmp_path)
    record: dict[str, Any] = {}
    result = reproduction.execute_notebook(
        reproduction.NOTEBOOK_RELATIVE_PATHS[0],
        submodule_root=upstream,
        output_dir=tmp_path / "runs" / "invalid-notebook",
        token="mock-token",
        client_factory=_fake_client_factory(
            record, "top concepts: []\n", include_image=False
        ),
        html_exporter_factory=_FakeHtmlExporter,
    )

    assert not result.ok
    assert result.error is not None
    assert "Acceptance validation failed" in result.error
    assert "training metrics" in result.error
    assert "nonblank raster plot" in result.error


def test_notebook_acceptance_rejects_a_uniform_image_artifact(tmp_path: Path) -> None:
    """A nonempty but visually blank image cannot satisfy plot acceptance."""

    blank_plot = tmp_path / "blank.png"
    Image.new("RGB", (8, 8), color="white").save(blank_plot)
    metrics = reproduction._notebook_metrics(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    outputs=[
                        nbformat.v4.new_output(
                            "stream", name="stdout", text=_valid_notebook_output()
                        )
                    ],
                    execution_count=1,
                )
            ]
        )
    )

    errors = reproduction._validate_notebook_acceptance(metrics, [blank_plot])

    assert any("nonblank raster plot" in error for error in errors)


def test_run_notebooks_uses_all_three_upstream_sources(tmp_path: Path) -> None:
    """The default notebook set covers every supported upstream featurizer."""

    upstream = _make_upstream(tmp_path)
    record: dict[str, Any] = {}
    client_factory = _fake_client_factory(record, _valid_notebook_output())

    results = reproduction.run_notebooks(
        upstream,
        output_dir=tmp_path / "runs" / "notebooks",
        token="mock-token",
        client_factory=client_factory,
        html_exporter_factory=_FakeHtmlExporter,
    )

    assert [result.name for result in results] == [
        "01_grassmannian",
        "02_group_lasso",
        "03_vanilla",
    ]
    assert all(result.ok for result in results)


def test_suite_stops_before_execution_when_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid environment yields an inspectable suite without heavy work."""

    upstream = _make_upstream(tmp_path)
    failed_preflight = reproduction.PreflightResult(
        ok=False,
        checked_at="2026-07-22T18:00:00+00:00",
        submodule_root=str(upstream),
        submodule_commit=None,
        python_version="3.12.0",
        package_versions={},
        assets={},
        cuda_available=False,
        cuda_device=None,
        hf_token_present=True,
        gated_model_access=True,
        errors=("CUDA is required",),
    )
    monkeypatch.setattr(
        reproduction,
        "preflight_environment",
        lambda *_args, **_kwargs: failed_preflight,
    )
    monkeypatch.setattr(
        reproduction,
        "run_readme_quickstart",
        lambda *_args, **_kwargs: pytest.fail("README execution must not start"),
    )
    monkeypatch.setattr(
        reproduction,
        "run_notebooks",
        lambda *_args, **_kwargs: pytest.fail("Notebook execution must not start"),
    )

    suite = reproduction.run_reproduction(
        "all",
        submodule_root=upstream,
        output_root=tmp_path / "runs",
        token="mock-token",
    )

    assert not suite.ok
    assert suite.results == ()
    assert Path(suite.output_dir, "summary.json").is_file()
