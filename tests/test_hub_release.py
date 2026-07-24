"""Offline tests for exact BSF training and Hugging Face release staging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bsf_experiments import hub_release
from bsf_experiments.model_phase import create_model
from bsf_experiments.types import FeaturizerKind


def test_recipe_catalog_matches_all_four_original_workflows() -> None:
    """The release catalog preserves constructor calls and trainer defaults."""

    recipes = hub_release.RELEASE_RECIPES

    assert tuple(recipes) == (
        "grassmannian-notebook",
        "group-lasso-notebook",
        "vanilla-notebook",
        "readme-quickstart",
    )
    assert recipes["readme-quickstart"].model_config.kind is FeaturizerKind.GRASSMANNIAN
    assert recipes["readme-quickstart"].model_config.l0 == 16
    assert recipes["readme-quickstart"].constructor_kwargs == {
        "d": 768,
        "n_groups": 256,
        "group_size": 3,
        "l0": 16,
    }
    assert recipes["readme-quickstart"].train_kwargs == {"epochs": 60}

    grassmannian = recipes["grassmannian-notebook"]
    assert grassmannian.model_config.l0 == 8
    assert grassmannian.train_kwargs == {"epochs": 300, "lr": 3e-3}

    group_lasso = recipes["group-lasso-notebook"]
    assert group_lasso.model_config.kind is FeaturizerKind.GROUP_LASSO
    assert group_lasso.model_config.target_l0 == 8
    assert group_lasso.model_config.coef == 1e-2
    assert group_lasso.model_config.gain == 10.0
    assert group_lasso.model_config.paper_version is False
    assert group_lasso.constructor_kwargs == {
        "d": "activation_width",
        "n_groups": 256,
        "group_size": 3,
        "target_l0": 8,
    }
    assert group_lasso.train_kwargs == {"epochs": 300}

    vanilla = recipes["vanilla-notebook"]
    assert vanilla.model_config.kind is FeaturizerKind.VANILLA
    assert vanilla.model_config.l0 == 8
    assert vanilla.train_kwargs == {"epochs": 300, "lr": 3e-3}

    assert all(recipe.effective_batch_size == 2048 for recipe in recipes.values())
    assert all(recipe.effective_snr == 0.1 for recipe in recipes.values())


def test_verify_exact_source_requires_pin_and_every_expected_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source checkout is accepted only when commit and blob IDs are exact."""

    for relative in hub_release.EXPECTED_SOURCE_BLOBS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    calls: list[tuple[str, ...]] = []

    def fake_git(_root: Path, *arguments: str) -> str:
        calls.append(arguments)
        if arguments == ("rev-parse", "HEAD"):
            return hub_release.UPSTREAM_COMMIT
        if arguments[:2] == ("status", "--porcelain"):
            return ""
        if arguments[:4] == (
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
        ):
            return ""
        revision_path = arguments[-1].split(":", maxsplit=1)[1]
        return hub_release.EXPECTED_SOURCE_BLOBS[revision_path]

    monkeypatch.setattr(hub_release, "_git", fake_git)

    evidence = hub_release.verify_exact_source(tmp_path)

    assert evidence["commit"] == hub_release.UPSTREAM_COMMIT
    assert evidence["blobs"] == hub_release.EXPECTED_SOURCE_BLOBS
    assert ("status", "--porcelain", "--untracked-files=all") in calls
    assert (
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "bsf",
    ) in calls

    def fake_shadow_git(root: Path, *arguments: str) -> str:
        if arguments[:4] == (
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
        ):
            return "bsf/__pycache__/data.cpython-312.pyc"
        return fake_git(root, *arguments)

    monkeypatch.setattr(hub_release, "_git", fake_shadow_git)
    with pytest.raises(ValueError, match="shadow/cache"):
        hub_release.verify_exact_source(tmp_path)

    monkeypatch.setattr(
        hub_release,
        "_git",
        lambda _root, *arguments: (
            "f" * 40
            if arguments == ("rev-parse", "HEAD")
            else ""
            if arguments[:2] == ("status", "--porcelain")
            or arguments[:4]
            == ("ls-files", "--others", "--ignored", "--exclude-standard")
            else hub_release.EXPECTED_SOURCE_BLOBS[
                arguments[-1].split(":", maxsplit=1)[1]
            ]
        ),
    )
    with pytest.raises(ValueError, match="exact upstream commit"):
        hub_release.verify_exact_source(tmp_path)


class _FakeModel:
    """Small stand-in whose constructor metadata can be inspected by tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.d = int(kwargs["d"])


class _FakeUpstream:
    """Faithful public surface needed by the exact release runner."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.data = SimpleNamespace(
            POS_MEAN=np.zeros((4, 3), dtype=np.float32),
            load_rabbit_images=self._load_images,
            dino_activations=self._extract,
            patch_grid=lambda patches: 2 if patches == 4 else 0,
        )
        self.GrassmannianBSF = self._model
        self.GroupLassoBSF = self._model
        self.VanillaBSF = self._model

    def _load_images(self, path: Path) -> np.ndarray:
        self.record["rabbit_path"] = path
        return np.zeros((2, 2, 2, 3), dtype=np.uint8)

    def _extract(self, images: np.ndarray) -> np.ndarray:
        self.record["images_shape"] = images.shape
        return np.arange(24, dtype=np.float32).reshape(2, 4, 3)

    def _model(self, **kwargs: Any) -> _FakeModel:
        self.record["constructor_kwargs"] = kwargs
        return _FakeModel(**kwargs)

    def train(self, model: _FakeModel, x: np.ndarray, **kwargs: Any) -> _FakeModel:
        self.record["train_kwargs"] = kwargs
        self.record["training_shape"] = x.shape
        self.record["training_dtype"] = x.dtype
        return model


def _source_fixture(root: Path) -> None:
    """Write the trusted binary/text inputs consumed after Git verification."""

    (root / "bsf").mkdir(parents=True)
    (root / "rabbit.npz").write_bytes(b"rabbit-data")
    (root / "bsf/pos_mean.npy").write_bytes(b"positional-mean")
    (root / "LICENSE").write_text("Goodfire MIT license\n", encoding="utf-8")


def test_train_exact_recipe_preserves_original_calls_and_curates_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful training records provenance and stages only approved files."""

    source = tmp_path / "source"
    _source_fixture(source)
    record: dict[str, Any] = {}
    upstream = _FakeUpstream(record)
    monkeypatch.setattr(
        hub_release,
        "verify_exact_source",
        lambda _root: {
            "repository": hub_release.UPSTREAM_REPOSITORY,
            "commit": hub_release.UPSTREAM_COMMIT,
            "blobs": dict(hub_release.EXPECTED_SOURCE_BLOBS),
        },
    )
    monkeypatch.setattr(hub_release, "_load_exact_upstream", lambda _root: upstream)
    monkeypatch.setattr(hub_release, "_validate_exact_data_shapes", lambda *_args: None)
    revisions = iter(
        (hub_release.EXPECTED_DINO_REVISION, hub_release.EXPECTED_DINO_REVISION)
    )
    monkeypatch.setattr(
        hub_release, "resolve_dino_revision", lambda _model_id: next(revisions)
    )
    monkeypatch.setattr(
        hub_release,
        "evaluate_reconstruction",
        lambda _model, _x, **_kwargs: {
            "r2": 0.81,
            "mean_l0": 8.0,
            "dead_groups": 1,
            "tokens": 8,
        },
    )
    monkeypatch.setattr(
        hub_release.torch,
        "manual_seed",
        lambda *_args, **_kwargs: pytest.fail("release training must not seed Torch"),
    )

    def fake_checkpoint(
        destination: Path,
        _model: _FakeModel,
        _config: Any,
        *,
        input_dim: int,
    ) -> Path:
        assert input_dim == 3
        destination.write_bytes(b"safe-v1-checkpoint")
        return destination

    monkeypatch.setattr(hub_release, "save_checkpoint", fake_checkpoint)
    dino_license = tmp_path / "DINO-LICENSE.md"
    dino_license.write_text("DINOv3 license\n", encoding="utf-8")
    monkeypatch.setattr(
        hub_release,
        "download_dino_license",
        lambda _revision: dino_license,
    )
    monkeypatch.setattr(
        hub_release,
        "validate_release_bundle",
        lambda path, **_kwargs: hub_release.validate_staging_directory(path),
    )

    result = hub_release.train_exact_recipe(
        "grassmannian-notebook",
        source_root=source,
        output_root=tmp_path / "release",
        collection_url="https://huggingface.co/collections/BurnyCoder/example-123",
    )

    assert record["constructor_kwargs"] == {
        "d": 3,
        "n_groups": 256,
        "group_size": 3,
        "l0": 8,
    }
    assert record["train_kwargs"] == {"epochs": 300, "lr": 3e-3}
    assert record["training_shape"] == (8, 3)
    assert record["training_dtype"] == np.dtype(np.float64)
    assert set(path.name for path in result.stage_dir.iterdir()) == {
        "checkpoint.pt",
        "manifest.json",
        "README.md",
        "LICENSE-goodfire.txt",
        "LICENSE-dinov3.md",
    }
    manifest = json.loads((result.stage_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["source"]["commit"] == hub_release.UPSTREAM_COMMIT
    assert manifest["dino"]["revision"] == hub_release.EXPECTED_DINO_REVISION
    assert manifest["metrics"]["r2"] == 0.81
    assert isinstance(manifest["randomness"]["torch_initial_seed"], int)
    assert manifest["inputs"]["rabbit_npz"]["sha256"]
    assert manifest["inputs"]["positional_mean"]["sha256"]
    assert manifest["inputs"]["preprocessed_activations"]["sha256"]
    assert (
        manifest["licenses"]["goodfire"]["sha256"]
        == hashlib.sha256(b"Goodfire MIT license\n").hexdigest()
    )
    assert (
        manifest["licenses"]["dinov3"]["sha256"]
        == hashlib.sha256(b"DINOv3 license\n").hexdigest()
    )
    model_card = (result.stage_dir / "README.md").read_text(encoding="utf-8")
    assert "license: other" in model_card
    assert "license_name: dinov3-license" in model_card
    assert "library_name: block-sparse-featurizer" in model_card
    assert "get_hub_checkpoint_spec" in model_card
    assert "download_hub_checkpoint" in model_card
    assert hub_release.APPLICATION_REPOSITORY in model_card
    assert "uv sync --frozen" in model_card
    assert "https://huggingface.co/collections/BurnyCoder/example-123" in model_card
    hub_release.validate_staging_directory(result.stage_dir)


def test_r2_gate_prevents_release_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A weak trained model leaves evidence locally but no publishable folder."""

    source = tmp_path / "source"
    _source_fixture(source)
    monkeypatch.setattr(
        hub_release,
        "verify_exact_source",
        lambda _root: {
            "repository": hub_release.UPSTREAM_REPOSITORY,
            "commit": hub_release.UPSTREAM_COMMIT,
            "blobs": dict(hub_release.EXPECTED_SOURCE_BLOBS),
        },
    )
    monkeypatch.setattr(
        hub_release, "_load_exact_upstream", lambda _root: _FakeUpstream({})
    )
    monkeypatch.setattr(hub_release, "_validate_exact_data_shapes", lambda *_args: None)
    revisions = iter(
        (hub_release.EXPECTED_DINO_REVISION, hub_release.EXPECTED_DINO_REVISION)
    )
    monkeypatch.setattr(
        hub_release, "resolve_dino_revision", lambda _model_id: next(revisions)
    )
    monkeypatch.setattr(
        hub_release,
        "evaluate_reconstruction",
        lambda *_args, **_kwargs: {"r2": 0.69},
    )
    monkeypatch.setattr(
        hub_release,
        "save_checkpoint",
        lambda destination, *_args, **_kwargs: (
            destination.write_bytes(b"checkpoint") and destination
        ),
    )

    with pytest.raises(ValueError, match="R² gate"):
        hub_release.train_exact_recipe(
            "vanilla-notebook",
            source_root=source,
            output_root=tmp_path / "release",
            collection_url="https://huggingface.co/collections/BurnyCoder/example-123",
        )

    assert not list((tmp_path / "release").glob("*/stage"))


def test_bundle_validation_failure_leaves_no_publishable_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atomic staging never exposes files that fail the final bundle checks."""

    run_dir = tmp_path / "run"
    source = tmp_path / "source"
    run_dir.mkdir()
    source.mkdir()
    (source / "LICENSE").write_text("Goodfire terms\n", encoding="utf-8")
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"invalid")
    dino_license = tmp_path / "DINO-LICENSE.md"
    dino_license.write_text("DINO terms\n", encoding="utf-8")
    monkeypatch.setattr(
        hub_release,
        "validate_release_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("synthetic bundle failure")
        ),
    )
    monkeypatch.setattr(hub_release, "_model_card", lambda *_args: "model card\n")

    with pytest.raises(ValueError, match="synthetic bundle failure"):
        hub_release._stage_release(
            run_dir=run_dir,
            source_root=source,
            checkpoint_path=checkpoint,
            manifest={},
            recipe=hub_release.RELEASE_RECIPES["readme-quickstart"],
            dino_license_path=dino_license,
            collection_url=hub_release.DEFAULT_COLLECTION_URL,
        )

    assert not (run_dir / "stage").exists()
    assert not list(run_dir.glob(".stage-*"))


def test_release_bundle_cross_checks_manifest_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication accepts only a manifest compatible with the hardened v1 file."""

    stage = tmp_path / "stage"
    stage.mkdir()
    recipe = hub_release.RELEASE_RECIPES["grassmannian-notebook"]
    config = recipe.model_config
    model = create_model(config, input_dim=768)
    checkpoint = hub_release.save_checkpoint(
        stage / "checkpoint.pt",
        model,
        config,
        input_dim=768,
    )
    checkpoint_bytes = checkpoint.read_bytes()
    goodfire_license = "Goodfire fixture terms\n"
    dino_license = "DINO fixture terms\n"
    goodfire_license_sha = hashlib.sha256(goodfire_license.encode()).hexdigest()
    dino_license_sha = hashlib.sha256(dino_license.encode()).hexdigest()
    monkeypatch.setattr(
        hub_release,
        "EXPECTED_GOODFIRE_LICENSE_SHA256",
        goodfire_license_sha,
    )
    monkeypatch.setattr(
        hub_release,
        "EXPECTED_DINO_LICENSE_SHA256",
        dino_license_sha,
    )
    manifest = {
        "schema_version": 1,
        "metrics": {"r2": 0.81},
        "acceptance": {
            "passed": True,
            "minimum_reconstruction_r2": 0.70,
        },
        "randomness": {
            "manual_seed_set": False,
            "torch_initial_seed": 123,
        },
        "timing": {
            "started_at": "2026-07-23T00:00:00+00:00",
            "finished_at": "2026-07-23T00:00:01+00:00",
            "duration_seconds": 1.0,
        },
        "environment": {"python": "3.12"},
        "source": {
            "repository": hub_release.UPSTREAM_REPOSITORY,
            "commit": hub_release.UPSTREAM_COMMIT,
            "blobs": dict(hub_release.EXPECTED_SOURCE_BLOBS),
        },
        "dino": {
            "model_id": hub_release.DINO_MODEL_ID,
            "revision": hub_release.EXPECTED_DINO_REVISION,
            "patch_grid": 14,
        },
        "checkpoint": {
            "filename": "checkpoint.pt",
            "format_version": 1,
            "size_bytes": len(checkpoint_bytes),
            "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        },
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
            "model_config": hub_release._json_model_config(config),
        },
        "inputs": {
            "rabbit_npz": {
                "sha256": "b" * 64,
                "git_blob": hub_release.EXPECTED_SOURCE_BLOBS["rabbit.npz"],
            },
            "positional_mean": {
                "sha256": "c" * 64,
                "git_blob": hub_release.EXPECTED_SOURCE_BLOBS["bsf/pos_mean.npy"],
            },
            "preprocessed_activations": {
                "shape": [58_800, 768],
                "dtype": "float64",
                "sha256": "a" * 64,
            },
        },
        "licenses": {
            "goodfire": {
                "filename": "LICENSE-goodfire.txt",
                "sha256": goodfire_license_sha,
                "git_blob": hub_release.EXPECTED_SOURCE_BLOBS["LICENSE"],
            },
            "dinov3": {
                "filename": "LICENSE-dinov3.md",
                "sha256": dino_license_sha,
                "revision": hub_release.EXPECTED_DINO_REVISION,
            },
        },
        "collection_url": hub_release.DEFAULT_COLLECTION_URL,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (stage / "README.md").write_text(
        hub_release._model_card(
            recipe,
            manifest,
            hub_release.DEFAULT_COLLECTION_URL,
        ),
        encoding="utf-8",
    )
    (stage / "LICENSE-goodfire.txt").write_text(
        goodfire_license,
        encoding="utf-8",
    )
    (stage / "LICENSE-dinov3.md").write_text(
        dino_license,
        encoding="utf-8",
    )

    assert (
        hub_release.validate_release_bundle(
            stage,
            expected_recipe_id=recipe.recipe_id,
        )
        == stage.resolve()
    )

    with pytest.raises(ValueError, match="publication target"):
        hub_release.validate_release_bundle(
            stage,
            expected_recipe_id="vanilla-notebook",
        )

    manifest["metrics"]["r2"] = 0.69
    (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="R² gate"):
        hub_release.validate_release_bundle(stage)

    manifest["metrics"]["r2"] = 0.81
    manifest["recipe"]["model_config"]["kind"] = FeaturizerKind.VANILLA.value
    (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exact upstream recipe"):
        hub_release.validate_release_bundle(stage)

    manifest["recipe"]["model_config"] = hub_release._json_model_config(config)
    fake_token = "hf_aSecureSyntheticCredential123"
    monkeypatch.setenv("HF_TOKEN", fake_token)
    manifest["unexpected_secret"] = fake_token
    (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="credential material"):
        hub_release.validate_release_bundle(stage)


def test_publication_plan_uses_hf_cli_without_token_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication is explicit, reviewable, and delegates auth to HF_TOKEN."""

    stage = tmp_path / "stage"
    stage.mkdir()
    for name in hub_release.STAGED_FILENAMES:
        (stage / name).write_bytes(b"x")
    monkeypatch.setattr(
        hub_release,
        "validate_release_bundle",
        lambda path, **_kwargs: hub_release.validate_staging_directory(path),
    )

    plan = hub_release.build_publication_plan(
        "grassmannian-notebook",
        stage_dir=stage,
    )
    flattened = "\n".join(" ".join(command) for command in plan.commands)

    assert plan.repo_id == ("BurnyCoder/bsf-dinov3-rabbits-grassmannian-notebook")
    assert plan.commands[0][:3] == ("hf", "repos", "create")
    assert plan.commands[1][:2] == ("hf", "upload")
    assert "--token" not in flattened
    assert "HF_TOKEN" not in flattened

    (stage / "training.log").write_text("not curated", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        hub_release.build_publication_plan(
            "grassmannian-notebook",
            stage_dir=stage,
        )


def test_release_main_emits_machine_readable_publication_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module entry point can be wired directly to a project script."""

    stage = tmp_path / "stage"
    stage.mkdir()
    for name in hub_release.STAGED_FILENAMES:
        (stage / name).write_bytes(b"x")
    monkeypatch.setattr(
        hub_release,
        "validate_release_bundle",
        lambda path, **_kwargs: hub_release.validate_staging_directory(path),
    )

    status = hub_release.release_main(
        ["plan-publish", "readme-quickstart", "--stage-dir", str(stage)]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["repo_id"].endswith("readme-quickstart")
    assert isinstance(payload["commands"], list)
