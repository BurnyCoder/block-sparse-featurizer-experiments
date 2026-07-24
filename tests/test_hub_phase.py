"""Tests for bounded, immutable Hugging Face checkpoint resolution."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from bsf_experiments.hub_phase import (
    CHECKPOINT_CATALOG,
    HubDownloadMetadata,
    HubCheckpointSpec,
    download_hub_checkpoint,
    get_hub_checkpoint_spec,
)
from bsf_experiments.presets import PRESETS
from bsf_experiments.types import (
    FeaturizerKind,
    ModelConfig,
    PretrainedRecipe,
)


_REVISION = "1" * 40


def _spec(payload: bytes) -> HubCheckpointSpec:
    """Return one valid synthetic immutable catalog record."""

    return HubCheckpointSpec(
        repo_id="BurnyCoder/test-bsf",
        revision=_REVISION,
        filename="checkpoint.pt",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        max_bytes=1024,
        input_dim=3,
        model_config=ModelConfig(
            kind=FeaturizerKind.GRASSMANNIAN,
            n_groups=2,
            group_size=1,
            l0=1,
        ),
    )


def test_catalog_has_one_immutable_record_for_every_recipe() -> None:
    """Every selectable recipe is pinned to its published repository and bytes."""

    assert isinstance(CHECKPOINT_CATALOG, MappingProxyType)
    assert set(CHECKPOINT_CATALOG) == set(PretrainedRecipe)
    expected = {
        PretrainedRecipe.GRASSMANNIAN_NOTEBOOK: (
            "BurnyCoder/bsf-dinov3-rabbits-grassmannian-notebook",
            "6d874cd7c713d0464ec1769cb667f08aeb43720e",
            "f029890c4fa34fe9dcaf350d03870b5b3f035daa3d1fe97c457299d76754748d",
            2_362_853,
            PRESETS["grassmannian_notebook"].model,
        ),
        PretrainedRecipe.GROUP_LASSO_NOTEBOOK: (
            "BurnyCoder/bsf-dinov3-rabbits-group-lasso-notebook",
            "c0e9c501963ed28d022ecce5fd7b7beafed4720f",
            "46d8d0a68e263f4518350f9334959d3d349bf26d347be013f748da8aa660fde8",
            4_726_389,
            PRESETS["group_lasso_notebook"].model,
        ),
        PretrainedRecipe.VANILLA_NOTEBOOK: (
            "BurnyCoder/bsf-dinov3-rabbits-vanilla-notebook",
            "bcfdceb086e57f2d5f64d0036a1d08cdc8610442",
            "b87e2a9548abf5c909152ef2f7f89085bbd614a81981db24e976362849aa9d06",
            4_724_489,
            PRESETS["vanilla_notebook"].model,
        ),
        PretrainedRecipe.README_QUICKSTART: (
            "BurnyCoder/bsf-dinov3-rabbits-readme-quickstart",
            "4f1fc7b7ce3da7c8a41325fc06ee4d3aee11a2ca",
            "449e7dfe65587f2959d8263197805e2f2f33cc7c391dbeb7949539fb58a8e321",
            2_362_853,
            PRESETS["readme"].model,
        ),
    }
    for recipe, (
        repo_id,
        revision,
        sha256,
        size_bytes,
        model_config,
    ) in expected.items():
        spec = get_hub_checkpoint_spec(recipe)
        assert (
            spec.repo_id,
            spec.revision,
            spec.sha256,
            spec.size_bytes,
            spec.model_config,
        ) == (repo_id, revision, sha256, size_bytes, model_config)
        assert spec.input_dim == 768
        assert spec.filename == "checkpoint.pt"
    with pytest.raises(TypeError):
        CHECKPOINT_CATALOG[PretrainedRecipe.README_QUICKSTART] = _spec(b"x")  # type: ignore[index]


def test_incomplete_catalog_entries_fail_closed() -> None:
    """Placeholder-like metadata cannot trigger an unpinned Hub download."""

    placeholder = replace(
        _spec(b"x"),
        revision="PENDING_PUBLICATION",
        sha256="PENDING_PUBLICATION",
    )
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        get_hub_checkpoint_spec(
            PretrainedRecipe.README_QUICKSTART,
            catalog={PretrainedRecipe.README_QUICKSTART: placeholder},
        )


def test_download_preflights_then_verifies_size_revision_and_sha256(
    tmp_path: Path,
) -> None:
    """A pinned checkpoint is returned only after both remote and local checks."""

    payload = b"safe tensor-only checkpoint fixture"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(payload)
    spec = _spec(payload)
    calls: list[dict[str, object]] = []
    metadata = []

    def fake_download(**kwargs):
        calls.append(dict(kwargs))
        if kwargs["dry_run"]:
            return SimpleNamespace(
                commit_hash=_REVISION,
                file_size=len(payload),
                filename="checkpoint.pt",
                is_cached=True,
            )
        return str(checkpoint)

    resolved = download_hub_checkpoint(
        spec,
        downloader=fake_download,
        metadata_callback=metadata.append,
    )

    assert resolved == checkpoint
    assert metadata == [
        HubDownloadMetadata(
            repo_id=spec.repo_id,
            filename=spec.filename,
            resolved_commit=spec.revision,
            remote_size=len(payload),
            is_cached=True,
        )
    ]
    assert calls == [
        {
            "repo_id": spec.repo_id,
            "filename": spec.filename,
            "repo_type": "model",
            "revision": spec.revision,
            "dry_run": True,
        },
        {
            "repo_id": spec.repo_id,
            "filename": spec.filename,
            "repo_type": "model",
            "revision": spec.revision,
            "dry_run": False,
        },
    ]


@pytest.mark.parametrize(
    ("dry_run", "message"),
    [
        (SimpleNamespace(commit_hash="2" * 40, file_size=5), "revision"),
        (SimpleNamespace(commit_hash=_REVISION, file_size=2048), "download limit"),
    ],
)
def test_download_rejects_bad_preflight_without_fetching(
    dry_run: SimpleNamespace,
    message: str,
) -> None:
    """A mutable revision or oversized remote object stops before a real download."""

    spec = _spec(b"small")
    calls = 0

    def fake_download(**_kwargs):
        nonlocal calls
        calls += 1
        return dry_run

    with pytest.raises(ValueError, match=message):
        download_hub_checkpoint(spec, downloader=fake_download)

    assert calls == 1


def test_download_rejects_remote_size_that_differs_from_catalog_pin() -> None:
    """A changed object size is rejected before downloading even below the budget."""

    spec = _spec(b"small")
    calls = 0

    def fake_download(**_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            commit_hash=_REVISION,
            file_size=spec.size_bytes + 1,
            is_cached=False,
        )

    with pytest.raises(ValueError, match="trusted catalog size"):
        download_hub_checkpoint(spec, downloader=fake_download)

    assert calls == 1


def test_download_rejects_post_download_size_and_digest_mismatches(
    tmp_path: Path,
) -> None:
    """A changed or corrupt cache object never reaches the checkpoint loader."""

    advertised = b"advertised"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"different!")
    spec = _spec(advertised)

    def fake_download(**kwargs):
        if kwargs["dry_run"]:
            return SimpleNamespace(
                commit_hash=_REVISION,
                file_size=len(advertised),
                filename="checkpoint.pt",
            )
        return str(checkpoint)

    with pytest.raises(ValueError, match="SHA-256"):
        download_hub_checkpoint(spec, downloader=fake_download)

    checkpoint.write_bytes(b"short")
    with pytest.raises(ValueError, match="trusted catalog size"):
        download_hub_checkpoint(spec, downloader=fake_download)

    checkpoint.write_bytes(b"x" * 2048)
    with pytest.raises(ValueError, match="download limit"):
        download_hub_checkpoint(spec, downloader=fake_download)


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "main"},
        {"sha256": ""},
        {"filename": "../checkpoint.pt"},
        {"size_bytes": 0},
        {"max_bytes": 0},
        {"size_bytes": 2_048},
        {"input_dim": 0},
    ],
)
def test_invalid_injected_catalog_record_fails_before_download(
    change: dict[str, object],
) -> None:
    """Tests and future callers cannot bypass catalog validation through injection."""

    spec = replace(_spec(b"x"), **change)
    catalog = {PretrainedRecipe.README_QUICKSTART: spec}

    with pytest.raises(ValueError):
        get_hub_checkpoint_spec(
            PretrainedRecipe.README_QUICKSTART,
            catalog=catalog,
        )
