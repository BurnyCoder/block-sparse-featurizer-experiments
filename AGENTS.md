# Repository working agreement

## Purpose and boundaries

This outer project reproduces, orchestrates, tests, and visualizes the pinned
`vendor/block-sparse-featurizer` research library. Keep algorithmic changes in
the submodule. Keep data validation, application state, security, artifacts,
logging, and presentation in the outer `bsf_experiments` package.

`ExperimentPipeline` is the one readable orchestration boundary. The CLI,
Gradio app, and Jupyter launcher must call that boundary instead of duplicating
workflow logic.

## Module map

- `types.py`: framework-neutral configuration, `ModelSource`/
  `PretrainedRecipe`, state, event, and concept records.
- `backbone_identity.py`: immutable DINOv3 model ID and release revision shared
  by runtime extraction, reproduction preflight, and release evidence.
- `config.py`: non-secret `.env` loading and local-only settings validation.
- `data_phase.py`: RGB ingestion, fixed DINO extraction, centering, and scaling.
- `model_phase.py`: validated factory for the three upstream featurizers.
- `training_phase.py`: deterministic adapter around upstream training hooks.
- `hub_phase.py`: trusted immutable Hub catalog, preflight, cache, and integrity
  checks.
- `hub_release.py`: exact unseeded upstream training, evidence, release gating,
  curated staging, and reviewable `hf` CLI publication plans.
- `analysis_phase.py`: encoding, reconstruction metrics, atoms, and ranking.
- `visualization_phase.py`: validation plus upstream `plot_concepts` delegation.
- `artifacts.py`: safe checkpoints, arrays, figures, and result bundles.
- `sessions.py`: locked server-side state, TTL, reset, and cancellation.
- `logging_utils.py`: complete timestamped terminal/file logs with redaction.
- `pipeline.py`: imported phases assembled into public workflows.
- `ui.py`: presentation/event adapters only; browser state stays scalar.
- `reproduction.py`: exact original README/notebook execution and acceptance gates.

## Local development

- Use Python 3.12, `uv`, the repository `.venv`, and committed `uv.lock`.
- Initialize submodules before syncing: `git submodule update --init --recursive`.
- Keep `.env`, credentials, weights, logs, caches, and generated outputs out of
  Git. Never print, stage, copy, or document the value of `HF_TOKEN`.
- Before every commit, verify `.env` is ignored/untracked and scan staged content
  plus generated logs for credential patterns and the exact local token value.
- Add behavior test-first, then run focused tests and `uv run pytest`.
- Run `uv run ruff check .` and `uv run ruff format --check .` before review.
- Run `uv run pytest -m gpu` only when CUDA and gated DINOv3 access are expected.
- Exercise changed UI flows through the local browser, including a validation
  failure, cancellation/reset, Train and Hugging Face sources, downloads, and a
  fresh checkpoint reload.

## Design and security rules

- Keep phase implementations focused and reusable; do not copy numerical logic
  into callbacks or UI handlers.
- Store models and arrays only in `SessionRegistry`; `gr.State` holds an opaque
  session ID and nothing credential-bearing or large.
- Serialize every GPU-capable Gradio event through the shared concurrency group.
- Preserve loopback-only launch, `share=False`, the generated-output allowlist,
  upload limits, cache expiry, and `.env` blocked path.
- Save checkpoints as an allowlisted primitive configuration plus CPU
  `state_dict`; load with `weights_only=True`, `map_location="cpu"`, and strict
  application schema validation after enforcing archive-member and expanded
  storage budgets.
- Treat the Hugging Face collection as discovery metadata only. Resolve
  checkpoints through the static `PretrainedRecipe` catalog at full commit
  hashes, preflight remote size, reuse the standard cache, and verify local size
  plus SHA-256 before strict loading.
- Hub-mode failures must remain visible and must never silently invoke local
  training. Public checkpoint downloads must not require an explicit token;
  gated DINO extraction continues to use environment-based authentication and
  must pass the shared full revision to both Transformers loaders.
- Reject pickle-backed NPZ/object arrays and path traversal in every export.
- Sanitize structured config, messages, exceptions, and full traceback text
  before either logger handler formats them.
- Cancellation must be cooperative and reset-safe: a stale worker may never
  repopulate state after reset or expiry.
- Link non-obvious integration decisions to authoritative library documentation
  in nearby module/callable comments. Keep comments explanatory, not mechanical.

## Documentation and delivery

- Update `README.md` whenever commands, controls, settings, artifacts, public
  workflows, dependencies, reproduction gates, or architecture change.
- Keep the Mermaid graph aligned with the real pipeline and module boundaries.
- Keep release metrics and immutable repository/checkpoint identities in each
  model's `manifest.json` and the trusted catalog rather than copying values
  into documentation. Publish only the curated five-file stage, include both
  upstream license files, and use token-free `hf` CLI arguments with credentials
  supplied from the ignored environment.
- Commit on a feature branch in meaningful units. Review correctness, security,
  maintainability, reliability, and architecture once before opening a PR.
- Use a pull request for outer changes. For upstream behavior, merge an upstream
  PR first and repin the outer submodule to that merge commit.
- Never publish generated data or secrets. Public GitHub history must remain
  independently cloneable with `--recurse-submodules` and `uv sync --frozen`.
