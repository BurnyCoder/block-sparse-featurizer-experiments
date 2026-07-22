# Repository working agreement

## Global context

This outer project orchestrates, tests, and visualizes the pinned
`vendor/block-sparse-featurizer` research library. Keep upstream algorithms in
the submodule and put application concerns in the outer package.

## Local development rules

- Use Python 3.12, `uv`, the repository `.venv`, and the committed `uv.lock`.
- Keep `.env`, credentials, generated models, logs, and experiment outputs out
  of Git. Never print or log secret values.
- Build behavior test-first and run the relevant unit and integration tests
  before committing.
- Keep `ExperimentPipeline` as the readable orchestration boundary; implement
  individual phases in focused modules and avoid duplicated workflows.
- Add module and callable docstrings plus focused inline comments that explain
  local intent and link to authoritative upstream or library documentation
  where an integration choice is non-obvious.
- Update README documentation and its Mermaid architecture diagram whenever a
  public workflow, configuration option, or dependency changes.
- Use feature branches, meaningful commits, pull requests, and review before
  merging to `main`.
