"""Readable orchestration boundary for every local BSF workbench phase.

Global context
--------------
The phase modules own numerical and filesystem implementation details. This
wrapper sequences those phases, updates locked server-side session state, and
records complete sanitized events for the notebook, CLI, and Gradio frontends.
Only opaque session IDs need to cross a presentation boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import copy
import logging
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np

from .analysis_phase import (
    encode_features,
    evaluate_reconstruction,
    model_atoms,
    rank_concepts,
    select_top_concepts,
)
from .artifacts import ArtifactStore, restore_checkpoint
from .config import AppConfig, load_app_config
from .data_phase import (
    extract_dino_activations,
    load_dataset_images,
    preprocess_dino_activations,
)
from .hub_phase import (
    CHECKPOINT_CATALOG,
    HubCheckpointSpec,
    HubDownloader,
    download_hub_checkpoint,
    get_hub_checkpoint_spec,
)
from .logging_utils import create_run_logger, log_event, redact_text
from .model_phase import create_model
from .reproduction import preflight_environment
from .sessions import ExperimentSession, SessionRegistry
from .training_phase import ProgressCallback, train_model
from .types import (
    ConceptRecord,
    DatasetConfig,
    ExperimentStage,
    ExperimentState,
    ModelConfig,
    ModelSource,
    PlotConfig,
    PretrainedRecipe,
    TrainingConfig,
    TrainingEvent,
)
from .visualization_phase import render_concepts


_STAGE_LABELS = {
    ExperimentStage.EMPTY: "Load image data first.",
    ExperimentStage.DATA_LOADED: "Extract DINO activations first.",
    ExperimentStage.ACTIVATIONS_READY: "Center and scale activations first.",
    ExperimentStage.PREPROCESSED: "Initialize a featurizer first.",
}


class ExperimentPipeline:
    """Coordinate one or more isolated experiments through focused phase APIs."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        registry: SessionRegistry | None = None,
        hub_catalog: Mapping[PretrainedRecipe, HubCheckpointSpec] | None = None,
        hub_downloader: HubDownloader | None = None,
    ) -> None:
        """Create the local application service and its server-side registry."""

        self.config = config or load_app_config()
        # ``SessionRegistry`` defines ``__len__`` and an empty injected registry
        # is therefore falsey; an explicit ``None`` check preserves dependency
        # injection for deterministic clocks and tests.
        self.registry = (
            registry
            if registry is not None
            else SessionRegistry(self.config.session_ttl_seconds)
        )
        self._resource_lock = threading.RLock()
        self._stores: dict[str, ArtifactStore] = {}
        self._loggers: dict[str, logging.Logger] = {}
        self._log_paths: dict[str, Path] = {}
        self._hub_catalog = (
            CHECKPOINT_CATALOG
            if hub_catalog is None
            else MappingProxyType(dict(hub_catalog))
        )
        self._hub_downloader = hub_downloader

    def _ensure_resources(self, session_id: str) -> None:
        """Allocate one timestamped artifact directory and logger per session."""

        with self._resource_lock:
            if session_id in self._stores:
                return
            store = ArtifactStore.create(
                self.config.output_dir, f"session-{session_id[:8]}"
            )
            logger, log_path = create_run_logger(
                "workbench", store.run_dir, level=self.config.log_level
            )
            self._stores[session_id] = store
            self._loggers[session_id] = logger
            self._log_paths[session_id] = log_path
            log_event(logger, "session.created", {"session_id": session_id})

    def _session(self, session_id: str) -> ExperimentSession:
        """Return a live session and ensure its non-browser resources exist."""

        session = self.registry.get(session_id)
        self._ensure_resources(session_id)
        return session

    def _logger(self, session_id: str) -> logging.Logger:
        """Return the session's redacting dual-destination logger."""

        self._ensure_resources(session_id)
        return self._loggers[session_id]

    def _store(self, session_id: str) -> ArtifactStore:
        """Return the session's allowlisted generated-artifact directory."""

        self._ensure_resources(session_id)
        return self._stores[session_id]

    @contextmanager
    def _phase(
        self,
        session_id: str,
        name: str,
        configuration: Any | None = None,
        *,
        expected_state: ExperimentState | None = None,
    ) -> Iterator[None]:
        """Log a complete phase lifecycle and retain a sanitized failure message."""

        logger = self._logger(session_id)
        log_event(logger, f"{name}.started", configuration or {})
        try:
            yield
        except Exception as error:
            try:
                session = self.registry.get(session_id)
                with session.locked_state() as state:
                    if expected_state is None or state is expected_state:
                        state.last_error = redact_text(
                            f"{type(error).__name__}: {error}"
                        )
            except KeyError:
                pass
            logger.exception("phase=%s failed", name)
            raise
        else:
            log_event(logger, f"{name}.completed", {})

    @staticmethod
    def _require(value: Any, message: str) -> Any:
        """Return a required state value or raise one actionable workflow error."""

        if value is None:
            raise ValueError(message)
        return value

    def _require_current_state(
        self,
        session_id: str,
        session: ExperimentSession,
        state: ExperimentState,
        expected_state: ExperimentState,
        phase: str,
    ) -> None:
        """Reject a result produced for state replaced by reset, removal, or expiry.

        The identity check treats each ``ExperimentState`` instance as a cheap
        generation token. The registry check additionally rejects work retained
        by a worker after its whole ``ExperimentSession`` was removed or expired.
        Both checks happen immediately before committing compute performed
        outside the session lock.
        """

        try:
            registered_session = self.registry.get(session_id, touch=True)
        except KeyError as error:
            raise RuntimeError(
                f"Session expired or was removed during {phase}; result was discarded."
            ) from error
        if state is not expected_state or registered_session is not session:
            raise RuntimeError(
                f"Session was reset during {phase}; result was discarded."
            )

    @staticmethod
    def _release_model(model: Any | None) -> None:
        """Move a discarded model to CPU so CUDA memory can be reclaimed promptly."""

        if model is not None and callable(getattr(model, "to", None)):
            try:
                model.to("cpu")
            except Exception:
                # A malformed test double or already-torn-down runtime must not
                # prevent reset from releasing every other session reference.
                pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def create_session(self, session_id: str | None = None) -> str:
        """Create server-side state and return only its opaque browser-safe ID."""

        session = self.registry.create(session_id)
        self._ensure_resources(session.session_id)
        with session.locked_state() as state:
            state.artifacts["log"] = self.log_path(session.session_id)
        return session.session_id

    def ensure_session(self, session_id: str | None) -> str:
        """Reuse a live opaque ID or allocate one for a newly loaded browser tab."""

        if session_id:
            # Atomic get-or-create handles concurrent events after a registry TTL
            # race without changing the opaque ID retained by browser state.
            session = self.registry.get_or_create(session_id)
            self._ensure_resources(session_id)
            with session.locked_state() as state:
                state.artifacts.setdefault("log", self.log_path(session_id))
            return session_id
        return self.create_session()

    def snapshot(self, session_id: str) -> ExperimentState:
        """Return a shallow diagnostic snapshot without exposing it through Gradio."""

        with self._session(session_id).locked_state() as state:
            return copy.copy(state)

    def check_environment(self, session_id: str) -> dict[str, Any]:
        """Check assets, CUDA, packages, and gated fixed-backbone access safely."""

        with self._phase(session_id, "environment.check"):
            result = preflight_environment(
                require_cuda=self.config.device.startswith("cuda"),
                check_hf=True,
            )
            report = result.to_dict()
            log_event(self._logger(session_id), "environment.result", report)
            return report

    def load_dataset(self, session_id: str, config: DatasetConfig) -> np.ndarray:
        """Load one validated image source and invalidate all downstream results."""

        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
        with self._phase(
            session_id, "data.load", config, expected_state=expected_state
        ):
            images = load_dataset_images(
                config,
                max_total_bytes=self.config.max_upload_mb * 1024 * 1024,
            )
            with session.locked_state() as state:
                self._require_current_state(
                    session_id, session, state, expected_state, "data loading"
                )
                old_model = state.model
                state.dataset_config = config
                state.images = images
                state.activations = None
                state.preprocessed_activations = None
                state.grid = None
                state.model_config = None
                state.model = None
                state.codes = None
                state.atoms = None
                state.metrics.clear()
                state.concepts.clear()
                state.last_error = None
                state.stage = ExperimentStage.DATA_LOADED
            self._release_model(old_model)
            log_event(self._logger(session_id), "data.loaded", {"shape": images.shape})
            return images

    def extract_activations(self, session_id: str) -> np.ndarray:
        """Extract fixed DINOv3 ViT-B/16 patch activations for loaded images."""

        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
            images = self._require(state.images, "Load image data before extraction.")
            dataset = self._require(
                state.dataset_config, "Dataset configuration is missing."
            )
        with self._phase(
            session_id,
            "activations.extract",
            dataset,
            expected_state=expected_state,
        ):
            activations = extract_dino_activations(
                images,
                device=dataset.device,
                batch_size=dataset.extraction_batch_size,
            )
            with session.locked_state() as state:
                self._require_current_state(
                    session_id, session, state, expected_state, "DINO extraction"
                )
                old_model = state.model
                state.activations = activations
                state.preprocessed_activations = None
                state.grid = None
                state.model_config = None
                state.model = None
                state.codes = None
                state.atoms = None
                state.metrics.clear()
                state.concepts.clear()
                state.stage = ExperimentStage.ACTIVATIONS_READY
            self._release_model(old_model)
            log_event(
                self._logger(session_id),
                "activations.extracted",
                {"shape": activations.shape},
            )
            return activations

    def center_and_scale(self, session_id: str) -> np.ndarray:
        """Apply the exact upstream position-centering and RMS normalization."""

        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
            activations = self._require(
                state.activations, "Extract DINO activations before preprocessing."
            )
        with self._phase(
            session_id,
            "activations.preprocess",
            expected_state=expected_state,
        ):
            matrix, grid = preprocess_dino_activations(activations)
            with session.locked_state() as state:
                self._require_current_state(
                    session_id, session, state, expected_state, "preprocessing"
                )
                old_model = state.model
                state.preprocessed_activations = matrix
                state.grid = grid
                state.model_config = None
                state.model = None
                state.codes = None
                state.atoms = None
                state.metrics.clear()
                state.concepts.clear()
                state.stage = ExperimentStage.PREPROCESSED
            self._release_model(old_model)
            log_event(
                self._logger(session_id),
                "activations.preprocessed",
                {"shape": matrix.shape, "grid": grid},
            )
            return matrix

    def initialize_model(self, session_id: str, config: ModelConfig) -> Any:
        """Construct the selected validated BSF variant for the current feature width."""

        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
            matrix = self._require(
                state.preprocessed_activations,
                "Center and scale activations before model initialization.",
            )
        with self._phase(
            session_id,
            "model.initialize",
            config,
            expected_state=expected_state,
        ):
            model = create_model(config, input_dim=int(matrix.shape[1]))
            try:
                with session.locked_state() as state:
                    self._require_current_state(
                        session_id,
                        session,
                        state,
                        expected_state,
                        "model initialization",
                    )
                    old_model = state.model
                    state.model_config = config
                    state.model = model
                    state.codes = None
                    state.atoms = None
                    state.metrics.clear()
                    state.concepts.clear()
                    state.stage = ExperimentStage.MODEL_READY
            except Exception:
                self._release_model(model)
                raise
            if old_model is not model:
                self._release_model(old_model)
            return model

    def train(
        self,
        session_id: str,
        config: TrainingConfig,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """Train with structured progress and cooperative session cancellation."""

        session = self._session(session_id)
        with session.locked_state() as state:
            if state.stage is ExperimentStage.TRAINING:
                raise ValueError("Training is already in progress for this session.")
            training_state = state
            model = self._require(state.model, "Initialize a model before training.")
            matrix = self._require(
                state.preprocessed_activations,
                "Center and scale activations before training.",
            )
            self._require_current_state(
                session_id, session, state, training_state, "training startup"
            )
            # Obtaining and clearing this exact token while holding the same
            # state lock used by ``cancel_training`` closes the lost-cancel race.
            token = session.cancellation_token()
            token.clear()
            state.metrics.clear()
            state.codes = None
            state.atoms = None
            state.concepts.clear()
            state.last_error = None
            state.stage = ExperimentStage.TRAINING

        def record_progress(event: TrainingEvent) -> None:
            """Persist live metrics, log the full payload, then notify the caller."""

            with session.locked_state() as state:
                if (
                    state is training_state
                    and state.stage is ExperimentStage.TRAINING
                    and state.model is model
                ):
                    state.metrics["cancelled"] = event.cancelled
                    for key, value in (
                        ("loss", event.loss),
                        ("r2", event.r2),
                        ("mean_l0", event.mean_l0),
                        ("dead_groups", event.dead_groups),
                    ):
                        if value is not None:
                            state.metrics[key] = value
            log_event(self._logger(session_id), "training.progress", event)
            if progress_callback is not None:
                progress_callback(event)

        try:
            with self._phase(
                session_id,
                "training",
                config,
                expected_state=training_state,
            ):
                trained_model = train_model(
                    model,
                    matrix,
                    config,
                    progress_callback=record_progress,
                    should_stop=token.is_set,
                )
        except Exception:
            with session.locked_state() as state:
                current_worker = (
                    state is training_state
                    and state.stage is ExperimentStage.TRAINING
                    and state.model is model
                )
                if current_worker:
                    state.stage = ExperimentStage.MODEL_READY
                stale_worker = not current_worker
            if stale_worker:
                self._release_model(model)
            raise
        with session.locked_state() as state:
            stale_worker = not (
                state is training_state
                and state.stage is ExperimentStage.TRAINING
                and state.model is model
            )
            if not stale_worker:
                state.model = trained_model
                state.codes = None
                state.atoms = None
                state.concepts.clear()
                state.stage = ExperimentStage.TRAINED
        if stale_worker:
            self._release_model(trained_model)
            if trained_model is not model:
                self._release_model(model)
        elif trained_model is not model:
            self._release_model(model)
        return trained_model

    def cancel_training(self, session_id: str) -> bool:
        """Signal only an active trainer, synchronized with training startup."""

        try:
            session = self.registry.get(session_id, touch=True)
        except KeyError:
            return False
        with session.locked_state() as state:
            if state.stage is not ExperimentStage.TRAINING:
                return False
            # ``train`` captures and clears the current token under this same
            # lock, so this signal cannot be cleared by a concurrent startup.
            session.request_cancel()
        log_event(self._logger(session_id), "training.cancel_requested", {})
        return True

    def encode(
        self,
        session_id: str,
        *,
        device: str | None = None,
        batch_size: int = 20_000,
    ) -> np.ndarray:
        """Encode every token and retain codes only in locked server memory."""

        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
            model = self._require(
                state.model, "Initialize or load a model before encoding."
            )
            matrix = self._require(
                state.preprocessed_activations,
                "Center and scale activations before encoding.",
            )
        settings = {"device": device or self.config.device, "batch_size": batch_size}
        with self._phase(
            session_id,
            "features.encode",
            settings,
            expected_state=expected_state,
        ):
            codes = encode_features(model, matrix, **settings)
            atoms = model_atoms(model)
            with session.locked_state() as state:
                self._require_current_state(
                    session_id, session, state, expected_state, "feature encoding"
                )
                state.codes = codes
                state.atoms = atoms
            log_event(
                self._logger(session_id), "features.encoded", {"shape": codes.shape}
            )
            return codes

    def evaluate(
        self,
        session_id: str,
        *,
        device: str | None = None,
        batch_size: int = 20_000,
    ) -> dict[str, float | int]:
        """Reconstruct the training matrix and store R²/L0/dead-group metrics."""

        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
            model = self._require(
                state.model, "Initialize or load a model before evaluation."
            )
            matrix = self._require(
                state.preprocessed_activations,
                "Center and scale activations before evaluation.",
            )
        settings = {"device": device or self.config.device, "batch_size": batch_size}
        with self._phase(
            session_id,
            "features.evaluate",
            settings,
            expected_state=expected_state,
        ):
            metrics = evaluate_reconstruction(model, matrix, **settings)
            with session.locked_state() as state:
                self._require_current_state(
                    session_id,
                    session,
                    state,
                    expected_state,
                    "reconstruction evaluation",
                )
                state.metrics.update(metrics)
            log_event(self._logger(session_id), "features.metrics", metrics)
            return dict(metrics)

    def rank(
        self,
        session_id: str,
        *,
        min_firings: int = 0,
        firing_threshold: float = 1e-6,
    ) -> list[ConceptRecord]:
        """Rank every learned group by energy after an optional firing filter."""

        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
            codes = self._require(
                state.codes, "Encode features before ranking concepts."
            )
        settings = {"min_firings": min_firings, "firing_threshold": firing_threshold}
        with self._phase(
            session_id,
            "features.rank",
            settings,
            expected_state=expected_state,
        ):
            concepts = rank_concepts(codes, **settings)
            with session.locked_state() as state:
                self._require_current_state(
                    session_id, session, state, expected_state, "concept ranking"
                )
                state.concepts = concepts
                state.stage = ExperimentStage.ANALYZED
            log_event(
                self._logger(session_id), "features.ranked", {"count": len(concepts)}
            )
            return list(concepts)

    def select_top(self, session_id: str, count: int) -> list[int]:
        """Return group IDs from the current complete searchable ranking."""

        with self._session(session_id).locked_state() as state:
            if not state.concepts:
                raise ValueError("Rank concepts before selecting the top groups.")
            return select_top_concepts(state.concepts, count)

    def visualize(
        self,
        session_id: str,
        selected: Sequence[int],
        config: PlotConfig,
    ) -> tuple[Any, dict[str, Path]]:
        """Render selected concepts and export both PNG and PDF downloads."""

        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
            codes = self._require(state.codes, "Encode features before visualization.")
            atoms = self._require(state.atoms, "Encode features before visualization.")
            images = self._require(state.images, "Load images before visualization.")
            grid = self._require(state.grid, "Center and scale activations first.")
        with self._phase(
            session_id,
            "visualization.render",
            {"selected": list(selected), "plot": config},
            expected_state=expected_state,
        ):
            figure = render_concepts(
                codes, atoms, images, selected, grid=grid, config=config
            )
            paths = self._store(session_id).save_figure(figure)
            with session.locked_state() as state:
                self._require_current_state(
                    session_id, session, state, expected_state, "visualization"
                )
                state.artifacts.update(
                    {f"concepts_{key}": value for key, value in paths.items()}
                )
            log_event(
                self._logger(session_id),
                "visualization.exported",
                {key: str(value) for key, value in paths.items()},
            )
            return figure, paths

    def save_checkpoint(self, session_id: str) -> Path:
        """Save allowlisted model configuration plus a CPU tensor state dictionary."""

        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
            model = self._require(state.model, "Initialize or load a model first.")
            config = self._require(
                state.model_config, "Model configuration is missing."
            )
        with self._phase(
            session_id,
            "checkpoint.save",
            config,
            expected_state=expected_state,
        ):
            path = self._store(session_id).save_checkpoint(model, config)
            with session.locked_state() as state:
                self._require_current_state(
                    session_id, session, state, expected_state, "checkpoint saving"
                )
                state.artifacts["checkpoint"] = path
            return path

    def load_checkpoint(self, session_id: str, path: str | Path) -> ModelConfig:
        """Safely restore a weights-only checkpoint and validate current feature width."""

        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        upload_limit_bytes = self.config.max_upload_mb * 1024 * 1024
        if checkpoint_path.stat().st_size > upload_limit_bytes:
            raise ValueError("Checkpoint exceeds the configured upload limit.")
        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
        with self._phase(
            session_id,
            "checkpoint.load",
            {"name": checkpoint_path.name},
            expected_state=expected_state,
        ):
            model = None
            try:
                model, config = restore_checkpoint(
                    checkpoint_path,
                    max_uncompressed_bytes=upload_limit_bytes,
                )
                with session.locked_state() as state:
                    self._require_current_state(
                        session_id,
                        session,
                        state,
                        expected_state,
                        "checkpoint loading",
                    )
                    matrix = state.preprocessed_activations
                    if matrix is not None and int(matrix.shape[1]) != int(model.d):
                        raise ValueError(
                            "Checkpoint input dimension does not match current "
                            "activations."
                        )
                    old_model = state.model
                    state.model = model
                    state.model_config = config
                    state.codes = None
                    state.atoms = None
                    state.metrics.clear()
                    state.concepts.clear()
                    state.stage = ExperimentStage.MODEL_READY
            except Exception:
                self._release_model(model)
                raise
            self._release_model(old_model)
            return config

    def load_hub_checkpoint(
        self,
        session_id: str,
        recipe: PretrainedRecipe | str,
    ) -> ModelConfig:
        """Download and atomically restore one pinned, integrity-checked model."""

        spec = get_hub_checkpoint_spec(recipe, catalog=self._hub_catalog)
        selected_recipe = PretrainedRecipe(recipe)
        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
            expected_matrix = state.preprocessed_activations
            expected_model = state.model
        phase_config = {
            "recipe": selected_recipe.value,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "filename": spec.filename,
        }
        with self._phase(
            session_id,
            "checkpoint.hugging_face.load",
            phase_config,
            expected_state=expected_state,
        ):
            model = None
            try:
                checkpoint_path = download_hub_checkpoint(
                    spec,
                    downloader=self._hub_downloader,
                    metadata_callback=lambda metadata: log_event(
                        self._logger(session_id),
                        "checkpoint.hugging_face.preflight",
                        metadata,
                    ),
                )
                model, config = restore_checkpoint(
                    checkpoint_path,
                    max_uncompressed_bytes=spec.max_bytes,
                )
                if config != spec.model_config:
                    raise ValueError(
                        "Hub checkpoint does not match its trusted catalog model "
                        "configuration."
                    )
                if int(model.d) != spec.input_dim:
                    raise ValueError(
                        "Hub checkpoint input dimension does not match its trusted "
                        f"catalog entry: {int(model.d)} != {spec.input_dim}."
                    )
                with session.locked_state() as state:
                    self._require_current_state(
                        session_id,
                        session,
                        state,
                        expected_state,
                        "Hugging Face checkpoint loading",
                    )
                    if (
                        state.preprocessed_activations is not expected_matrix
                        or state.model is not expected_model
                    ):
                        raise RuntimeError(
                            "Session changed during Hugging Face checkpoint loading; "
                            "result was discarded."
                        )
                    matrix = state.preprocessed_activations
                    if matrix is not None and int(matrix.shape[1]) != int(model.d):
                        raise ValueError(
                            "Checkpoint input dimension does not match current "
                            "activations."
                        )
                    old_model = state.model
                    state.model = model
                    state.model_config = config
                    state.codes = None
                    state.atoms = None
                    state.metrics.clear()
                    state.concepts.clear()
                    state.stage = ExperimentStage.MODEL_READY
            except Exception:
                self._release_model(model)
                raise
            self._release_model(old_model)
            return config

    def export_arrays(self, session_id: str) -> Path:
        """Export every available numeric array as one safe compressed NPZ."""

        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
            arrays = {
                name: value
                for name, value in (
                    ("images", state.images),
                    ("activations", state.activations),
                    ("preprocessed_activations", state.preprocessed_activations),
                    ("codes", state.codes),
                    ("atoms", state.atoms),
                )
                if value is not None
            }
        with self._phase(
            session_id,
            "artifacts.export_arrays",
            {"names": list(arrays)},
            expected_state=expected_state,
        ):
            path = self._store(session_id).save_arrays(arrays)
            with session.locked_state() as state:
                self._require_current_state(
                    session_id, session, state, expected_state, "array export"
                )
                state.artifacts["arrays"] = path
            return path

    def export_results(self, session_id: str) -> Path:
        """Bundle generated artifacts with complete serializable result metadata."""

        session = self._session(session_id)
        with session.locked_state() as state:
            expected_state = state
            result = {
                "session_id": state.session_id,
                "stage": state.stage,
                "dataset_config": state.dataset_config,
                "model_config": state.model_config,
                "metrics": dict(state.metrics),
                "concepts": list(state.concepts),
            }
        with self._phase(
            session_id,
            "artifacts.export_results",
            expected_state=expected_state,
        ):
            path = self._store(session_id).save_result_bundle(result)
            with session.locked_state() as state:
                self._require_current_state(
                    session_id, session, state, expected_state, "result export"
                )
                state.artifacts["results"] = path
            return path

    def log_path(self, session_id: str) -> Path:
        """Return the complete sanitized log inside the generated-artifact root."""

        self._ensure_resources(session_id)
        return self._log_paths[session_id]

    def read_log(self, session_id: str, *, tail_lines: int = 200) -> str:
        """Read a bounded UI tail while the downloadable file retains every line."""

        if tail_lines <= 0:
            raise ValueError("tail_lines must be greater than zero")
        # Polling the live log is user activity and therefore extends the same
        # registry TTL as every other UI action.
        self.registry.touch(session_id)
        lines = self.log_path(session_id).read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-tail_lines:])

    def run_current_pipeline(
        self,
        session_id: str,
        dataset: DatasetConfig,
        model: ModelConfig,
        training: TrainingConfig,
        *,
        model_source: ModelSource | str = ModelSource.TRAIN,
        pretrained_recipe: PretrainedRecipe | str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[ConceptRecord]:
        """Run the readable end-to-end phase sequence used by the UI preset action."""

        try:
            source = ModelSource(model_source)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unknown model source: {model_source!r}") from error
        selected_recipe = None
        if source is ModelSource.HUGGING_FACE:
            if pretrained_recipe is None:
                raise ValueError(
                    "Select a pretrained recipe when using the Hugging Face source."
                )
            try:
                selected_recipe = PretrainedRecipe(pretrained_recipe)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Unknown pretrained recipe: {pretrained_recipe!r}"
                ) from error

        self.load_dataset(session_id, dataset)
        self.extract_activations(session_id)
        self.center_and_scale(session_id)
        if source is ModelSource.TRAIN:
            self.initialize_model(session_id, model)
            self.train(session_id, training, progress_callback=progress_callback)
        else:
            # The validation above makes this non-optional in the Hub branch.
            assert selected_recipe is not None
            self.load_hub_checkpoint(session_id, selected_recipe)
        self.encode(session_id, device=training.device)
        self.evaluate(session_id, device=training.device)
        return self.rank(session_id)

    def reset_session(self, session_id: str) -> None:
        """Cancel work and release large in-memory values without deleting artifacts."""

        session = self._session(session_id)
        with session.locked_state() as state:
            old_model = state.model
            training_active = state.stage is ExperimentStage.TRAINING
        log_event(self._logger(session_id), "session.reset", {})
        self.registry.reset(session_id)
        with self.registry.get(session_id).locked_state() as state:
            state.artifacts["log"] = self.log_path(session_id)
        if not training_active:
            self._release_model(old_model)

    @staticmethod
    def _close_logger(logger: logging.Logger) -> None:
        """Flush and detach every handler so Windows and POSIX release the file."""

        for handler in tuple(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

    def remove_session(self, session_id: str) -> None:
        """Delete browser-owned server state when Gradio expires its opaque ID."""

        try:
            session = self.registry.get(session_id, touch=False)
        except KeyError:
            session = None

        model = None
        if session is not None:
            with session.locked_state() as state:
                try:
                    still_registered = (
                        self.registry.get(session_id, touch=False) is session
                    )
                except KeyError:
                    still_registered = False
                if still_registered and state.stage is ExperimentStage.TRAINING:
                    # Gradio may unload browser state while its training callback
                    # still owns the model. Signal it, but leave model ownership
                    # and logger cleanup to the worker and subsequent TTL sweep.
                    session.request_cancel()
                    log_event(
                        self._logger(session_id),
                        "session.removal_deferred",
                        {"reason": "training_active"},
                    )
                    return
                if still_registered and self.registry.remove(session_id):
                    model = state.model

        self._release_model(model)
        with self._resource_lock:
            logger = self._loggers.pop(session_id, None)
            self._stores.pop(session_id, None)
            self._log_paths.pop(session_id, None)
        if logger is not None:
            self._close_logger(logger)

    def cleanup_expired(self) -> tuple[str, ...]:
        """Remove stale registry sessions and close their file handlers."""

        expired = set(self.registry.cleanup_expired())
        live_ids = set(self.registry.ids())
        with self._resource_lock:
            # Registry access methods may opportunistically expire entries before
            # this timer runs, so reconcile resources against the live ID set.
            expired.update(set(self._stores) - live_ids)
        for session_id in expired:
            with self._resource_lock:
                logger = self._loggers.pop(session_id, None)
                self._stores.pop(session_id, None)
                self._log_paths.pop(session_id, None)
            if logger is not None:
                self._close_logger(logger)
        if expired:
            self._release_model(None)
        return tuple(sorted(expired))

    def close(self) -> None:
        """Cancel all sessions and close every logger owned by this pipeline."""

        self.registry.close()
        with self._resource_lock:
            loggers = tuple(self._loggers.values())
            self._loggers.clear()
            self._stores.clear()
            self._log_paths.clear()
        for logger in loggers:
            self._close_logger(logger)
        self._release_model(None)


__all__ = ["ExperimentPipeline"]
