"""Local Gradio 6 workbench for the fixed DINOv3 block-sparse workflow.

Global context
--------------
Gradio components hold only scalar controls, artifact paths, and an opaque
session ID. Models and arrays remain in :class:`ExperimentPipeline`'s locked
server registry, which owns TTL expiry. A Gradio load callback creates each
browser's unique ID and its delete callback releases closed-page state; see
https://www.gradio.app/docs/gradio/state.
"""

from __future__ import annotations

import math
from pathlib import Path
from queue import Empty, Queue
import threading
from typing import Any, Iterator, NoReturn

from .config import AppConfig, load_app_config
from .logging_utils import redact_text
from .pipeline import ExperimentPipeline
from .presets import get_preset
from .types import (
    ConceptRecord,
    DatasetConfig,
    DatasetKind,
    FeaturizerKind,
    ModelConfig,
    PlotConfig,
    TrainingConfig,
    TrainingEvent,
)


FIXED_DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
CONCEPT_HEADERS = ["rank", "group_id", "firing_count", "firing_rate", "energy"]
GPU_EVENT_OPTIONS = {"concurrency_id": "gpu", "concurrency_limit": 1}
_WORKER_POLL_SECONDS = 0.2


def normalize_upload_paths(value: Any) -> tuple[Path, ...]:
    """Normalize Gradio ``File`` strings/FileData values into local paths."""

    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple)) else (value,)
    paths: list[Path] = []
    for item in values:
        raw_path = getattr(item, "path", item)
        if not isinstance(raw_path, (str, Path)):
            raise ValueError("Uploaded files did not provide readable server paths.")
        paths.append(Path(raw_path))
    return tuple(paths)


def _integer(value: Any, label: str) -> int:
    """Convert a Gradio numeric value while rejecting fractional integers."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be an integer.")
    parsed = int(value)
    if not math.isfinite(float(value)) or parsed != value:
        raise ValueError(f"{label} must be an integer.")
    return parsed


def _model_config(
    kind: str,
    n_groups: Any,
    group_size: Any,
    l0: Any,
    coef: Any,
    target_l0: Any,
    gain: Any,
    paper_version: bool,
) -> ModelConfig:
    """Build the typed model contract shared by the UI and phase factory."""

    return ModelConfig(
        kind=FeaturizerKind(kind),
        n_groups=_integer(n_groups, "Group count"),
        group_size=_integer(group_size, "Group size"),
        l0=_integer(l0, "L0"),
        coef=float(coef),
        target_l0=_integer(target_l0, "Target L0"),
        gain=float(gain),
        paper_version=bool(paper_version),
    )


def _training_config(
    epochs: Any,
    lr: Any,
    batch_size: Any,
    snr: Any,
    log_every: Any,
    seed: Any,
    device: str,
) -> TrainingConfig:
    """Build a validated-on-use training contract from scalar controls."""

    return TrainingConfig(
        epochs=_integer(epochs, "Epochs"),
        lr=float(lr),
        batch_size=_integer(batch_size, "Batch size"),
        snr=float(snr),
        log_every=_integer(log_every, "Log interval"),
        seed=_integer(seed, "Seed"),
        device=str(device),
    )


def _plot_config(
    n_img: Any,
    ncol_img: Any,
    clip: Any,
    saturation: Any,
    drop_low_norm: Any,
    max_points: Any,
    point_size: Any,
    concept_gap: Any,
) -> PlotConfig:
    """Build only the effective upstream ``plot_concepts`` controls."""

    return PlotConfig(
        n_img=_integer(n_img, "Images per concept"),
        ncol_img=_integer(ncol_img, "Image columns"),
        clip=float(clip),
        saturation=float(saturation),
        drop_low_norm=float(drop_low_norm),
        max_points=_integer(max_points, "Maximum points"),
        point_size=float(point_size),
        concept_gap=float(concept_gap),
    )


def preset_control_values(name: str) -> tuple[Any, ...]:
    """Return every model/training control value for one exact upstream preset."""

    preset = get_preset(name)
    model = preset.model
    training = preset.training
    return (
        model.kind.value,
        model.n_groups,
        model.group_size,
        model.l0,
        model.coef,
        model.target_l0,
        model.gain,
        model.paper_version,
        training.epochs,
        training.lr,
        training.batch_size,
        training.snr,
        training.log_every,
        training.seed,
        training.device,
    )


def format_training_event(event: TrainingEvent) -> str:
    """Format every requested live training measurement without hiding missing data."""

    def metric(label: str, value: float | int | None, precision: int) -> str:
        """Render one optional metric consistently in the compact status line."""

        if value is None:
            return f"{label}=—"
        return f"{label}={float(value):.{precision}f}"

    fields = [
        f"epoch {event.epoch}/{event.total_epochs}",
        metric("loss", event.loss, 5),
        metric("R²", event.r2, 4),
        metric("L0", event.mean_l0, 3),
        f"dead groups={event.dead_groups if event.dead_groups is not None else '—'}",
    ]
    if event.cancelled:
        fields.append("cancelled")
    if event.message:
        fields.append(event.message)
    return " · ".join(fields)


def safe_ui_error_message(error: Exception) -> str:
    """Redact credentials before an exception is returned to the browser."""

    return redact_text(str(error))


def ui_error(error: Exception) -> NoReturn:
    """Raise a browser-safe Gradio error without printing or chaining the cause.

    ``print_exception=False`` is Gradio's supported switch for suppressing the
    backend exception print, while ``from None`` suppresses Python exception
    chaining so a credential-bearing lower-level message cannot reach stderr.
    See https://www.gradio.app/docs/gradio/error.
    """

    import gradio as gr

    raise gr.Error(
        safe_ui_error_message(error),
        print_exception=False,
    ) from None


def _concept_rows(records: list[ConceptRecord]) -> list[list[int | float]]:
    """Convert immutable ranking records into a Gradio Dataframe value."""

    return [list(record.as_dict().values()) for record in records]


def _metrics_tuple(metrics: dict[str, float | int]) -> tuple[Any, Any, Any]:
    """Extract the three live metric component values in a stable order."""

    return metrics.get("r2"), metrics.get("mean_l0"), metrics.get("dead_groups")


class WorkbenchController:
    """Framework-light event adapter around :class:`ExperimentPipeline`."""

    def __init__(self, pipeline: ExperimentPipeline) -> None:
        """Attach one process-wide pipeline used by all isolated browser sessions."""

        self.pipeline = pipeline

    def _touch_session(self, session_id: str) -> None:
        """Keep registry-owned state alive while its background worker is active."""

        try:
            self.pipeline.registry.touch(session_id)
        except KeyError:
            # Teardown still must join a worker if an external reset/remove won
            # the race; recreating an expired state here would be incorrect.
            pass

    def _finish_worker(
        self,
        thread: threading.Thread,
        session_id: str,
        *,
        worker_reported_done: bool,
    ) -> None:
        """Cancel an abandoned worker and join it before releasing the UI event.

        Timed joins are used only to refresh the registry TTL between waits. The
        final unbounded join guarantees the worker cannot outlive Gradio's shared
        GPU concurrency slot.
        """

        if thread.is_alive() and not worker_reported_done:
            self.pipeline.cancel_training(session_id)
        while thread.is_alive():
            self._touch_session(session_id)
            thread.join(timeout=_WORKER_POLL_SECONDS)
        thread.join()

    def training_stream(
        self,
        session_id: str,
        config: TrainingConfig,
    ) -> Iterator[tuple[str, Any, Any, Any, str]]:
        """Train in a worker and stream callback events while cancellation stays live.

        A generator is Gradio's documented streaming-output mechanism:
        https://www.gradio.app/guides/streaming-outputs.
        """

        messages: Queue[TrainingEvent | Exception | None] = Queue()

        def worker() -> None:
            """Run the blocking upstream trainer and forward terminal state."""

            try:
                self.pipeline.train(
                    session_id,
                    config,
                    progress_callback=messages.put,
                )
            except Exception as error:
                messages.put(error)
            finally:
                messages.put(None)

        thread = threading.Thread(
            target=worker,
            name=f"bsf-training-{session_id[:8]}",
            daemon=True,
        )
        thread.start()
        worker_reported_done = False
        try:
            while not worker_reported_done:
                self._touch_session(session_id)
                try:
                    message = messages.get(timeout=_WORKER_POLL_SECONDS)
                except Empty:
                    continue
                if message is None:
                    worker_reported_done = True
                    continue
                if isinstance(message, Exception):
                    raise message
                yield (
                    format_training_event(message),
                    message.r2,
                    message.mean_l0,
                    message.dead_groups,
                    self.pipeline.read_log(session_id),
                )
        finally:
            self._finish_worker(
                thread,
                session_id,
                worker_reported_done=worker_reported_done,
            )
        state = self.pipeline.snapshot(session_id)
        r2, mean_l0, dead_groups = _metrics_tuple(state.metrics)
        yield (
            "Training finished."
            if not state.metrics.get("cancelled")
            else "Training stopped.",
            r2,
            mean_l0,
            dead_groups,
            self.pipeline.read_log(session_id),
        )

    def pipeline_stream(
        self,
        session_id: str,
        model: ModelConfig,
        training: TrainingConfig,
        *,
        extraction_batch_size: int,
    ) -> Iterator[tuple[str, Any, Any, Any, list[list[Any]], Any, str]]:
        """Run the current controls end-to-end and stream phase/training progress."""

        messages: Queue[tuple[str, Any] | Exception | None] = Queue()

        def progress(event: TrainingEvent) -> None:
            """Forward structured trainer events to the UI generator."""

            messages.put(("training", event))

        def worker() -> None:
            """Execute missing data phases, then the chosen model workflow."""

            try:
                state = self.pipeline.snapshot(session_id)
                if state.images is None:
                    messages.put(("status", "Loading bundled rabbits…"))
                    self.pipeline.load_dataset(
                        session_id,
                        DatasetConfig(
                            DatasetKind.BUNDLED_RABBITS,
                            extraction_batch_size=extraction_batch_size,
                            device=training.device,
                        ),
                    )
                state = self.pipeline.snapshot(session_id)
                if state.activations is None:
                    messages.put(("status", "Extracting fixed DINOv3 activations…"))
                    self.pipeline.extract_activations(session_id)
                state = self.pipeline.snapshot(session_id)
                if state.preprocessed_activations is None:
                    messages.put(("status", "Centering and scaling activations…"))
                    self.pipeline.center_and_scale(session_id)
                messages.put(("status", "Initializing featurizer…"))
                self.pipeline.initialize_model(session_id, model)
                messages.put(("status", "Training…"))
                self.pipeline.train(
                    session_id,
                    training,
                    progress_callback=progress,
                )
                if self.pipeline.snapshot(session_id).metrics.get("cancelled"):
                    messages.put(
                        ("status", "Pipeline stopped after training cancellation.")
                    )
                    return
                messages.put(("status", "Encoding and evaluating…"))
                self.pipeline.encode(session_id, device=training.device)
                self.pipeline.evaluate(session_id, device=training.device)
                records = self.pipeline.rank(session_id)
                messages.put(("done", records))
            except Exception as error:
                messages.put(error)
            finally:
                messages.put(None)

        thread = threading.Thread(
            target=worker,
            name=f"bsf-pipeline-{session_id[:8]}",
            daemon=True,
        )
        thread.start()
        worker_reported_done = False
        try:
            while not worker_reported_done:
                self._touch_session(session_id)
                try:
                    message = messages.get(timeout=_WORKER_POLL_SECONDS)
                except Empty:
                    continue
                if message is None:
                    worker_reported_done = True
                    continue
                if isinstance(message, Exception):
                    raise message
                kind, payload = message
                state = self.pipeline.snapshot(session_id)
                r2, mean_l0, dead_groups = _metrics_tuple(state.metrics)
                rows = _concept_rows(state.concepts)
                selected = [
                    record.group_id for record in state.concepts[: min(3, len(rows))]
                ]
                if kind == "training":
                    status = format_training_event(payload)
                elif kind == "done":
                    rows = _concept_rows(payload)
                    selected = [
                        record.group_id for record in payload[: min(3, len(payload))]
                    ]
                    status = f"Pipeline complete: ranked {len(payload)} concepts."
                else:
                    status = str(payload)
                yield (
                    status,
                    r2,
                    mean_l0,
                    dead_groups,
                    rows,
                    selected,
                    self.pipeline.read_log(session_id),
                )
        finally:
            self._finish_worker(
                thread,
                session_id,
                worker_reported_done=worker_reported_done,
            )


def build_app(
    config: AppConfig | None = None,
    *,
    pipeline: ExperimentPipeline | None = None,
) -> Any:
    """Build and return the local Gradio Blocks app without launching a server."""

    # The presentation dependency stays lazy so reproduction CLI help remains
    # usable in minimal/headless environments.
    import gradio as gr

    settings = config or load_app_config()
    service = pipeline or ExperimentPipeline(settings)
    controller = WorkbenchController(service)
    cache_interval = max(60, min(settings.session_ttl_seconds, 3_600))

    def session_id(value: str | None) -> str:
        """Resolve a stale/empty browser ID without storing any server object."""

        return service.ensure_session(value)

    def log_tail(identifier: str) -> str:
        """Return the readable UI tail of the complete downloadable log."""

        return service.read_log(identifier)

    with gr.Blocks(
        title="Block-Sparse Featurizer Workbench",
        fill_width=True,
        delete_cache=(cache_interval, settings.session_ttl_seconds),
        analytics_enabled=False,
    ) as demo:

        def remove_browser_session(value: str) -> None:
            """Release only a real per-load ID when Gradio deletes browser state."""

            if value:
                service.remove_session(value)

        browser_session = gr.State(
            value="",
            delete_callback=remove_browser_session,
        )

        def create_browser_session() -> tuple[str, Path]:
            """Allocate one registry entry and its immediately downloadable log."""

            identifier = service.create_session()
            return identifier, service.log_path(identifier)

        gr.Markdown(
            "# Block-Sparse Featurizer Workbench\n"
            f"Fixed backbone: `{FIXED_DINO_MODEL}`. Models and arrays stay server-side."
        )
        status = gr.Textbox(
            value="Choose a preset or load data to begin.",
            label="Status",
            interactive=False,
        )

        with gr.Tab("Presets"):
            gr.Markdown(
                "Presets update controls; **Run Current Pipeline** executes them."
            )
            with gr.Row():
                preset_readme = gr.Button("README Quickstart")
                preset_grassmannian = gr.Button("Grassmannian Notebook")
                preset_group_lasso = gr.Button("Group Lasso Notebook")
                preset_vanilla = gr.Button("Vanilla Notebook")
                run_pipeline = gr.Button("Run Current Pipeline", variant="primary")

        with gr.Tab("Data"):
            with gr.Row():
                check_environment = gr.Button("Check Environment")
                load_rabbits = gr.Button("Load Rabbits")
                extract_activations = gr.Button(
                    "Extract DINO Activations", variant="primary"
                )
                center_scale = gr.Button("Center & Scale")
            environment_report = gr.JSON(label="Environment report")
            with gr.Row():
                npz_upload = gr.File(
                    label="NPZ with nonempty arr_0 RGB uint8 array",
                    file_types=[".npz"],
                    type="filepath",
                )
                load_npz = gr.Button("Load NPZ")
            with gr.Row():
                image_uploads = gr.File(
                    label="PNG/JPEG/WebP images (same dimensions)",
                    file_count="multiple",
                    file_types=[".png", ".jpg", ".jpeg", ".webp"],
                    type="filepath",
                )
                load_images = gr.Button("Load Uploaded Images")
            extraction_batch_size = gr.Number(
                value=64,
                precision=0,
                minimum=1,
                label="DINO extraction batch size",
            )

        with gr.Tab("Model"):
            gr.Markdown("Choose one of the three BSF variants supported upstream.")
            with gr.Row():
                featurizer = gr.Dropdown(
                    choices=[kind.value for kind in FeaturizerKind],
                    value=FeaturizerKind.GRASSMANNIAN.value,
                    label="Featurizer",
                )
                n_groups = gr.Number(
                    value=256, precision=0, minimum=1, label="Group count"
                )
                group_size = gr.Number(
                    value=3, precision=0, minimum=1, label="Group size"
                )
                l0 = gr.Number(value=16, precision=0, minimum=1, label="L0")
            with gr.Row():
                coef = gr.Number(value=1e-2, minimum=0, label="Group Lasso coefficient")
                target_l0 = gr.Number(
                    value=16, precision=0, minimum=1, label="Target L0"
                )
                gain = gr.Number(value=10.0, minimum=0, label="Target controller gain")
                paper_version = gr.Checkbox(value=False, label="Paper version")
                initialize_model = gr.Button("Initialize Model", variant="primary")

        with gr.Tab("Training"):
            with gr.Row():
                epochs = gr.Number(value=60, precision=0, minimum=1, label="Epochs")
                learning_rate = gr.Number(value=4e-4, minimum=0, label="Learning rate")
                training_batch_size = gr.Number(
                    value=2048, precision=0, minimum=1, label="Batch size"
                )
                snr = gr.Number(value=0.1, minimum=0, label="SNR")
            with gr.Row():
                log_every = gr.Number(
                    value=5, precision=0, minimum=1, label="Log interval"
                )
                seed = gr.Number(value=0, precision=0, minimum=0, label="Seed")
                device = gr.Dropdown(
                    choices=["auto", "cpu", "cuda"],
                    value=settings.device,
                    allow_custom_value=True,
                    label="Device (cuda:N is accepted)",
                )
                train = gr.Button("Train", variant="primary")
                stop = gr.Button("Stop Training", variant="stop")
            with gr.Row():
                live_r2 = gr.Number(label="Live R²", interactive=False)
                live_l0 = gr.Number(label="Live L0", interactive=False)
                dead_groups = gr.Number(
                    label="Dead groups", precision=0, interactive=False
                )

        with gr.Tab("Features"):
            with gr.Row():
                encode = gr.Button("Encode Features")
                evaluate = gr.Button("Reconstruct & Evaluate")
                rank = gr.Button("Rank Concepts")
                top_n = gr.Number(value=3, precision=0, minimum=1, label="Top N")
                select_top = gr.Button("Select Top N")
            concept_table = gr.Dataframe(
                headers=CONCEPT_HEADERS,
                datatype=["number"] * len(CONCEPT_HEADERS),
                value=[],
                type="array",
                interactive=False,
                show_search="filter",
                label="Every learned group",
            )

        with gr.Tab("Visualization"):
            concepts = gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                filterable=True,
                label="Concept group IDs",
            )
            with gr.Row():
                n_img = gr.Number(
                    value=10, precision=0, minimum=1, label="Images per concept"
                )
                ncol_img = gr.Number(
                    value=5, precision=0, minimum=1, label="Image columns"
                )
                clip = gr.Slider(0.1, 100, value=98, label="Overlay clip percentile")
                saturation = gr.Number(value=1.0, minimum=0, label="Overlay saturation")
            with gr.Row():
                drop_low_norm = gr.Slider(
                    0, 0.99, value=0, step=0.01, label="Drop low-norm fraction"
                )
                max_points = gr.Number(
                    value=5000, precision=0, minimum=8, label="Max points"
                )
                point_size = gr.Number(value=4.0, minimum=0.01, label="Point size")
                concept_gap = gr.Number(value=0.6, minimum=0, label="Concept gap")
                render = gr.Button("Render Concept Plot", variant="primary")
            concept_plot = gr.Plot(label="Concept manifolds and source overlays")
            with gr.Row():
                png_download = gr.DownloadButton("Download PNG")
                pdf_download = gr.DownloadButton("Download PDF")

        with gr.Tab("Artifacts"):
            with gr.Row():
                save_checkpoint = gr.Button("Save Checkpoint")
                checkpoint_download = gr.DownloadButton("Download Checkpoint")
            with gr.Row():
                checkpoint_upload = gr.File(
                    label="Load weights-only checkpoint",
                    file_types=[".pt"],
                    type="filepath",
                )
                load_checkpoint = gr.Button("Load Checkpoint")
            with gr.Row():
                export_results = gr.Button("Export Results Bundle")
                result_download = gr.DownloadButton("Download Results Bundle")
                export_arrays = gr.Button("Export Arrays")
                arrays_download = gr.DownloadButton("Download Arrays")
                full_log_download = gr.DownloadButton("Download Full Log")
                reset = gr.Button("Reset Session", variant="stop")

        full_log = gr.Textbox(
            label="Sanitized live log (latest 200 lines; download retains the full log)",
            lines=14,
            max_lines=20,
            interactive=False,
            autoscroll=True,
        )

        demo.load(
            create_browser_session,
            inputs=[],
            outputs=[browser_session, full_log_download],
            queue=False,
            api_visibility="private",
        )

        model_inputs = [
            featurizer,
            n_groups,
            group_size,
            l0,
            coef,
            target_l0,
            gain,
            paper_version,
        ]
        training_inputs = [
            epochs,
            learning_rate,
            training_batch_size,
            snr,
            log_every,
            seed,
            device,
        ]
        preset_outputs = [*model_inputs, *training_inputs]

        for button, name in (
            (preset_readme, "readme"),
            (preset_grassmannian, "grassmannian_notebook"),
            (preset_group_lasso, "group_lasso_notebook"),
            (preset_vanilla, "vanilla_notebook"),
        ):
            button.click(
                fn=lambda preset_name=name: preset_control_values(preset_name),
                inputs=[],
                outputs=preset_outputs,
                queue=False,
                api_visibility="private",
            )

        def check_environment_action(
            raw_session: str | None,
        ) -> tuple[dict[str, Any], str, str]:
            """Return the safe structured preflight plus status and log."""

            identifier = session_id(raw_session)
            try:
                report = service.check_environment(identifier)
            except Exception as error:
                ui_error(error)
            return (
                report,
                ("Environment ready." if report["ok"] else "Environment has errors."),
                log_tail(identifier),
            )

        check_environment.click(
            check_environment_action,
            inputs=browser_session,
            outputs=[environment_report, status, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def load_action(
            raw_session: str | None,
            kind: DatasetKind,
            paths: tuple[Path, ...],
            batch_value: Any,
            device_value: str,
        ) -> tuple[str, str]:
            """Load any supported data source through one reusable controller."""

            identifier = session_id(raw_session)
            try:
                images = service.load_dataset(
                    identifier,
                    DatasetConfig(
                        kind=kind,
                        paths=paths,
                        extraction_batch_size=_integer(
                            batch_value, "Extraction batch size"
                        ),
                        device=device_value,
                    ),
                )
            except Exception as error:
                ui_error(error)
            return (
                f"Loaded {len(images)} RGB images with shape {tuple(images.shape[1:])}.",
                log_tail(identifier),
            )

        load_rabbits.click(
            lambda sid, batch, dev: load_action(
                sid, DatasetKind.BUNDLED_RABBITS, (), batch, dev
            ),
            inputs=[browser_session, extraction_batch_size, device],
            outputs=[status, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def load_npz_action(
            raw_session: str | None, upload: Any, batch: Any, device_value: str
        ) -> tuple[str, str]:
            """Require exactly one NPZ upload before delegating to shared loading."""

            paths = normalize_upload_paths(upload)
            if len(paths) != 1:
                ui_error(ValueError("Choose exactly one NPZ archive before loading."))
            return load_action(raw_session, DatasetKind.NPZ, paths, batch, device_value)

        load_npz.click(
            load_npz_action,
            inputs=[browser_session, npz_upload, extraction_batch_size, device],
            outputs=[status, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def load_images_action(
            raw_session: str | None, uploads: Any, batch: Any, device_value: str
        ) -> tuple[str, str]:
            """Require at least one loose image before shared ingestion."""

            paths = normalize_upload_paths(uploads)
            if not paths:
                ui_error(ValueError("Choose at least one PNG, JPEG, or WebP image."))
            return load_action(
                raw_session, DatasetKind.UPLOADED_IMAGES, paths, batch, device_value
            )

        load_images.click(
            load_images_action,
            inputs=[browser_session, image_uploads, extraction_batch_size, device],
            outputs=[status, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def simple_phase(
            raw_session: str | None, method: Any, success: str
        ) -> tuple[str, str]:
            """Run a no-configuration phase with consistent UI error handling."""

            identifier = session_id(raw_session)
            try:
                result = method(identifier)
            except Exception as error:
                ui_error(error)
            detail = tuple(result.shape) if hasattr(result, "shape") else "done"
            return f"{success}: {detail}.", log_tail(identifier)

        extract_activations.click(
            lambda sid: simple_phase(
                sid, service.extract_activations, "DINO activations ready"
            ),
            inputs=browser_session,
            outputs=[status, full_log],
            **GPU_EVENT_OPTIONS,
        )
        center_scale.click(
            lambda sid: simple_phase(
                sid, service.center_and_scale, "Normalized tokens ready"
            ),
            inputs=browser_session,
            outputs=[status, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def initialize_action(raw_session: str | None, *values: Any) -> tuple[str, str]:
            """Build one selected BSF variant from the visible model controls."""

            identifier = session_id(raw_session)
            try:
                config_value = _model_config(*values)
                service.initialize_model(identifier, config_value)
            except Exception as error:
                ui_error(error)
            return f"Initialized {config_value.kind.value} featurizer.", log_tail(
                identifier
            )

        initialize_model.click(
            initialize_action,
            inputs=[browser_session, *model_inputs],
            outputs=[status, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def train_action(
            raw_session: str | None, *values: Any
        ) -> Iterator[tuple[Any, ...]]:
            """Stream training progress or expose one actionable domain error."""

            identifier = session_id(raw_session)
            try:
                config_value = _training_config(*values)
                yield from controller.training_stream(identifier, config_value)
            except Exception as error:
                ui_error(error)

        train.click(
            train_action,
            inputs=[browser_session, *training_inputs],
            outputs=[status, live_r2, live_l0, dead_groups, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def stop_action(raw_session: str | None) -> str:
            """Set the cooperative cancellation flag without entering the GPU queue."""

            identifier = session_id(raw_session)
            return (
                "Cancellation requested; training will stop between mini-batches."
                if service.cancel_training(identifier)
                else "No live training session was found."
            )

        stop.click(
            stop_action,
            inputs=browser_session,
            outputs=status,
            queue=False,
            api_visibility="private",
        )

        def encode_action(
            raw_session: str | None, device_value: str
        ) -> tuple[str, str]:
            """Encode every token and report the server-side code shape."""

            identifier = session_id(raw_session)
            try:
                codes = service.encode(identifier, device=device_value)
            except Exception as error:
                ui_error(error)
            return f"Encoded feature blocks with shape {tuple(codes.shape)}.", log_tail(
                identifier
            )

        encode.click(
            encode_action,
            inputs=[browser_session, device],
            outputs=[status, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def evaluate_action(
            raw_session: str | None, device_value: str
        ) -> tuple[Any, ...]:
            """Evaluate reconstruction and populate all metric displays."""

            identifier = session_id(raw_session)
            try:
                metrics = service.evaluate(identifier, device=device_value)
            except Exception as error:
                ui_error(error)
            r2, mean_l0, dead = _metrics_tuple(metrics)
            return (
                "Reconstruction metrics updated.",
                r2,
                mean_l0,
                dead,
                log_tail(identifier),
            )

        evaluate.click(
            evaluate_action,
            inputs=[browser_session, device],
            outputs=[status, live_r2, live_l0, dead_groups, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def rank_action(raw_session: str | None) -> tuple[Any, ...]:
            """Rank every group and update both searchable table and selector choices."""

            identifier = session_id(raw_session)
            try:
                records = service.rank(identifier)
            except Exception as error:
                ui_error(error)
            choices = [record.group_id for record in records]
            return (
                f"Ranked all {len(records)} learned groups.",
                _concept_rows(records),
                gr.update(choices=choices, value=[]),
                log_tail(identifier),
            )

        rank.click(
            rank_action,
            inputs=browser_session,
            outputs=[status, concept_table, concepts, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def select_action(raw_session: str | None, count: Any) -> tuple[Any, str]:
            """Select the requested leading concept IDs from the current ranking."""

            identifier = session_id(raw_session)
            try:
                selected = service.select_top(identifier, _integer(count, "Top N"))
            except Exception as error:
                ui_error(error)
            return gr.update(value=selected), f"Selected {len(selected)} top concepts."

        select_top.click(
            select_action,
            inputs=[browser_session, top_n],
            outputs=[concepts, status],
            **GPU_EVENT_OPTIONS,
        )

        plot_inputs = [
            n_img,
            ncol_img,
            clip,
            saturation,
            drop_low_norm,
            max_points,
            point_size,
            concept_gap,
        ]

        def render_action(
            raw_session: str | None, selected: list[Any] | None, *values: Any
        ) -> tuple[Any, Path, Path, str, str]:
            """Validate selected concept IDs, render, and expose two file formats."""

            identifier = session_id(raw_session)
            try:
                group_ids = [
                    _integer(value, "Concept ID") for value in (selected or [])
                ]
                figure, paths = service.visualize(
                    identifier,
                    group_ids,
                    _plot_config(*values),
                )
            except Exception as error:
                ui_error(error)
            return (
                figure,
                paths["png"],
                paths["pdf"],
                "Concept plot rendered.",
                log_tail(identifier),
            )

        render.click(
            render_action,
            inputs=[browser_session, concepts, *plot_inputs],
            outputs=[concept_plot, png_download, pdf_download, status, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def save_checkpoint_action(raw_session: str | None) -> tuple[Path, str, str]:
            """Save and expose a safe state-dict-only checkpoint."""

            identifier = session_id(raw_session)
            try:
                path = service.save_checkpoint(identifier)
            except Exception as error:
                ui_error(error)
            return path, "Checkpoint saved.", log_tail(identifier)

        save_checkpoint.click(
            save_checkpoint_action,
            inputs=browser_session,
            outputs=[checkpoint_download, status, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def load_checkpoint_action(
            raw_session: str | None, upload: Any
        ) -> tuple[Any, ...]:
            """Restore safe weights and synchronize the visible model controls."""

            identifier = session_id(raw_session)
            paths = normalize_upload_paths(upload)
            if len(paths) != 1:
                ui_error(
                    ValueError("Choose exactly one .pt checkpoint before loading.")
                )
            try:
                loaded = service.load_checkpoint(identifier, paths[0])
            except Exception as error:
                ui_error(error)
            return (
                loaded.kind.value,
                loaded.n_groups,
                loaded.group_size,
                loaded.l0,
                loaded.coef,
                loaded.target_l0,
                loaded.gain,
                loaded.paper_version,
                None,
                None,
                None,
                [],
                gr.update(choices=[], value=[]),
                None,
                None,
                None,
                "Checkpoint loaded; encode again before analysis.",
                log_tail(identifier),
            )

        load_checkpoint.click(
            load_checkpoint_action,
            inputs=[browser_session, checkpoint_upload],
            outputs=[
                *model_inputs,
                live_r2,
                live_l0,
                dead_groups,
                concept_table,
                concepts,
                concept_plot,
                png_download,
                pdf_download,
                status,
                full_log,
            ],
            **GPU_EVENT_OPTIONS,
        )

        def export_action(
            raw_session: str | None, method: Any, success: str
        ) -> tuple[Path, str, str]:
            """Run one generated-artifact export with consistent feedback."""

            identifier = session_id(raw_session)
            try:
                path = method(identifier)
            except Exception as error:
                ui_error(error)
            return path, success, log_tail(identifier)

        export_results.click(
            lambda sid: export_action(
                sid, service.export_results, "Results bundle exported."
            ),
            inputs=browser_session,
            outputs=[result_download, status, full_log],
            **GPU_EVENT_OPTIONS,
        )
        export_arrays.click(
            lambda sid: export_action(sid, service.export_arrays, "Arrays exported."),
            inputs=browser_session,
            outputs=[arrays_download, status, full_log],
            **GPU_EVENT_OPTIONS,
        )

        def reset_action(raw_session: str | None) -> tuple[Any, ...]:
            """Release large state and clear every derived output component."""

            identifier = session_id(raw_session)
            service.reset_session(identifier)
            return (
                "Session reset; generated files remain available on disk.",
                None,
                None,
                None,
                [],
                gr.update(choices=[], value=[]),
                None,
                None,
                None,
                None,
                None,
                None,
                log_tail(identifier),
            )

        def cancel_for_reset(raw_session: str | None) -> None:
            """Signal a worker immediately before the serialized reset phase."""

            service.cancel_training(session_id(raw_session))

        reset_cancel = reset.click(
            cancel_for_reset,
            inputs=browser_session,
            outputs=[],
            queue=False,
            api_visibility="private",
        )
        reset_cancel.then(
            reset_action,
            inputs=browser_session,
            outputs=[
                status,
                live_r2,
                live_l0,
                dead_groups,
                concept_table,
                concepts,
                concept_plot,
                png_download,
                pdf_download,
                checkpoint_download,
                result_download,
                arrays_download,
                full_log,
            ],
            api_visibility="private",
            **GPU_EVENT_OPTIONS,
        )

        def run_pipeline_action(
            raw_session: str | None,
            extraction_batch: Any,
            *values: Any,
        ) -> Iterator[tuple[Any, ...]]:
            """Build typed current controls and stream the full pipeline."""

            identifier = session_id(raw_session)
            try:
                model_value = _model_config(*values[: len(model_inputs)])
                training_value = _training_config(*values[len(model_inputs) :])
                for update in controller.pipeline_stream(
                    identifier,
                    model_value,
                    training_value,
                    extraction_batch_size=_integer(
                        extraction_batch, "Extraction batch size"
                    ),
                ):
                    status_value, r2, l0_value, dead, rows, selected, log_value = update
                    choices = [int(row[1]) for row in rows]
                    yield (
                        status_value,
                        r2,
                        l0_value,
                        dead,
                        rows,
                        gr.update(choices=choices, value=selected),
                        log_value,
                    )
            except Exception as error:
                ui_error(error)

        run_pipeline.click(
            run_pipeline_action,
            inputs=[
                browser_session,
                extraction_batch_size,
                *model_inputs,
                *training_inputs,
            ],
            outputs=[
                status,
                live_r2,
                live_l0,
                dead_groups,
                concept_table,
                concepts,
                full_log,
            ],
            **GPU_EVENT_OPTIONS,
        )

        cleanup_timer = gr.Timer(value=cache_interval, active=True, render=False)

        def cleanup() -> None:
            """Release stale registry entries in addition to Gradio cache cleanup."""

            service.cleanup_expired()

        cleanup_timer.tick(
            cleanup,
            inputs=[],
            outputs=[],
            queue=False,
            api_visibility="private",
        )

    # Notebook users may close this service explicitly after calling ``close`` on
    # the returned Blocks object; the attribute is intentionally process-local.
    setattr(demo, "bsf_pipeline", service)
    return demo


def launch_app(config: AppConfig | None = None) -> None:
    """Launch on loopback only with sharing disabled and bounded file exposure."""

    settings = config or load_app_config()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    app = build_app(settings)
    # Shared concurrency IDs serialize all GPU-capable actions at one worker.
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="127.0.0.1",
        server_port=settings.port,
        share=False,
        allowed_paths=[str(settings.output_dir)],
        blocked_paths=[str(settings.env_file)],
        max_file_size=settings.max_upload_mb * 1024 * 1024,
        show_error=True,
    )


__all__ = [
    "CONCEPT_HEADERS",
    "FIXED_DINO_MODEL",
    "GPU_EVENT_OPTIONS",
    "WorkbenchController",
    "build_app",
    "format_training_event",
    "launch_app",
    "normalize_upload_paths",
    "preset_control_values",
    "safe_ui_error_message",
    "ui_error",
]
