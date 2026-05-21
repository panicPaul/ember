"""Error-MCMC paper training notebook for Ember."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import importlib.util
    import sys
    import types
    from pathlib import Path
    from typing import Any, Literal

    import ember_core as ember
    import ember_native_faster_gs.fastgs as ember_fastgs_native
    import ember_splatting_training as ember_splatting
    import marimo as mo
    import torch
    from ember_core.densification import Schedule
    from ember_core.training import TrainingProfilerConfig, TrainingResult
    from marimo_config_gui import (
        ConfigPreset,
        ConfigPresetCatalog,
        create_config_gui,
    )
    from pydantic import Field
    from torch import Tensor

    NOTEBOOK_PATH = Path(__file__).resolve()
    NOTEBOOK_DIR = NOTEBOOK_PATH.parent
    REPO_ROOT = NOTEBOOK_DIR.parents[1]

    def _ensure_namespace_package(name: str, path: Path) -> types.ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        module.__package__ = name
        paths = list(getattr(module, "__path__", []))
        path_str = str(path)
        if path_str not in paths:
            paths.append(path_str)
        module.__path__ = paths
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            setattr(sys.modules[parent_name], attr, module)
        return module

    _ensure_namespace_package("papers", REPO_ROOT / "papers")
    _ensure_namespace_package("papers.fastgs", REPO_ROOT / "papers" / "fastgs")
    _ensure_namespace_package("papers.error_mcmc", NOTEBOOK_DIR)
    if "papers.fastgs.notebook" in sys.modules:
        fastgs = sys.modules["papers.fastgs.notebook"]
    else:
        fastgs_path = REPO_ROOT / "papers" / "fastgs" / "notebook.py"
        fastgs_spec = importlib.util.spec_from_file_location(
            "papers.fastgs.notebook",
            fastgs_path,
        )
        if fastgs_spec is None or fastgs_spec.loader is None:
            raise ImportError(f"Could not load {fastgs_path}.")
        fastgs = importlib.util.module_from_spec(fastgs_spec)
        sys.modules[fastgs_spec.name] = fastgs
        fastgs_spec.loader.exec_module(fastgs)

    from papers.error_mcmc._densification import ErrorMCMC

    ErrorMCMCDensification = ErrorMCMC
    ErrorMCMCFinalCleanup = fastgs.FastGSFinalCleanup
    FastGSScheduledAdam = fastgs.FastGSScheduledAdam
    FastGSSHTrainingHook = fastgs.FastGSSHTrainingHook
    rgb_l1_ssim_loss = fastgs.rgb_l1_ssim_loss
    DEFAULTS_DIR = NOTEBOOK_DIR / "defaults"
    DEFAULT_CHECKPOINT_ROOT = (
        REPO_ROOT / "checkpoints" / "papers" / "error_mcmc"
    )
    ErrorMCMCBackendName = Literal["faster_gs.fastgs"]
    ErrorMCMCDefaultName = Literal[
        "garden_error_mcmc",
        "garden_debug_val",
    ]
    ErrorMCMCScoreAggregation = Literal[
        "mean",
        "topk_mean",
        "max",
        "visibility_normalized",
    ]
    sys.modules.setdefault("papers.error_mcmc.notebook", sys.modules[__name__])
    ember_fastgs_native.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Error-MCMC training

    ## IO
    """)
    return


@app.cell(hide_code=True)
def _(training_controls):
    training_controls
    return


@app.cell(hide_code=True)
def _(training_preparation_status):
    training_preparation_status
    return


@app.cell(hide_code=True)
def _(preset_selector):
    preset_selector
    return


@app.cell(hide_code=True)
def _(config_gui):
    config_gui.stacked()
    return


@app.cell(hide_code=True)
def _(training_result_view):
    training_result_view
    return


@app.cell(hide_code=True)
def _(training_viewer):
    training_viewer
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## IO wiring and execution
    """)
    return


@app.cell
def _():
    prepare_button = mo.ui.run_button(
        label="Prepare training inspector",
        full_width=True,
    )
    train_button = mo.ui.run_button(
        label="Start training",
        full_width=True,
    )
    stop_button = mo.ui.run_button(
        label="Stop training",
        full_width=True,
    )
    training_status_refresh = mo.ui.refresh(
        options=["1s"],
        default_interval="1s",
        label="Training status",
    )
    training_inspector_refresh = mo.ui.refresh(
        options=["5s", "10s", "30s", "1m"],
        default_interval="10s",
        label="Image refresh",
    )
    training_controls = mo.vstack(
        [
            prepare_button,
            train_button,
            stop_button,
            training_status_refresh,
            training_inspector_refresh,
        ],
        gap=0.5,
    )
    return (
        prepare_button,
        stop_button,
        train_button,
        training_controls,
        training_inspector_refresh,
        training_status_refresh,
    )


@app.cell
def _():
    is_script_mode = not mo.running_in_notebook()
    return (is_script_mode,)


@app.cell
def _(current_config):
    training_preparation_handle = None
    training_preparation_snapshot = None
    if current_config is not None:
        training_preparation_handle, training_preparation_snapshot = (
            ember_splatting.create_training_preparation(
                load_scene=lambda: ember.load_scene_record(
                    build_scene_load_config(current_config)
                ),
                prepare_frame_view_catalog=lambda scene_record: (
                    ember.build_prepared_frame_view_catalog(
                        scene_record,
                        build_prepared_frame_dataset_config(current_config),
                    )
                ),
            )
        )
    return training_preparation_handle, training_preparation_snapshot


@app.cell
def _(
    current_config,
    is_script_mode,
    prepare_button,
    train_button,
    training_preparation_handle,
):
    should_prepare_training_inputs = (
        is_script_mode or bool(prepare_button.value) or bool(train_button.value)
    )
    if (
        should_prepare_training_inputs
        and current_config is not None
        and training_preparation_handle is not None
    ):
        training_preparation_handle.start(wait=is_script_mode)
    return


@app.cell
def _(ember_splatting, training_preparation_snapshot):
    preparation_status_snapshot = (
        training_preparation_snapshot()
        if training_preparation_snapshot is not None
        else None
    )
    training_preparation_status = (
        ember_splatting.render_training_preparation_status(
            preparation_status_snapshot
        )
    )
    return (training_preparation_status,)


@app.cell
def _(ember_splatting, training_preparation_snapshot):
    preparation_outputs_snapshot = (
        training_preparation_snapshot()
        if training_preparation_snapshot is not None
        else None
    )
    (
        scene_load_error,
        scene_record,
        frame_dataset,
        frame_dataset_error,
        frame_view_catalog,
    ) = ember_splatting.training_preparation_outputs(
        preparation_outputs_snapshot
    )
    return (
        scene_load_error,
        scene_record,
        frame_dataset,
        frame_dataset_error,
        frame_view_catalog,
    )


@app.cell
def _(current_config, frame_dataset):
    training_config = (
        resolve_training_config(current_config, frame_dataset)
        if current_config is not None and frame_dataset is not None
        else None
    )
    return (training_config,)


@app.cell
def _(current_config, frame_dataset, is_script_mode, training_config):
    training_viewer_handle = (
        ember_splatting.create_training_run(
            frame_dataset,
            training_config,
            config=current_config.training.viewer,
            title="Error-MCMC training inspector",
        )
        if not is_script_mode
        and current_config is not None
        and frame_dataset is not None
        and training_config is not None
        else None
    )
    return (training_viewer_handle,)


@app.cell
def _(frame_view_catalog, is_script_mode):
    training_inspector = (
        None
        if is_script_mode or frame_view_catalog is None
        else ember_splatting.create_training_view_inspector(
            frame_view_catalog,
        )
    )
    return (training_inspector,)


@app.cell
def _(
    frame_view_catalog,
    training_inspector,
    training_inspector_refresh,
    training_viewer_handle,
):
    training_viewer = (
        None
        if training_inspector is None
        else training_inspector.panel(
            training_viewer_handle,
            frame_view_catalog,
            refresh=training_inspector_refresh,
        )
    )
    return (training_viewer,)


@app.cell
def _(
    current_config,
    frame_dataset,
    is_script_mode,
    train_button,
    training_config,
    training_viewer_handle,
):
    should_start_training = bool(train_button.value)
    if (
        is_script_mode
        and current_config is not None
        and frame_dataset is not None
        and training_config is not None
    ):
        training_result = run_error_mcmc_training(
            frame_dataset,
            current_config,
            training_config,
        )
    else:
        training_result = None
        if (
            should_start_training
            and frame_dataset is not None
            and training_config is not None
            and training_viewer_handle is not None
        ):
            training_viewer_handle.start_training(
                frame_dataset,
                training_config,
            )
    return (training_result,)


@app.cell
def _(stop_button, training_viewer_handle):
    should_stop = bool(stop_button.value)
    if should_stop and training_viewer_handle is not None:
        training_viewer_handle.request_stop()
    return


@app.cell
def _(training_result, training_status_refresh, training_viewer_handle):
    _ = training_status_refresh.value
    training_result_view = (
        ember_splatting.render_training_status_panel_from_handle(
            training_viewer_handle,
            training_result=training_result,
        )
    )
    return (training_result_view,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Method and config
    """)
    return


@app.class_definition
class ErrorMCMCConfigBase(fastgs.FastGSConfigBase):
    """Strict base model for Error-MCMC paper configs."""


@app.class_definition
class ErrorMCMCSceneConfig(fastgs.FastGSSceneConfig):
    """Scene-record loading options."""


@app.class_definition
class ErrorMCMCDataConfig(fastgs.FastGSDataConfig):
    """Prepared-frame dataset options."""


@app.class_definition
class ErrorMCMCInitializationConfig(fastgs.FastGSInitializationConfig):
    """Gaussian initialization tuned for MCMC-style relocation."""

    use_mcmc: bool = True

    def build(
        self,
        context: ember.TrainingRunContext,
    ) -> ember.InitializationSpec:
        """Build the runtime initializer spec."""
        del context
        return ember.InitializationSpec(
            initializer=ember.bound_callable(
                target=(
                    "papers.error_mcmc.notebook."
                    "initialize_error_mcmc_model_from_scene_record"
                ),
                kwargs=self.model_dump(mode="python"),
                bind={"device": ember.ctx.run.device},
            )
        )


@app.class_definition
class ErrorMCMCRenderConfig(fastgs.FastGSRenderConfig):
    """Native FastGS render pipeline config for Error-MCMC."""

    backend: ErrorMCMCBackendName = "faster_gs.fastgs"


@app.class_definition
class ErrorMCMCOptimizationConfig(fastgs.FastGSOptimizationConfig):
    """FastGS optimizer config with MCMC opacity learning rate."""

    opacity_lr: float = Field(default=0.05, gt=0.0)


@app.class_definition
class ErrorMCMCLossConfig(fastgs.FastGSLossConfig):
    """Training loss config with light MCMC regularization defaults."""

    lambda_opacity_regularization: float = Field(default=0.01, ge=0.0)
    lambda_scale_regularization: float = Field(default=0.01, ge=0.0)

    def build(self, context: ember.TrainingRunContext) -> ember.LossConfig:
        """Build the runtime loss config."""
        del context
        return ember.loss_config(
            "papers.error_mcmc.notebook.rgb_l1_ssim_loss",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class ErrorMCMCDensificationConfig(ErrorMCMCConfigBase):
    """Error-aware MCMC relocation and capped growth config."""

    refine_every: int = Field(default=100, ge=1)
    start_iter: int = Field(default=600, ge=0)
    stop_iter: int = Field(default=24_900, ge=0)
    min_opacity: float = Field(default=0.005, gt=0.0, lt=1.0)
    max_primitives: int = Field(default=1_000_000, ge=1)
    cap_growth_factor: float = Field(default=1.05, gt=1.0)
    inject_position_noise: bool = True
    noise_lr_scale: float = Field(default=500_000.0, gt=0.0)
    loss_thresh: float = Field(default=0.06, ge=0.0)
    probe_view_count: int = Field(default=32, ge=1)
    score_aggregation: ErrorMCMCScoreAggregation = "topk_mean"
    score_top_k: int = Field(default=3, ge=1)
    opacity_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    opacity_power: float = Field(default=0.5, ge=0.0)
    normalize_error_score: bool = False

    def build(self, context: ember.TrainingRunContext) -> ember.CallableSpec:
        """Build the runtime Error-MCMC densification spec."""
        del context
        payload = self.model_dump(mode="python")
        start_iter = payload.pop("start_iter")
        stop_iter = payload.pop("stop_iter")
        refine_every = payload.pop("refine_every")
        max_primitives = payload.pop("max_primitives")
        payload["schedule"] = Schedule(
            start_iteration=start_iter,
            end_iteration=stop_iter,
            frequency=refine_every,
        )
        payload["cap_max"] = max_primitives
        return ember.bound_callable(
            target="papers.error_mcmc.notebook.ErrorMCMCDensification",
            kwargs=payload,
        )


@app.class_definition
class ErrorMCMCFinalCleanupConfig(fastgs.FastGSFinalCleanupConfig):
    """Checkpoint cleanup config."""

    def build(self, context: ember.TrainingRunContext) -> ember.CallableSpec:
        """Build the runtime final cleanup spec."""
        del context
        return ember.bound_callable(
            target="papers.error_mcmc.notebook.ErrorMCMCFinalCleanup",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class ErrorMCMCMipSplattingConfig(fastgs.FastGSMipSplattingConfig):
    """Full Mip-Splatting controls enabled by default for Error-MCMC."""

    enabled: bool = True
    screen_filter_enabled: bool = True


@app.class_definition
class ErrorMCMCDensificationStackConfig(ErrorMCMCConfigBase):
    """Typed Error-MCMC densification stack config."""

    error_mcmc: ErrorMCMCDensificationConfig = Field(
        default_factory=ErrorMCMCDensificationConfig
    )
    morton: fastgs.FastGSMortonOrderingConfig = Field(
        default_factory=fastgs.FastGSMortonOrderingConfig
    )
    final_cleanup: ErrorMCMCFinalCleanupConfig = Field(
        default_factory=ErrorMCMCFinalCleanupConfig
    )

    def build(
        self,
        context: ember.TrainingRunContext,
        *,
        mip_splatting: fastgs.FastGSMipSplattingConfig,
    ) -> ember.DensificationConfig:
        """Build the runtime Error-MCMC densification stack."""
        builders = [
            self.error_mcmc.build(context),
            self.morton.build(context),
        ]
        if mip_splatting.enabled:
            builders.append(
                mip_splatting.three_dimensional_filter.build(context)
            )
        builders.append(self.final_cleanup.build(context))
        return ember.densification_config(*builders)


@app.class_definition
class ErrorMCMCTrainingConfig(ErrorMCMCConfigBase):
    """Typed user-facing Error-MCMC training config."""

    runtime: ember.RuntimeConfig = Field(default_factory=ember.RuntimeConfig)
    profiler: TrainingProfilerConfig = Field(
        default_factory=TrainingProfilerConfig
    )
    batching: ember.BatchingConfig = Field(default_factory=ember.BatchingConfig)
    initialization: ErrorMCMCInitializationConfig = Field(
        default_factory=ErrorMCMCInitializationConfig
    )
    render: ErrorMCMCRenderConfig = Field(default_factory=ErrorMCMCRenderConfig)
    mip_splatting: ErrorMCMCMipSplattingConfig = Field(
        default_factory=ErrorMCMCMipSplattingConfig
    )
    optimization: ErrorMCMCOptimizationConfig = Field(
        default_factory=ErrorMCMCOptimizationConfig
    )
    loss: ErrorMCMCLossConfig = Field(default_factory=ErrorMCMCLossConfig)
    densification: ErrorMCMCDensificationStackConfig = Field(
        default_factory=ErrorMCMCDensificationStackConfig
    )
    checkpoint: ember.CheckpointExportConfig = Field(
        default_factory=ember.CheckpointExportConfig
    )
    viewer: ember_splatting.TrainingViewerConfig = Field(
        default_factory=ember_splatting.TrainingViewerConfig
    )

    def to_training_config(
        self,
        frame_dataset: ember.PreparedFrameDataset | None = None,
    ) -> ember.TrainingConfig:
        """Materialize this typed config into Ember's runtime config."""
        camera_extent = (
            ember.compute_frame_camera_extent(frame_dataset)
            if frame_dataset is not None
            else 1.0
        )
        context = ember.TrainingRunContext(
            frame_dataset=frame_dataset,
            camera_extent=camera_extent,
            max_steps=self.runtime.max_steps,
            backend=self.render.backend,
            device=torch.device(self.runtime.device),
        )
        return ember.TrainingConfig(
            runtime=self.runtime,
            profiler=self.profiler,
            batching=self.batching,
            initialization=self.initialization.build(context),
            render=self.render.build(
                context,
                mip_splatting_screen_filter=(
                    self.mip_splatting.enabled
                    and self.mip_splatting.screen_filter_enabled
                ),
            ),
            optimization=self.optimization.build(context),
            loss=self.loss.build(context),
            densification=self.densification.build(
                context,
                mip_splatting=self.mip_splatting,
            ),
            hooks=ember.HookConfig(
                builders=[
                    ember.bound_callable(
                        target=(
                            "papers.error_mcmc.notebook.FastGSSHTrainingHook"
                        ),
                        kwargs=self.render.training_backend_options.model_dump(
                            mode="python"
                        ),
                    )
                ]
            ),
            checkpoint=self.checkpoint,
        )


@app.class_definition
class ErrorMCMCExperimentConfig(ErrorMCMCConfigBase):
    """Resolved Error-MCMC experiment config."""

    preset: ErrorMCMCDefaultName = "garden_error_mcmc"
    scene: ErrorMCMCSceneConfig = Field(default_factory=ErrorMCMCSceneConfig)
    data: ErrorMCMCDataConfig = Field(default_factory=ErrorMCMCDataConfig)
    training: ErrorMCMCTrainingConfig


@app.cell
def _():
    error_mcmc_presets = error_mcmc_preset_catalog()
    config_gui = create_config_gui(
        ErrorMCMCExperimentConfig,
        presets=error_mcmc_presets,
        path_defaults_source=DEFAULTS_DIR,
        label="Error-MCMC config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    preset_selector = config_gui.preset_selector(
        label="Error-MCMC preset",
    )
    return (preset_selector,)


@app.cell
def _(config_gui):
    current_config = config_gui.validated_config()
    return (current_config,)


@app.function
def default_checkpoint_dir(
    preset: ErrorMCMCDefaultName,
    backend: ErrorMCMCBackendName,
) -> Path:
    """Return the default checkpoint directory for a preset/backend pair."""
    return DEFAULT_CHECKPOINT_ROOT / preset / backend


@app.function
def error_mcmc_preset_catalog() -> ConfigPresetCatalog[
    ErrorMCMCExperimentConfig
]:
    """Return the notebook's named JSON preset catalog."""
    return ConfigPresetCatalog(
        model_cls=ErrorMCMCExperimentConfig,
        presets={
            "garden_error_mcmc": ConfigPreset(
                name="garden_error_mcmc",
                path=DEFAULTS_DIR / "garden_error_mcmc.json",
                label="Garden Error-MCMC",
                base_dir=REPO_ROOT,
            ),
            "garden_debug_val": ConfigPreset(
                name="garden_debug_val",
                path=DEFAULTS_DIR / "garden_debug_val.json",
                label="Garden debug validation",
                base_dir=REPO_ROOT,
            ),
        },
        default="garden_error_mcmc",
        path_defaults=(
            REPO_ROOT
            / "papers"
            / "fastgs"
            / "defaults"
            / ".path_defaults.json",
        ),
    )


@app.function
def resolve_checkpoint_output_dir(
    config: ErrorMCMCExperimentConfig,
) -> Path:
    """Mirror checkpoint dirs by preset and backend unless user changed them."""
    default_parent = DEFAULT_CHECKPOINT_ROOT / config.preset
    output_dir = config.training.checkpoint.output_dir.expanduser()
    if output_dir.parent == default_parent:
        return default_checkpoint_dir(
            config.preset,
            config.training.render.backend,
        )
    return output_dir


@app.function
def resolve_training_config(
    config: ErrorMCMCExperimentConfig,
    frame_dataset: ember.PreparedFrameDataset | None = None,
) -> ember.TrainingConfig:
    """Apply paper notebook runtime defaults to native Ember training config."""
    checkpoint = config.training.checkpoint.model_copy(
        update={
            "output_dir": resolve_checkpoint_output_dir(config),
        },
    )
    training = config.training.model_copy(
        update={"checkpoint": checkpoint},
        deep=True,
    )
    return training.to_training_config(frame_dataset)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Utilities
    """)
    return


@app.function
def initialize_error_mcmc_model_from_scene_record(
    scene_record: ember.SceneRecord,
    *,
    modules: dict[str, torch.nn.Module] | None = None,
    parameters: dict[str, torch.nn.Parameter] | None = None,
    buffers: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
    sh_degree: int = 3,
    use_mcmc: bool = True,
    default_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    device: torch.device | None = None,
) -> ember.InitializedModel:
    """Initialize Error-MCMC Gaussians from the FastGS initializer."""
    return fastgs.initialize_fastgs_model_from_scene_record(
        scene_record,
        modules=modules,
        parameters=parameters,
        buffers=buffers,
        metadata=metadata,
        sh_degree=sh_degree,
        use_mcmc=use_mcmc,
        default_color=default_color,
        device=device,
    )


@app.function
def build_scene_load_config(
    config: ErrorMCMCExperimentConfig,
) -> ember.ColmapSceneConfig:
    """Translate paper config into an Ember scene loader config."""
    return fastgs.build_scene_load_config(config)


@app.function
def build_prepared_frame_dataset_config(
    config: ErrorMCMCExperimentConfig,
) -> ember.PreparedFrameDatasetConfig:
    """Translate paper config into an Ember frame dataset config."""
    return fastgs.build_prepared_frame_dataset_config(config)


@app.function
def run_error_mcmc_training(
    frame_dataset: ember.PreparedFrameDataset,
    experiment_config: ErrorMCMCExperimentConfig,
    training_config: ember.TrainingConfig | None = None,
    training_viewer_handle: ember_splatting.TrainingViewerHandle | None = None,
) -> TrainingResult:
    """Run Error-MCMC training from a native Ember training config."""
    resolved_training_config = training_config or resolve_training_config(
        experiment_config,
        frame_dataset,
    )
    return ember.run_training(
        frame_dataset,
        resolved_training_config,
        runtime_hooks=(
            ()
            if training_viewer_handle is None
            else training_viewer_handle.runtime_hooks()
        ),
    )


if __name__ == "__main__":
    app.run()
