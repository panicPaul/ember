"""Triangle Splatting paper training notebook for Ember."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path
    from typing import Literal

    import ember_core as ember
    import ember_native_triangle_splatting as ember_triangle_splatting
    import ember_native_triangle_splatting.triangle_splatting as triangle_native
    import ember_splatting_training as ember_splatting
    import marimo as mo
    import torch
    from ember_core.training import TrainingProfilerConfig, TrainingResult
    from marimo_config_gui import (
        ConfigPreset,
        ConfigPresetCatalog,
        create_config_gui,
    )
    from pydantic import BaseModel, Field

    NOTEBOOK_PATH = Path(__file__).resolve()
    NOTEBOOK_DIR = NOTEBOOK_PATH.parent
    REPO_ROOT = NOTEBOOK_DIR.parents[1]
    DEFAULTS_DIR = NOTEBOOK_DIR / "defaults"
    DEFAULT_CHECKPOINT_ROOT = (
        REPO_ROOT / "checkpoints" / "papers" / "triangle_splatting"
    )
    TriangleSplattingBackendName = Literal["triangle_splatting.core"]
    TriangleSplattingDefaultName = Literal["garden", "garden_debug_val"]
    sys.modules.setdefault(
        "papers.triangle_splatting.notebook",
        sys.modules[__name__],
    )
    ember_triangle_splatting.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Triangle Splatting training

    ## IO
    """)
    return


@app.cell(hide_code=True)
def _(training_preparation_status):
    training_preparation_status
    return


@app.cell
def _():
    presets = triangle_splatting_preset_catalog()
    config_gui = create_config_gui(
        TriangleSplattingExperimentConfig,
        presets=presets,
        path_defaults_source=DEFAULTS_DIR,
        label="Triangle Splatting config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    preset_selector = config_gui.preset_selector(
        label="Triangle Splatting preset",
    )
    return (preset_selector,)


@app.cell
def _(config_gui):
    current_config = config_gui.validated_config()
    return (current_config,)


@app.cell(hide_code=True)
def _(preset_selector):
    preset_selector
    return


@app.cell(hide_code=True)
def _(config_gui):
    config_gui.stacked()
    return


@app.cell(hide_code=True)
def _(training_controls):
    training_controls
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
    ## Method and config
    """)
    return


@app.class_definition
class TriangleSplattingConfigBase(BaseModel):
    """Strict base model for Triangle Splatting configs."""

    model_config = {"extra": "forbid", "populate_by_name": True}


@app.class_definition
class TriangleSplattingSceneConfig(TriangleSplattingConfigBase):
    """Scene-record loading options."""

    path: Path = Path("dataset/mipnerf360/garden")
    image_directory: str = "images_4"
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    align_horizon: bool = True


@app.class_definition
class TriangleSplattingDataConfig(TriangleSplattingConfigBase):
    """Prepared-frame dataset options."""

    camera_sensor_id: str | None = None
    image_scale_factor: float = Field(default=1.0, gt=0.0)
    split_target: Literal["train", "val", "all"] = "train"
    split_every_n: int | None = Field(default=8, ge=1)
    materialization_stage: Literal["none", "decoded", "prepared"] = "prepared"
    materialization_mode: Literal["lazy", "eager"] = "eager"
    materialization_num_workers: int | None = 8
    normalize_images: bool = True
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic"


@app.class_definition
class TriangleSplattingInitializationConfig(TriangleSplattingConfigBase):
    """Typed Triangle Splatting initialization config."""

    sh_degree: int = Field(default=3, ge=0)
    vertices_per_triangle: int = Field(default=3, ge=3)
    initial_opacity: float = Field(default=0.28, gt=0.0, lt=1.0)
    initial_triangle_size: float = Field(default=2.23, gt=0.0)
    initial_sigma: float = Field(default=1.16, gt=0.01)
    include_sky_dome: bool = True
    seed: int = 0

    def build(
        self,
        context: ember.TrainingRunContext,
    ) -> ember.InitializationSpec:
        """Build the runtime initializer spec."""
        del context
        return ember.InitializationSpec(
            initializer=ember.bound_callable(
                target=(
                    "ember_native_triangle_splatting.triangle_splatting."
                    "initialize_triangle_splatting_model_from_scene_record"
                ),
                kwargs=self.model_dump(mode="python"),
                bind={"device": ember.ctx.run.device},
            )
        )


@app.class_definition
class TriangleSplattingRenderConfig(TriangleSplattingConfigBase):
    """Typed Triangle Splatting render pipeline config."""

    backend: TriangleSplattingBackendName = "triangle_splatting.core"
    near_plane: float = Field(default=0.01, gt=0.0)
    far_plane: float = Field(default=100.0, gt=0.0)
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    return_alpha: bool = True
    return_depth: bool = True
    return_normals: bool = True
    clamp_output: bool = False
    depth_mode: Literal["median", "expected"] = "median"

    def build(
        self, context: ember.TrainingRunContext
    ) -> ember.RenderPipelineSpec:
        """Build the runtime render pipeline spec."""
        del context
        return ember.RenderPipelineSpec(
            backend=self.backend,
            backend_options={
                "near_plane": self.near_plane,
                "far_plane": self.far_plane,
                "background_color": list(self.background_color),
                "clamp_output": self.clamp_output,
                "depth_mode": self.depth_mode,
            },
            return_alpha=self.return_alpha,
            return_depth=self.return_depth,
            return_normals=self.return_normals,
        )


@app.class_definition
class TriangleSplattingOptimizationConfig(TriangleSplattingConfigBase):
    """Typed upstream Triangle Splatting optimizer defaults."""

    optimizer: str = "torch.optim.Adam"
    feature_lr: float = Field(default=0.0025, gt=0.0)
    opacity_lr: float = Field(default=0.014, gt=0.0)
    sigma_lr: float = Field(default=0.0008, gt=0.0)
    triangle_vertices_lr_init: float = Field(default=0.0018, gt=0.0)
    triangle_vertices_lr_final_factor: float = Field(default=0.01, gt=0.0)
    triangle_vertices_lr_max_steps: int = Field(default=30_000, ge=1)
    adam_eps: float = Field(default=1e-15, gt=0.0)

    def build(
        self,
        context: ember.TrainingRunContext,
    ) -> ember.OptimizationConfig:
        """Build runtime optimizer groups from the typed recipe."""
        del context
        return triangle_native.triangle_splatting_optimization_config(
            self.model_dump(mode="python")
        )


@app.class_definition
class TriangleSplattingLossConfig(TriangleSplattingConfigBase):
    """Typed upstream Triangle Splatting training loss config."""

    lambda_dssim: float = Field(default=0.2, ge=0.0)
    lambda_opacity: float = Field(default=0.0055, ge=0.0)
    lambda_size: float = Field(default=1e-8, ge=0.0)
    lambda_normals: float = Field(default=0.0001, ge=0.0)
    lambda_dist: float = Field(default=0.0, ge=0.0)
    iteration_mesh: int = Field(default=5000, ge=0)
    densify_until_iter: int = Field(default=25_000, ge=0)
    outdoor: bool = True

    def build(self, context: ember.TrainingRunContext) -> ember.LossConfig:
        """Build the runtime loss config."""
        del context
        return ember.loss_config(
            "ember_native_triangle_splatting.triangle_splatting."
            "triangle_splatting_loss",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class TriangleSplattingDensificationConfig(TriangleSplattingConfigBase):
    """Typed upstream Triangle Splatting split/clone/prune schedule."""

    densification_interval: int = Field(default=500, ge=1)
    densify_from_iter: int = Field(default=500, ge=0)
    densify_until_iter: int = Field(default=25_000, ge=0)
    max_shapes: int = Field(default=5_200_000, ge=1)
    add_shape: float = Field(default=1.3, gt=1.0)
    split_size: float = Field(default=24.0, gt=0.0)
    max_noise_factor: float = Field(default=1.5, ge=0.0)
    opacity_dead: float = Field(default=0.014, ge=0.0, lt=1.0)
    importance_threshold: float = Field(default=0.022, ge=0.0)
    proba_distr: Literal[0, 1, 2] = 2
    outdoor: bool = True

    def build(
        self, context: ember.TrainingRunContext
    ) -> ember.DensificationConfig:
        """Build the runtime densification config."""
        del context
        return ember.densification_config(
            ember.callable_spec(
                "ember_native_triangle_splatting.triangle_splatting."
                "TriangleSplattingDensification",
                kwargs=self.model_dump(mode="python"),
            )
        )


@app.class_definition
class TriangleSplattingTrainingConfig(TriangleSplattingConfigBase):
    """Typed user-facing Triangle Splatting training config."""

    runtime: ember.RuntimeConfig = Field(
        default_factory=lambda: ember.RuntimeConfig(
            device="cuda",
            seed=0,
            max_steps=30_000,
        )
    )
    batching: ember.BatchingConfig = Field(
        default_factory=lambda: ember.BatchingConfig(
            batch_size=1,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )
    )
    initialization: TriangleSplattingInitializationConfig = Field(
        default_factory=TriangleSplattingInitializationConfig
    )
    render: TriangleSplattingRenderConfig = Field(
        default_factory=TriangleSplattingRenderConfig
    )
    optimization: TriangleSplattingOptimizationConfig = Field(
        default_factory=TriangleSplattingOptimizationConfig
    )
    loss: TriangleSplattingLossConfig = Field(
        default_factory=TriangleSplattingLossConfig
    )
    densification: TriangleSplattingDensificationConfig = Field(
        default_factory=TriangleSplattingDensificationConfig
    )
    profiler: TrainingProfilerConfig = Field(
        default_factory=TrainingProfilerConfig
    )
    checkpoint: ember.CheckpointExportConfig = Field(
        default_factory=lambda: ember.CheckpointExportConfig(
            output_dir=DEFAULT_CHECKPOINT_ROOT / "latest",
            export_ply=False,
            overwrite=False,
        )
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
            render=self.render.build(context),
            optimization=self.optimization.build(context),
            loss=self.loss.build(context),
            densification=self.densification.build(context),
            checkpoint=self.checkpoint,
        )


@app.class_definition
class TriangleSplattingExperimentConfig(TriangleSplattingConfigBase):
    """Resolved Triangle Splatting experiment config."""

    preset: TriangleSplattingDefaultName = "garden"
    scene: TriangleSplattingSceneConfig = Field(
        default_factory=TriangleSplattingSceneConfig
    )
    data: TriangleSplattingDataConfig = Field(
        default_factory=TriangleSplattingDataConfig
    )
    training: TriangleSplattingTrainingConfig = Field(
        default_factory=TriangleSplattingTrainingConfig
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Training assembly
    """)
    return


@app.function
def default_checkpoint_dir(
    preset: TriangleSplattingDefaultName,
    backend: TriangleSplattingBackendName,
) -> Path:
    """Return the default checkpoint directory for a preset/backend pair."""
    return DEFAULT_CHECKPOINT_ROOT / preset / backend


@app.function
def triangle_splatting_preset_catalog() -> ConfigPresetCatalog:
    """Return the notebook's named JSON preset catalog."""
    return ConfigPresetCatalog(
        model_cls=TriangleSplattingExperimentConfig,
        presets={
            "garden": ConfigPreset(
                name="garden",
                path=DEFAULTS_DIR / "garden.json",
                label="Garden upstream",
                base_dir=REPO_ROOT,
            ),
            "garden_debug_val": ConfigPreset(
                name="garden_debug_val",
                path=DEFAULTS_DIR / "garden_debug_val.json",
                label="Garden debug validation",
                base_dir=REPO_ROOT,
            ),
        },
        default="garden",
    )


@app.function
def resolve_checkpoint_output_dir(
    config: TriangleSplattingExperimentConfig,
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
    config: TriangleSplattingExperimentConfig,
    frame_dataset: ember.PreparedFrameDataset | None = None,
) -> ember.TrainingConfig:
    """Apply paper notebook defaults to native Ember training config."""
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


@app.function
def triangle_splatting_source_image_root(
    config: TriangleSplattingExperimentConfig,
) -> Path:
    """Return the image root matching upstream's ``-i images_4`` flag."""
    if config.scene.image_root is not None:
        return config.scene.image_root.expanduser()
    return config.scene.path.expanduser() / config.scene.image_directory


@app.function
def build_scene_load_config(
    config: TriangleSplattingExperimentConfig,
) -> ember.ColmapSceneConfig:
    """Translate paper config into an Ember scene loader config."""
    source_pipes = (
        (ember.HorizonAlignPipeConfig(),) if config.scene.align_horizon else ()
    )
    return ember.ColmapSceneConfig(
        path=config.scene.path.expanduser(),
        image_root=triangle_splatting_source_image_root(config),
        undistort_output_dir=(
            config.scene.undistort_output_dir.expanduser()
            if config.scene.undistort_output_dir is not None
            else None
        ),
        source_pipes=source_pipes,
    )


@app.function
def build_prepared_frame_dataset_config(
    config: TriangleSplattingExperimentConfig,
) -> ember.PreparedFrameDatasetConfig:
    """Translate paper config into an Ember frame dataset config."""
    split = (
        ember.SplitConfig(target="all", every_n=None, train_ratio=None)
        if config.data.split_target == "all"
        else ember.SplitConfig(
            target=config.data.split_target,
            every_n=config.data.split_every_n,
            train_ratio=None,
        )
    )
    return ember.PreparedFrameDatasetConfig(
        camera_sensor_id=config.data.camera_sensor_id,
        split=split,
        materialization=ember.MaterializationConfig(
            stage=config.data.materialization_stage,
            mode=config.data.materialization_mode,
            num_workers=config.data.materialization_num_workers,
        ),
        image_preparation=ember.ImagePreparationConfig(
            normalize=config.data.normalize_images,
            resize_width_scale=config.data.image_scale_factor,
            resize_width_target=None,
            interpolation=config.data.interpolation,
        ),
    )


@app.function
def run_triangle_splatting_training(
    frame_dataset: ember.PreparedFrameDataset,
    *,
    experiment_config: TriangleSplattingExperimentConfig,
    training_config: ember.TrainingConfig | None = None,
    training_viewer_handle: ember_splatting.TrainingViewerHandle | None = None,
) -> TrainingResult:
    """Run Triangle Splatting training from an Ember training config."""
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


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## IO wiring
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


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Execution
    """)
    return


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
def _(training_preparation_snapshot):
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
def _(training_preparation_snapshot):
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
        frame_dataset,
        frame_dataset_error,
        frame_view_catalog,
        scene_load_error,
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
            title="Triangle Splatting training inspector",
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
        training_result = run_triangle_splatting_training(
            frame_dataset,
            experiment_config=current_config,
            training_config=training_config,
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
def _(
    frame_dataset_error,
    scene_load_error,
    training_result,
    training_status_refresh,
    training_viewer_handle,
):
    _ = training_status_refresh.value
    training_result_view = (
        ember_splatting.render_training_status_panel_from_handle(
            training_viewer_handle,
            preparation_errors=[
                ("Scene loading failed", scene_load_error),
                ("Frame preparation failed", frame_dataset_error),
            ],
            training_result=training_result,
        )
    )
    return (training_result_view,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Utilities
    """)
    return


if __name__ == "__main__":
    app.run()
