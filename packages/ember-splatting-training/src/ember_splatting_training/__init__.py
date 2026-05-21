"""Optional splatting training utilities for Ember."""

from importlib.metadata import PackageNotFoundError, version

try:
    from ember_splatting_training._version import __version__
except ImportError:
    try:
        __version__ = version("ember-splatting-training")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "COMPACT_TRAINING_METRIC_NAMES",
    "TRAINING_VIEW_RENDER_MODES",
    "TRAINING_VIEW_RENDER_MODE_LABELS",
    "FastGSDensificationRecipe",
    "FastGSFinalPruneMode",
    "FusedAdam",
    "Gaussian3DGSOptimizationRecipe",
    "GaussianFastGS",
    "GaussianMCMC",
    "GaussianMipSplatting3DFilter",
    "GaussianMortonOrdering",
    "TrainingPreparationErrorRows",
    "TrainingPreparationHandle",
    "TrainingPreparationSnapshot",
    "TrainingStatusInfoRows",
    "TrainingViewDepthRangeMode",
    "TrainingViewInspection",
    "TrainingViewInspector",
    "TrainingViewInspectorConfig",
    "TrainingViewInspectorControls",
    "TrainingViewMapContext",
    "TrainingViewMapResult",
    "TrainingViewMapSpec",
    "TrainingViewPreview",
    "TrainingViewRenderMode",
    "TrainingViewerConfig",
    "TrainingViewerErrorMap",
    "TrainingViewerHandle",
    "TrainingViewerHook",
    "TrainingViewerSnapshot",
    "TrainingViserViewerConfig",
    "active_sh_bases_for_step",
    "add_noise",
    "available_training_view_render_modes",
    "checkpoint_logs_dir",
    "create_training_preparation",
    "create_training_run",
    "create_training_view_inspector",
    "create_training_viewer",
    "crop_scalar_map_to",
    "dssim_loss",
    "empty_scalar_frame",
    "escape_markdown_table_cell",
    "fastergs_training_backend_options",
    "fastgs_l1_metric_map",
    "fastgs_normalize_score",
    "filter_scalars",
    "find_event_files",
    "format_training_duration",
    "format_training_metric_parts",
    "format_training_status",
    "format_training_status_info_table",
    "format_training_status_info_value",
    "gaussian_3dgs_optimization_config",
    "gaussian_3dgs_parameter_groups",
    "image_loss_error_map",
    "image_loss_label_for_terms",
    "morton_codes",
    "morton_order",
    "normalize_training_status_info_rows",
    "psnr_map",
    "read_scalar_records",
    "read_scalars",
    "relocation_adjustment",
    "render_training_preparation_status",
    "render_training_status_panel",
    "render_training_status_panel_from_handle",
    "render_training_view_inspector",
    "rgb_l1_dssim_loss",
    "scalar_line_chart",
    "scalar_tags",
    "select_training_preparation_error",
    "snapshot_training_viewer",
    "ssim_score",
    "training_config_for_notebook_thread",
    "training_inspector_spinner",
    "training_preparation_outputs",
    "training_status_snapshot_from_result",
    "training_view_image_loss_terms",
    "training_view_render_mode_options",
    "training_view_render_modes_for_config",
    "viridis_error_map",
]


def __getattr__(name: str) -> object:
    """Load optional FasterGS-backed exports only when requested."""
    match name:
        case "FusedAdam":
            from ember_splatting_training.optim import FusedAdam

            return FusedAdam
        case "GaussianMCMC" | "add_noise" | "relocation_adjustment":
            from ember_splatting_training import densification

            return getattr(densification, name)
        case (
            "FastGSFinalPruneMode"
            | "GaussianFastGS"
            | "GaussianMipSplatting3DFilter"
            | "GaussianMortonOrdering"
            | "active_sh_bases_for_step"
            | "fastgs_l1_metric_map"
            | "fastgs_normalize_score"
            | "fastergs_training_backend_options"
            | "morton_codes"
            | "morton_order"
        ):
            from ember_splatting_training import fastergs

            return getattr(fastergs, name)
        case "dssim_loss" | "rgb_l1_dssim_loss" | "ssim_score":
            from ember_splatting_training import losses

            return getattr(losses, name)
        case (
            "COMPACT_TRAINING_METRIC_NAMES"
            | "escape_markdown_table_cell"
            | "format_training_duration"
            | "format_training_metric_parts"
            | "format_training_status"
            | "format_training_status_info_table"
            | "format_training_status_info_value"
            | "normalize_training_status_info_rows"
            | "TrainingViewerConfig"
            | "TrainingPreparationErrorRows"
            | "TrainingStatusInfoRows"
            | "TRAINING_VIEW_RENDER_MODE_LABELS"
            | "TRAINING_VIEW_RENDER_MODES"
            | "TrainingPreparationHandle"
            | "TrainingPreparationSnapshot"
            | "TrainingViewDepthRangeMode"
            | "TrainingViewInspection"
            | "TrainingViewInspector"
            | "TrainingViewInspectorConfig"
            | "TrainingViewInspectorControls"
            | "TrainingViewMapContext"
            | "TrainingViewMapResult"
            | "TrainingViewMapSpec"
            | "TrainingViewPreview"
            | "TrainingViewRenderMode"
            | "TrainingViewerErrorMap"
            | "TrainingViewerHandle"
            | "TrainingViewerHook"
            | "TrainingViewerSnapshot"
            | "TrainingViserViewerConfig"
            | "available_training_view_render_modes"
            | "create_training_preparation"
            | "create_training_run"
            | "create_training_view_inspector"
            | "create_training_viewer"
            | "crop_scalar_map_to"
            | "image_loss_error_map"
            | "image_loss_label_for_terms"
            | "psnr_map"
            | "render_training_view_inspector"
            | "render_training_preparation_status"
            | "render_training_status_panel"
            | "render_training_status_panel_from_handle"
            | "select_training_preparation_error"
            | "snapshot_training_viewer"
            | "ssim_error_map"
            | "training_inspector_spinner"
            | "training_config_for_notebook_thread"
            | "training_preparation_outputs"
            | "training_status_snapshot_from_result"
            | "training_view_image_loss_terms"
            | "training_view_render_mode_options"
            | "training_view_render_modes_for_config"
            | "viridis_error_map"
        ):
            from ember_splatting_training import training_viewer

            return getattr(training_viewer, name)
        case (
            "checkpoint_logs_dir"
            | "empty_scalar_frame"
            | "filter_scalars"
            | "find_event_files"
            | "read_scalar_records"
            | "read_scalars"
            | "scalar_line_chart"
            | "scalar_tags"
        ):
            from ember_splatting_training import tensorboard_analysis

            return getattr(tensorboard_analysis, name)
        case "FastGSDensificationRecipe":
            from ember_splatting_training import typed_recipes

            return getattr(typed_recipes, name)
        case (
            "Gaussian3DGSOptimizationRecipe"
            | "gaussian_3dgs_optimization_config"
            | "gaussian_3dgs_parameter_groups"
        ):
            from ember_splatting_training import recipes

            return getattr(recipes, name)
        case _:
            raise AttributeError(name)
