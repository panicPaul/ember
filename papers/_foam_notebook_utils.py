"""Shared notebook utilities for foam-family paper implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import ember_core as ember

InterpolationMode = Literal["nearest", "bilinear", "bicubic"]
SplitTarget = Literal["train", "val", "all"]
MaterializationStage = Literal["none", "decoded", "prepared"]
MaterializationMode = Literal["lazy", "eager"]


class FoamSceneConfig(Protocol):
    """Scene fields shared by foam paper notebooks."""

    path: Path
    image_root: Path | None
    undistort_output_dir: Path | None
    align_horizon: bool


class FoamDataConfig(Protocol):
    """Dataset fields shared by foam paper notebooks."""

    camera_sensor_id: str | None
    image_scale_factor: float
    cache_resized_images: bool
    resized_image_cache_root: Path | None
    max_resized_image_caches: int
    split_target: SplitTarget
    split_every_n: int | None
    materialization_stage: MaterializationStage
    materialization_mode: MaterializationMode
    materialization_num_workers: int | None
    normalize_images: bool
    interpolation: InterpolationMode


class FoamExperimentConfig(Protocol):
    """Experiment fields shared by foam paper notebooks."""

    scene: FoamSceneConfig
    data: FoamDataConfig


def build_foam_scene_load_config(
    config: FoamExperimentConfig,
) -> ember.ColmapSceneConfig:
    """Translate a foam paper config into an Ember scene loader config."""
    source_pipes = (
        (ember.HorizonAlignPipeConfig(),) if config.scene.align_horizon else ()
    )
    image_root = (
        config.scene.image_root.expanduser()
        if config.scene.image_root is not None
        else None
    )
    return ember.ColmapSceneConfig(
        path=config.scene.path.expanduser(),
        image_root=image_root,
        undistort_output_dir=(
            config.scene.undistort_output_dir.expanduser()
            if config.scene.undistort_output_dir is not None
            else None
        ),
        source_pipes=source_pipes,
    )


def build_foam_prepared_frame_dataset_config(
    config: FoamExperimentConfig,
) -> ember.PreparedFrameDatasetConfig:
    """Translate a foam paper config into an Ember frame dataset config."""
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
            resized_image_cache=ember.ResizedImageCacheConfig(
                enabled=config.data.cache_resized_images,
                cache_root=config.data.resized_image_cache_root,
                max_caches=config.data.max_resized_image_caches,
            ),
        ),
    )
