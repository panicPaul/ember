"""Shared on-disk resized-image cache utilities."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Literal

import torch
from tqdm.auto import tqdm

from ember_core.core.contracts import CameraState
from ember_core.data.config_contracts import (
    ImagePreparationConfig,
    PreparedFrameDatasetConfig,
    ResizedImageCacheConfig,
)
from ember_core.data.contracts import (
    CameraSensorDataset,
    DatasetFrame,
    PathCameraImageSource,
    ResizeSpec,
    SceneRecord,
    horizontal_fov_degrees,
)
from ember_core.data.preprocess import resize_intrinsics, resolve_resize_shape

ImageInterpolationMode = Literal["nearest", "bilinear", "bicubic"]


def resized_image_cache_directory_name(
    resize: ResizeSpec,
) -> str:
    """Return the deterministic cache directory name for a resize spec."""
    if resize.width_target is not None:
        return f"width_{resize.width_target}_{resize.interpolation}"
    assert resize.width_scale is not None
    scale_name = f"{resize.width_scale:.6f}".rstrip("0").rstrip(".")
    scale_name = scale_name.replace(".", "p")
    return f"scale_{scale_name}_{resize.interpolation}"


def resized_image_cache_parent(
    *,
    scene_record: SceneRecord,
    image_source_root: Path,
    cache_config: ResizedImageCacheConfig,
) -> Path:
    """Return the parent directory for resized image cache variants."""
    if cache_config.cache_root is not None:
        return cache_config.cache_root.expanduser()
    if scene_record.root_path is not None:
        return scene_record.root_path / "ember_cache" / "resized_images"
    return image_source_root / "ember_cache" / "resized_images"


def resized_image_cache_root(
    *,
    scene_record: SceneRecord,
    image_source_root: Path,
    cache_config: ResizedImageCacheConfig,
    resize: ResizeSpec,
) -> Path:
    """Return the active resized image cache directory."""
    return resized_image_cache_parent(
        scene_record=scene_record,
        image_source_root=image_source_root,
        cache_config=cache_config,
    ) / resized_image_cache_directory_name(resize)


def enforce_resized_image_cache_limit(
    *,
    active_cache_root: Path,
    max_caches: int,
) -> None:
    """Keep only a bounded number of reusable resized image caches."""
    parent = active_cache_root.parent
    if not parent.exists():
        return
    cache_dirs = [
        path
        for path in parent.iterdir()
        if path.is_dir()
        and (path.name.startswith("scale_") or path.name.startswith("width_"))
    ]
    overflow = len(cache_dirs) - max_caches
    if overflow <= 0:
        return
    evictable = sorted(
        (path for path in cache_dirs if path != active_cache_root),
        key=lambda path: path.stat().st_mtime,
    )
    for stale_cache in evictable[:overflow]:
        shutil.rmtree(stale_cache)


def pillow_resampling(
    interpolation: ImageInterpolationMode,
) -> object:
    """Translate Ember interpolation names to Pillow resampling filters."""
    from PIL import Image

    if interpolation == "nearest":
        return Image.Resampling.NEAREST
    if interpolation == "bilinear":
        return Image.Resampling.BILINEAR
    if interpolation == "bicubic":
        return Image.Resampling.BICUBIC
    raise ValueError(f"Unsupported interpolation mode {interpolation!r}.")


def path_image_source_root(
    image_source: PathCameraImageSource,
) -> Path:
    """Return the common path root for a path-backed image source."""
    paths = tuple(image_source.frame_paths.values())
    if not paths:
        raise ValueError("Path-backed image source does not contain images.")
    if len(paths) == 1:
        return paths[0].parent
    return Path(os.path.commonpath([str(path) for path in paths]))


def materialize_resized_image_cache(
    *,
    source_root: Path,
    cache_root: Path,
    resize: ResizeSpec,
    max_caches: int,
    source_paths: Sequence[Path] | None = None,
    num_workers: int = 8,
    progress: bool = True,
) -> Path:
    """Create or update a resized image cache from full-resolution images."""
    from PIL import Image

    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    resolved_source_paths = (
        tuple(sorted(source_paths))
        if source_paths is not None
        else tuple(
            sorted(
                path
                for path in source_root.rglob("*")
                if path.is_file() and path.suffix.lower() in image_suffixes
            )
        )
    )
    if not resolved_source_paths:
        raise ValueError(f"No source images found under {source_root}.")
    resampling = pillow_resampling(resize.interpolation)
    enforce_resized_image_cache_limit(
        active_cache_root=cache_root,
        max_caches=max_caches,
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    def resize_one(source_path: Path) -> None:
        relative_path = source_path.relative_to(source_root)
        target_path = cache_root / relative_path
        if (
            target_path.exists()
            and target_path.stat().st_mtime >= source_path.stat().st_mtime
        ):
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            resized_width, resized_height = resolve_resize_shape(
                width,
                height,
                resize,
            )
            resized = rgb.resize(
                (resized_width, resized_height),
                resampling,
            )
            save_kwargs = (
                {"quality": 95}
                if target_path.suffix.lower() in {".jpg", ".jpeg"}
                else {}
            )
            resized.save(target_path, **save_kwargs)

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = [executor.submit(resize_one, path) for path in resolved_source_paths]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Preparing resized image cache",
            disable=not progress,
        ):
            future.result()

    metadata = {
        "source_root": str(source_root),
        "interpolation": resize.interpolation,
        "num_images": len(resolved_source_paths),
    }
    if resize.width_scale is not None:
        metadata["scale"] = resize.width_scale
    if resize.width_target is not None:
        metadata["width_target"] = resize.width_target
    (cache_root / "cache_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    cache_root.touch()
    enforce_resized_image_cache_limit(
        active_cache_root=cache_root,
        max_caches=max_caches,
    )
    return cache_root


def image_preparation_resize_spec(
    image_preparation: ImagePreparationConfig | None,
) -> ResizeSpec | None:
    """Return the resize spec requested by an image preparation config."""
    if image_preparation is None:
        return None
    if (
        image_preparation.resize_width_scale is None
        and image_preparation.resize_width_target is None
    ):
        return None
    return ResizeSpec(
        width_scale=image_preparation.resize_width_scale,
        width_target=image_preparation.resize_width_target,
        interpolation=image_preparation.interpolation,
    )


def resize_changes_camera_sensor(
    camera_sensor: CameraSensorDataset,
    resize: ResizeSpec,
) -> bool:
    """Return whether a resize spec changes any camera frame dimensions."""
    return any(
        resolve_resize_shape(frame.width, frame.height, resize)
        != (frame.width, frame.height)
        for frame in camera_sensor.frames
    )


def cached_frame_path(
    *,
    source_path: Path,
    source_root: Path,
    cache_root: Path,
) -> Path:
    """Return the cache path corresponding to one source image path."""
    return cache_root / source_path.relative_to(source_root)


def resized_camera_state(
    *,
    camera_sensor: CameraSensorDataset,
    resize: ResizeSpec,
) -> tuple[tuple[DatasetFrame, ...], CameraState]:
    """Return frame metadata and camera tensors for resized images."""
    camera_intrinsics = camera_sensor.camera.get_intrinsics()
    resized_frames: list[DatasetFrame] = []
    resized_widths: list[int] = []
    resized_heights: list[int] = []
    resized_fovs: list[float] = []
    resized_intrinsics: list[torch.Tensor] = []
    for frame in camera_sensor.frames:
        resized_width, resized_height = resolve_resize_shape(
            frame.width,
            frame.height,
            resize,
        )
        intrinsics = camera_intrinsics[frame.camera_index]
        scaled_intrinsics = resize_intrinsics(
            intrinsics,
            original_width=frame.width,
            original_height=frame.height,
            resized_width=resized_width,
            resized_height=resized_height,
        )
        resized_frames.append(
            replace(frame, width=resized_width, height=resized_height)
        )
        resized_widths.append(resized_width)
        resized_heights.append(resized_height)
        resized_fovs.append(horizontal_fov_degrees(resized_width, scaled_intrinsics))
        resized_intrinsics.append(scaled_intrinsics)

    resized_camera = replace(
        camera_sensor.camera,
        width=torch.tensor(
            resized_widths,
            dtype=camera_sensor.camera.width.dtype,
            device=camera_sensor.camera.width.device,
        ),
        height=torch.tensor(
            resized_heights,
            dtype=camera_sensor.camera.height.dtype,
            device=camera_sensor.camera.height.device,
        ),
        fov_degrees=torch.tensor(
            resized_fovs,
            dtype=camera_sensor.camera.fov_degrees.dtype,
            device=camera_sensor.camera.fov_degrees.device,
        ),
        intrinsics=torch.stack(resized_intrinsics, dim=0).to(
            camera_sensor.camera.get_intrinsics().device
        ),
    )
    return tuple(resized_frames), resized_camera


def camera_sensor_with_resized_cache(
    *,
    camera_sensor: CameraSensorDataset,
    source_root: Path,
    cache_root: Path,
    resize: ResizeSpec,
) -> CameraSensorDataset:
    """Return a camera sensor whose paths and metadata point at a resize cache."""
    if not isinstance(camera_sensor.image_source, PathCameraImageSource):
        raise TypeError("Resized image caches require PathCameraImageSource.")
    resized_frames, resized_camera = resized_camera_state(
        camera_sensor=camera_sensor,
        resize=resize,
    )
    cached_paths = {
        frame_id: cached_frame_path(
            source_path=source_path,
            source_root=source_root,
            cache_root=cache_root,
        )
        for frame_id, source_path in camera_sensor.image_source.frame_paths.items()
    }
    return replace(
        camera_sensor,
        frames=resized_frames,
        timestamps_us=tuple(frame.timestamp_us for frame in resized_frames),
        camera=resized_camera,
        image_source=PathCameraImageSource(frame_paths=cached_paths),
    )


def scene_record_with_camera_sensor(
    scene_record: SceneRecord,
    camera_sensor: CameraSensorDataset,
) -> SceneRecord:
    """Return a scene record with one camera sensor replaced."""
    sensors = tuple(
        camera_sensor if sensor.sensor_id == camera_sensor.sensor_id else sensor
        for sensor in scene_record.sensors
    )
    return replace(scene_record, sensors=sensors)


def prepared_config_without_resize(
    config: PreparedFrameDatasetConfig,
) -> PreparedFrameDatasetConfig:
    """Return a dataset config whose preparation no longer performs resize."""
    image_preparation = config.image_preparation
    if image_preparation is None:
        return config
    return config.model_copy(
        update={
            "image_preparation": image_preparation.model_copy(
                update={
                    "resize_width_scale": None,
                    "resize_width_target": None,
                }
            )
        }
    )


def resolve_resized_image_cache_for_dataset(
    *,
    scene_record: SceneRecord,
    config: PreparedFrameDatasetConfig | None,
) -> tuple[SceneRecord, PreparedFrameDatasetConfig | None]:
    """Apply the configured resized-image cache before dataset materialization."""
    if config is None or config.image_preparation is None:
        return scene_record, config
    resize = image_preparation_resize_spec(config.image_preparation)
    cache_config = config.image_preparation.resized_image_cache
    if resize is None or cache_config is None or not cache_config.enabled:
        return scene_record, config

    camera_sensor = scene_record.resolve_camera_sensor(config.camera_sensor_id)
    if not isinstance(camera_sensor.image_source, PathCameraImageSource):
        return scene_record, config
    if not resize_changes_camera_sensor(camera_sensor, resize):
        return scene_record, config

    source_root = path_image_source_root(camera_sensor.image_source)
    cache_root = resized_image_cache_root(
        scene_record=scene_record,
        image_source_root=source_root,
        cache_config=cache_config,
        resize=resize,
    )
    source_paths = tuple(camera_sensor.image_source.frame_paths.values())
    # Build the derived image tree before rewriting camera metadata to it.
    materialize_resized_image_cache(
        source_root=source_root,
        cache_root=cache_root,
        resize=resize,
        max_caches=cache_config.max_caches,
        source_paths=source_paths,
    )
    resized_sensor = camera_sensor_with_resized_cache(
        camera_sensor=camera_sensor,
        source_root=source_root,
        cache_root=cache_root,
        resize=resize,
    )
    return (
        scene_record_with_camera_sensor(scene_record, resized_sensor),
        prepared_config_without_resize(config),
    )


__all__ = [
    "cached_frame_path",
    "camera_sensor_with_resized_cache",
    "enforce_resized_image_cache_limit",
    "image_preparation_resize_spec",
    "materialize_resized_image_cache",
    "path_image_source_root",
    "pillow_resampling",
    "prepared_config_without_resize",
    "resize_changes_camera_sensor",
    "resized_camera_state",
    "resized_image_cache_directory_name",
    "resized_image_cache_parent",
    "resized_image_cache_root",
    "resolve_resized_image_cache_for_dataset",
    "scene_record_with_camera_sensor",
]
