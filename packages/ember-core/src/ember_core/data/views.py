"""Reusable prepared-frame view catalogs for notebook viewers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ember_core.core.contracts import CameraState
from ember_core.data.adapters import PreparedFrameCache, PreparedFrameDataset
from ember_core.data.config_contracts import (
    MaterializationConfig,
    PreparedFrameDatasetConfig,
    SplitConfig,
)
from ember_core.data.contracts import PreparedFrameSample, SceneRecord
from ember_core.data.resized_cache import (
    resolve_resized_image_cache_for_dataset,
)

PreparedFrameViewSplit = Literal["train", "val"]


@dataclass(frozen=True)
class PreparedFrameViewRef:
    """Stable reference to one prepared frame view."""

    split: PreparedFrameViewSplit
    index: int
    source_index: int
    frame_id: str
    label: str

    @property
    def key(self) -> str:
        """Return a compact stable key for UI state and callbacks."""
        return f"{self.split}:{self.index}:{self.source_index}"


class PreparedFrameViewCatalog:
    """Train/validation view catalog backed by one shared sample cache."""

    def __init__(
        self,
        scene_record: SceneRecord,
        *,
        config: PreparedFrameDatasetConfig | None = None,
        training_split: PreparedFrameViewSplit = "train",
    ) -> None:
        """Create train and validation views over a shared prepared cache."""
        self.scene_record, self.config = resolve_resized_image_cache_for_dataset(
            scene_record=scene_record,
            config=config or PreparedFrameDatasetConfig(),
        )
        self.training_split = training_split
        self.sample_cache = PreparedFrameCache(
            self.scene_record,
            config=self.config,
        )
        self.datasets_by_split = {
            split: self.build_dataset(
                split,
                lazy=split != self.training_split,
            )
            for split in ("train", "val")
        }
        self.views_by_split = {
            "train": self.build_views("train"),
            "val": self.build_views("val"),
        }

    @property
    def training_dataset(self) -> PreparedFrameDataset:
        """Return the dataset intended for training."""
        return self.dataset(self.training_split)

    def dataset(
        self,
        split: PreparedFrameViewSplit,
    ) -> PreparedFrameDataset:
        """Return the prepared dataset for a split."""
        return self.datasets_by_split[split]

    def views(
        self,
        split: PreparedFrameViewSplit,
    ) -> tuple[PreparedFrameViewRef, ...]:
        """Return the selectable prepared views for a split."""
        return self.views_by_split[split]

    def view_options(
        self,
        split: PreparedFrameViewSplit,
    ) -> dict[str, PreparedFrameViewRef]:
        """Return dropdown-friendly view labels mapped to references."""
        return {view.label: view for view in self.views(split)}

    def view_key_options(
        self,
        split: PreparedFrameViewSplit,
    ) -> dict[str, str]:
        """Return dropdown-friendly view labels mapped to stable keys."""
        return {view.label: view.key for view in self.views(split)}

    def view_ref_by_key(
        self,
        key: str | None,
    ) -> PreparedFrameViewRef | None:
        """Resolve a stable UI key back to a current view reference."""
        if key is None or key == "":
            return None
        try:
            split, index_text, source_index_text = key.split(":", maxsplit=2)
            index = int(index_text)
            source_index = int(source_index_text)
        except ValueError:
            return None
        if split not in ("train", "val"):
            return None
        views = self.views(split)
        if not 0 <= index < len(views):
            return None
        view_ref = views[index]
        if view_ref.source_index != source_index:
            return None
        return view_ref

    def camera(self, view_ref: PreparedFrameViewRef) -> CameraState:
        """Return the prepared camera for a view without loading RGB."""
        self.validate_view_ref(view_ref)
        return self.dataset(view_ref.split).prepared_camera(view_ref.index)

    def sample(self, view_ref: PreparedFrameViewRef) -> PreparedFrameSample:
        """Return the prepared sample for a view."""
        self.validate_view_ref(view_ref)
        return self.dataset(view_ref.split)[view_ref.index]

    def build_dataset(
        self,
        split: PreparedFrameViewSplit,
        *,
        lazy: bool,
    ) -> PreparedFrameDataset:
        """Build the prepared dataset for one split."""
        return PreparedFrameDataset(
            self.scene_record,
            config=self.config_for_split(split, lazy=lazy),
            sample_cache=self.sample_cache,
        )

    def config_for_split(
        self,
        split: PreparedFrameViewSplit,
        *,
        lazy: bool,
    ) -> PreparedFrameDatasetConfig:
        """Return this catalog's dataset config specialized for one split."""
        materialization = self.config.materialization
        if lazy and materialization is not None:
            materialization = materialization.model_copy(
                update={"mode": "lazy"}
            )
        elif lazy and materialization is None:
            materialization = MaterializationConfig(mode="lazy")
        return self.config.model_copy(
            update={
                "split": split_config_for_target(self.config.split, split),
                "materialization": materialization,
            }
        )

    def build_views(
        self,
        split: PreparedFrameViewSplit,
    ) -> tuple[PreparedFrameViewRef, ...]:
        """Build stable view references for one dataset split."""
        dataset = self.dataset(split)
        return tuple(
            PreparedFrameViewRef(
                split=split,
                index=index,
                source_index=source_index,
                frame_id=dataset.camera_stream.frames[source_index].frame_id,
                label=view_label(
                    split,
                    index=index,
                    source_index=source_index,
                    frame_id=dataset.camera_stream.frames[
                        source_index
                    ].frame_id,
                ),
            )
            for index, source_index in enumerate(dataset.indices)
        )

    def validate_view_ref(self, view_ref: PreparedFrameViewRef) -> None:
        """Raise when a view reference no longer matches this catalog."""
        views = self.views(view_ref.split)
        if not 0 <= view_ref.index < len(views):
            raise IndexError(
                f"View index {view_ref.index} is out of range for "
                f"{view_ref.split!r}."
            )
        current = views[view_ref.index]
        if current.source_index != view_ref.source_index:
            raise ValueError("Prepared frame view reference is stale.")


def split_config_for_target(
    split: SplitConfig | None,
    target: PreparedFrameViewSplit,
) -> SplitConfig | None:
    """Return a split config pinned to the requested target split."""
    if split is None:
        return None
    if split.target == "all" or split.mode == "none":
        return split
    return split.model_copy(update={"target": target})


def view_label(
    split: PreparedFrameViewSplit,
    *,
    index: int,
    source_index: int,
    frame_id: str,
) -> str:
    """Return a compact human-readable label for one prepared view."""
    return f"{split} {index:04d} | source {source_index:04d} | {frame_id}"


def build_prepared_frame_view_catalog(
    scene_record: SceneRecord,
    config: PreparedFrameDatasetConfig | None = None,
    *,
    training_split: PreparedFrameViewSplit = "train",
) -> PreparedFrameViewCatalog:
    """Build a shared train/validation view catalog for a scene record."""
    return PreparedFrameViewCatalog(
        scene_record,
        config=config,
        training_split=training_split,
    )


__all__ = [
    "PreparedFrameViewCatalog",
    "PreparedFrameViewRef",
    "PreparedFrameViewSplit",
    "build_prepared_frame_view_catalog",
]
