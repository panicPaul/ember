"""Checkpoint-local TensorBoard scalar logging for training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ember_core.training.config import TrainingLoggingConfig

_ALWAYS_LOGGED_METRICS = (
    "elapsed_seconds",
    "iterations_per_second",
    "step_seconds",
)
_MAX_HISTOGRAM_VALUES = 1_000_000


def checkpoint_log_dir(checkpoint_dir: str | Path) -> Path:
    """Return the canonical TensorBoard log directory for a checkpoint."""
    return Path(checkpoint_dir).expanduser() / "logs"


class TensorBoardTrainingLogger:
    """Low-overhead scalar writer for one checkpoint directory."""

    def __init__(
        self,
        config: TrainingLoggingConfig,
        *,
        checkpoint_dir: str | Path,
    ) -> None:
        self.config = config
        self.log_dir = checkpoint_log_dir(checkpoint_dir)
        self._writer: Any | None = None
        if config.enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self.log_dir))

    @property
    def enabled(self) -> bool:
        """Return whether this logger writes events."""
        return self._writer is not None

    def write_step(
        self,
        step: int,
        metrics: dict[str, float],
        *,
        histograms: Mapping[str, object] | None = None,
    ) -> None:
        """Write scalar metrics and optional histograms for one step."""
        if self._writer is None:
            return
        if step % self.config.log_every != 0 and not _has_refinement(metrics):
            for name in _ALWAYS_LOGGED_METRICS:
                _write_named_scalar(self._writer, step, name, metrics.get(name))
            return
        for name, value in sorted(metrics.items()):
            _write_named_scalar(self._writer, step, name, value)
        for name, values in sorted(dict(histograms or {}).items()):
            _write_named_histogram(self._writer, step, name, values)

    def close(self) -> None:
        """Flush and close the underlying TensorBoard writer."""
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None


def build_training_logger(
    config: TrainingLoggingConfig,
    *,
    checkpoint_dir: str | Path,
) -> TensorBoardTrainingLogger | None:
    """Build the checkpoint-local training logger when enabled."""
    if not config.enabled:
        return None
    return TensorBoardTrainingLogger(config, checkpoint_dir=checkpoint_dir)


def scalar_tag_for_metric(name: str) -> str:
    """Map internal metric names to stable TensorBoard scalar tags."""
    match name:
        case "loss":
            return "train/loss"
        case "iterations_per_second":
            return "train/iterations_per_second"
        case "primitives":
            return "train/primitives"
        case "elapsed_seconds" | "step_seconds":
            return f"time/{name}"
        case "l1" | "dssim" | "ssim_loss":
            return f"loss/{name}"
        case _ if name.endswith("_reg"):
            return f"loss/{name}"
        case _ if name.startswith("degeneracy_"):
            return f"geometry/degeneracy/{name.removeprefix('degeneracy_')}"
        case _ if name.endswith("_regularization"):
            return f"loss/{name}"
        case _ if name.startswith("render_hit_count_"):
            suffix = name.removeprefix("render_hit_count_")
            return f"render/contributors/{suffix}"
        case _ if name.startswith("render_overflow_count_"):
            suffix = name.removeprefix("render_overflow_count_")
            return f"render/overflow/{suffix}"
        case _ if name.startswith("render_alpha_"):
            return f"render/alpha/{name.removeprefix('render_alpha_')}"
        case _ if name.startswith("render_feature_"):
            return f"render/feature/{name.removeprefix('render_feature_')}"
        case _ if name.startswith("render_rgb_"):
            return f"render/rgb/{name.removeprefix('render_rgb_')}"
        case _ if name.startswith("scene_"):
            return f"scene/{name.removeprefix('scene_')}"
        case _ if name.startswith("mcmc_"):
            return f"densification/mcmc/{name.removeprefix('mcmc_')}"
        case _ if name.startswith("fastgs_"):
            return f"densification/fastgs/{name.removeprefix('fastgs_')}"
        case _ if name.startswith("collapse_"):
            return f"diagnostics/collapse/{name.removeprefix('collapse_')}"
        case _ if name.startswith("shader_"):
            return f"shader/{name.removeprefix('shader_')}"
        case _ if name.startswith("time_") and name.endswith("_ms"):
            phase = name.removeprefix("time_").removesuffix("_ms")
            return f"time/{phase}_ms"
        case _ if name.startswith("cuda_"):
            return f"cuda/{name.removeprefix('cuda_')}"
        case _ if name.startswith("refinement_"):
            return f"densification/{name.removeprefix('refinement_')}"
        case _:
            return f"metrics/{name}"


def histogram_tag_for_metric(name: str) -> str:
    """Map internal histogram names to stable TensorBoard tags."""
    match name:
        case "render_hit_count":
            return "render/contributors/hit_count"
        case "render_overflow_count":
            return "render/overflow/count"
        case _:
            return scalar_tag_for_metric(name)


def _is_finite_scalar(value: object) -> bool:
    if not isinstance(value, int | float):
        return False
    return math.isfinite(float(value))


def _has_refinement(metrics: dict[str, float]) -> bool:
    return any(name.startswith("refinement_") for name in metrics)


def _write_named_scalar(
    writer: Any,
    step: int,
    name: str,
    value: object,
) -> None:
    if name == "step" or not _is_finite_scalar(value):
        return
    writer.add_scalar(scalar_tag_for_metric(name), float(value), global_step=step)


def _write_named_histogram(
    writer: Any,
    step: int,
    name: str,
    values: object,
) -> None:
    histogram_values = _histogram_values(values)
    if histogram_values is None:
        return
    writer.add_histogram(
        histogram_tag_for_metric(name),
        histogram_values,
        global_step=step,
    )


def _histogram_values(values: object) -> Tensor | None:
    if not isinstance(values, Tensor):
        return None
    finite_values = values.detach().reshape(-1).to(torch.float32)
    finite_values = finite_values[torch.isfinite(finite_values)]
    if int(finite_values.numel()) == 0:
        return None
    if int(finite_values.numel()) > _MAX_HISTOGRAM_VALUES:
        indices = torch.linspace(
            0,
            int(finite_values.numel()) - 1,
            _MAX_HISTOGRAM_VALUES,
            dtype=torch.long,
            device=finite_values.device,
        )
        finite_values = finite_values[indices]
    return finite_values.cpu()


__all__ = [
    "TensorBoardTrainingLogger",
    "build_training_logger",
    "checkpoint_log_dir",
    "histogram_tag_for_metric",
    "scalar_tag_for_metric",
]
