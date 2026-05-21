"""Typed runtime outputs for the Triangle Splatting native runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from jaxtyping import Float, Int
from torch import Tensor


@dataclass(frozen=True)
class TriangleRasterizationResult:
    """Output of the full Triangle Splatting rasterization stage."""

    rendered_count: Int[Tensor, " 1"]
    image: Float[Tensor, " 3 height width"]
    auxiliary_image: Float[Tensor, " 7 height width"]
    radii: Int[Tensor, " num_triangles"]
    geometry_buffer: Tensor
    binning_buffer: Tensor
    image_buffer: Tensor
    screen_space_scale: Float[Tensor, " num_triangles"]
    density_factor: Float[Tensor, " num_triangles"]
    max_blending: Float[Tensor, " num_triangles"]

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build a rasterization result from the raw op outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, ...]:
        """Return the raw tensor tuple for custom-op composition."""
        return (
            self.rendered_count,
            self.image,
            self.auxiliary_image,
            self.radii,
            self.geometry_buffer,
            self.binning_buffer,
            self.image_buffer,
            self.screen_space_scale,
            self.density_factor,
            self.max_blending,
        )
