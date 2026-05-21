"""Public runtime API for the Triangle Splatting native backend."""

from ember_native_triangle_splatting.triangle_splatting.runtime.ops import (
    TriangleRasterizationResult,
    compute_relocation_op,
    mark_visible_op,
    rasterize_triangles,
    rasterize_triangles_bwd_op,
    rasterize_triangles_fwd_op,
    rasterize_triangles_op,
)

__all__ = [
    "TriangleRasterizationResult",
    "compute_relocation_op",
    "mark_visible_op",
    "rasterize_triangles",
    "rasterize_triangles_bwd_op",
    "rasterize_triangles_fwd_op",
    "rasterize_triangles_op",
]
