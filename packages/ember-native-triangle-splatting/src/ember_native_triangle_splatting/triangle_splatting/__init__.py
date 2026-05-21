"""Ember-native Triangle Splatting backend."""

from ember_native_triangle_splatting.triangle_splatting.renderer import (
    TriangleSplattingNativeRenderOptions,
    TriangleSplattingNativeRenderOutput,
    register,
    render_triangle_splatting_native,
)
from ember_native_triangle_splatting.triangle_splatting.runtime import (
    TriangleRasterizationResult,
    compute_relocation_op,
    mark_visible_op,
    rasterize_triangles,
)
from ember_native_triangle_splatting.triangle_splatting.training import (
    TriangleSplattingDensification,
    TriangleSplattingFamilyOps,
    TriangleSplattingOptimizationRecipe,
    camera_depth_normals,
    equilateral_area,
    generate_triangle_vertices,
    initialize_triangle_splatting_model_from_scene_record,
    initialize_triangle_splatting_scene_from_scene_record,
    register_triangle_splatting_family_ops,
    triangle_splatting_is_large_scene,
    triangle_splatting_loss,
    triangle_splatting_optimization_config,
    triangle_splatting_parameter_groups,
    triangle_splatting_rgb_to_sh,
    triangle_splatting_root_mean_squared_knn_distances,
)

__all__ = [
    "TriangleRasterizationResult",
    "TriangleSplattingDensification",
    "TriangleSplattingFamilyOps",
    "TriangleSplattingNativeRenderOptions",
    "TriangleSplattingNativeRenderOutput",
    "TriangleSplattingOptimizationRecipe",
    "camera_depth_normals",
    "compute_relocation_op",
    "equilateral_area",
    "generate_triangle_vertices",
    "initialize_triangle_splatting_model_from_scene_record",
    "initialize_triangle_splatting_scene_from_scene_record",
    "mark_visible_op",
    "rasterize_triangles",
    "register",
    "register_triangle_splatting_family_ops",
    "render_triangle_splatting_native",
    "triangle_splatting_is_large_scene",
    "triangle_splatting_loss",
    "triangle_splatting_optimization_config",
    "triangle_splatting_parameter_groups",
    "triangle_splatting_rgb_to_sh",
    "triangle_splatting_root_mean_squared_knn_distances",
]
