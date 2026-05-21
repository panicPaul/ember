"""Training helpers for Triangle Splatting scenes."""

from ember_native_triangle_splatting.triangle_splatting.training._impl import (
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
    "TriangleSplattingDensification",
    "TriangleSplattingFamilyOps",
    "TriangleSplattingOptimizationRecipe",
    "camera_depth_normals",
    "equilateral_area",
    "generate_triangle_vertices",
    "initialize_triangle_splatting_model_from_scene_record",
    "initialize_triangle_splatting_scene_from_scene_record",
    "register_triangle_splatting_family_ops",
    "triangle_splatting_is_large_scene",
    "triangle_splatting_loss",
    "triangle_splatting_optimization_config",
    "triangle_splatting_parameter_groups",
    "triangle_splatting_rgb_to_sh",
    "triangle_splatting_root_mean_squared_knn_distances",
]
