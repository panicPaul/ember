"""Triangle Splatting staged CUDA custom ops."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ember_native_triangle_splatting.triangle_splatting.runtime._extension import (
    load_extension,
)
from ember_native_triangle_splatting.triangle_splatting.runtime.types import (
    TriangleRasterizationResult,
)

TriangleRasterizeForwardOutput = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]
TriangleRasterizeBackwardOutput = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


def backend() -> Any:
    """Return the loaded Triangle Splatting CUDA extension."""
    return load_extension()


def requires_grad(*tensors: Tensor) -> bool:
    """Return whether any stage input needs gradients."""
    return any(tensor.requires_grad for tensor in tensors)


def zero_like_if_none(gradient: Tensor | None, reference: Tensor) -> Tensor:
    """Replace an optional gradient with a zero tensor matching a reference."""
    if gradient is None:
        return torch.zeros_like(reference)
    return gradient


@torch.library.custom_op(
    "triangle_splatting::rasterize_triangles_fwd",
    mutates_args=(),
)
def rasterize_triangles_fwd_op(
    background_color: Tensor,
    triangle_vertices: Tensor,
    triangle_sigma: Tensor,
    vertices_per_triangle: Tensor,
    triangle_vertex_offsets: Tensor,
    precomputed_colors: Tensor,
    triangle_opacities: Tensor,
    view_matrix: Tensor,
    projection_matrix: Tensor,
    num_triangles: int,
    tangent_fov_x: float,
    tangent_fov_y: float,
    image_height: int,
    image_width: int,
    spherical_harmonics: Tensor,
    sh_degree: int,
    camera_position: Tensor,
    prefiltered: bool,
    debug: bool,
) -> TriangleRasterizeForwardOutput:
    """Run the Triangle Splatting rasterization forward CUDA stage."""
    return backend().rasterize_triangles_fwd(
        background_color,
        triangle_vertices,
        triangle_sigma,
        vertices_per_triangle,
        triangle_vertex_offsets,
        precomputed_colors,
        triangle_opacities,
        view_matrix,
        projection_matrix,
        num_triangles,
        tangent_fov_x,
        tangent_fov_y,
        image_height,
        image_width,
        spherical_harmonics,
        sh_degree,
        camera_position,
        prefiltered,
        debug,
    )


@rasterize_triangles_fwd_op.register_fake
def rasterize_triangles_fwd_fake(
    background_color: Tensor,
    triangle_vertices: Tensor,
    triangle_sigma: Tensor,
    vertices_per_triangle: Tensor,
    triangle_vertex_offsets: Tensor,
    precomputed_colors: Tensor,
    triangle_opacities: Tensor,
    view_matrix: Tensor,
    projection_matrix: Tensor,
    num_triangles: int,
    tangent_fov_x: float,
    tangent_fov_y: float,
    image_height: int,
    image_width: int,
    spherical_harmonics: Tensor,
    sh_degree: int,
    camera_position: Tensor,
    prefiltered: bool,
    debug: bool,
) -> TriangleRasterizeForwardOutput:
    """Return fake tensors for tracing Triangle Splatting rasterization."""
    del (
        background_color,
        triangle_sigma,
        vertices_per_triangle,
        triangle_vertex_offsets,
        precomputed_colors,
        triangle_opacities,
        view_matrix,
        projection_matrix,
        tangent_fov_x,
        tangent_fov_y,
        spherical_harmonics,
        sh_degree,
        camera_position,
        prefiltered,
        debug,
    )
    device = triangle_vertices.device
    dtype = triangle_vertices.dtype
    return (
        torch.empty((1,), device=device, dtype=torch.int32),
        torch.empty((3, image_height, image_width), device=device, dtype=dtype),
        torch.empty((7, image_height, image_width), device=device, dtype=dtype),
        torch.empty((num_triangles,), device=device, dtype=torch.int32),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((num_triangles,), device=device, dtype=dtype),
        torch.empty((num_triangles,), device=device, dtype=dtype),
        torch.empty((num_triangles,), device=device, dtype=dtype),
    )


@torch.library.custom_op(
    "triangle_splatting::rasterize_triangles_bwd",
    mutates_args=(),
)
def rasterize_triangles_bwd_op(
    background_color: Tensor,
    triangle_vertices: Tensor,
    triangle_sigma: Tensor,
    vertices_per_triangle: Tensor,
    triangle_vertex_offsets: Tensor,
    triangle_radii: Tensor,
    precomputed_colors: Tensor,
    view_matrix: Tensor,
    projection_matrix: Tensor,
    num_triangles: int,
    tangent_fov_x: float,
    tangent_fov_y: float,
    grad_rendered_image: Tensor,
    grad_auxiliary_image: Tensor,
    spherical_harmonics: Tensor,
    sh_degree: int,
    camera_position: Tensor,
    geometry_buffer: Tensor,
    rendered_count: Tensor,
    binning_buffer: Tensor,
    image_buffer: Tensor,
    debug: bool,
) -> TriangleRasterizeBackwardOutput:
    """Run the Triangle Splatting rasterization backward CUDA stage."""
    return backend().rasterize_triangles_bwd(
        background_color,
        triangle_vertices,
        triangle_sigma,
        vertices_per_triangle,
        triangle_vertex_offsets,
        triangle_radii,
        precomputed_colors,
        view_matrix,
        projection_matrix,
        num_triangles,
        tangent_fov_x,
        tangent_fov_y,
        grad_rendered_image,
        grad_auxiliary_image,
        spherical_harmonics,
        sh_degree,
        camera_position,
        geometry_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
        debug,
    )


@rasterize_triangles_bwd_op.register_fake
def rasterize_triangles_bwd_fake(
    background_color: Tensor,
    triangle_vertices: Tensor,
    triangle_sigma: Tensor,
    vertices_per_triangle: Tensor,
    triangle_vertex_offsets: Tensor,
    triangle_radii: Tensor,
    precomputed_colors: Tensor,
    view_matrix: Tensor,
    projection_matrix: Tensor,
    num_triangles: int,
    tangent_fov_x: float,
    tangent_fov_y: float,
    grad_rendered_image: Tensor,
    grad_auxiliary_image: Tensor,
    spherical_harmonics: Tensor,
    sh_degree: int,
    camera_position: Tensor,
    geometry_buffer: Tensor,
    rendered_count: Tensor,
    binning_buffer: Tensor,
    image_buffer: Tensor,
    debug: bool,
) -> TriangleRasterizeBackwardOutput:
    """Return fake tensors for tracing Triangle Splatting backward."""
    del (
        background_color,
        vertices_per_triangle,
        triangle_vertex_offsets,
        triangle_radii,
        view_matrix,
        projection_matrix,
        num_triangles,
        tangent_fov_x,
        tangent_fov_y,
        grad_rendered_image,
        grad_auxiliary_image,
        sh_degree,
        camera_position,
        geometry_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
        debug,
    )
    return (
        torch.empty_like(triangle_vertices),
        torch.empty_like(triangle_sigma).reshape(-1),
        torch.empty_like(precomputed_colors),
        torch.empty(
            (triangle_sigma.shape[0], 1),
            device=triangle_sigma.device,
            dtype=triangle_sigma.dtype,
        ),
        torch.empty_like(spherical_harmonics),
        torch.empty(
            (triangle_sigma.shape[0], 3),
            device=triangle_sigma.device,
            dtype=triangle_sigma.dtype,
        ),
    )


def rasterize_triangles_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: TriangleRasterizeForwardOutput,
) -> None:
    """Save tensors required by the autograd backward pass."""
    differentiable_inputs = (
        inputs[1],
        inputs[2],
        inputs[5],
        inputs[6],
        inputs[14],
    )
    if not requires_grad(*differentiable_inputs):
        ctx.has_context = False
        return
    ctx.has_context = True
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[7],
        inputs[8],
        inputs[14],
        inputs[16],
        output[3],
        output[4],
        output[0],
        output[5],
        output[6],
    )
    ctx.num_triangles = inputs[9]
    ctx.tangent_fov_x = inputs[10]
    ctx.tangent_fov_y = inputs[11]
    ctx.sh_degree = inputs[15]
    ctx.debug = inputs[18]


def rasterize_triangles_backward(
    ctx: Any,
    grad_rendered_count: Tensor,
    grad_rendered_image: Tensor,
    grad_auxiliary_image: Tensor | None,
    grad_radii: Tensor,
    *grad_buffers: Tensor,
) -> tuple[Tensor | None, ...]:
    """Run the autograd backward bridge for Triangle Splatting."""
    del grad_rendered_count, grad_radii, grad_buffers
    if not ctx.has_context:
        return (None,) * 19
    (
        background_color,
        triangle_vertices,
        triangle_sigma,
        vertices_per_triangle,
        triangle_vertex_offsets,
        precomputed_colors,
        view_matrix,
        projection_matrix,
        spherical_harmonics,
        camera_position,
        triangle_radii,
        geometry_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
    ) = ctx.saved_tensors
    grad_auxiliary_image = zero_like_if_none(
        grad_auxiliary_image,
        torch.empty(
            (7, grad_rendered_image.shape[1], grad_rendered_image.shape[2]),
            dtype=grad_rendered_image.dtype,
            device=grad_rendered_image.device,
        ),
    )
    (
        grad_triangle_vertices,
        grad_triangle_sigma,
        grad_precomputed_colors,
        grad_triangle_opacities,
        grad_spherical_harmonics,
        grad_projected_means,
    ) = rasterize_triangles_bwd_op(
        background_color,
        triangle_vertices,
        triangle_sigma,
        vertices_per_triangle,
        triangle_vertex_offsets,
        triangle_radii,
        precomputed_colors,
        view_matrix,
        projection_matrix,
        ctx.num_triangles,
        ctx.tangent_fov_x,
        ctx.tangent_fov_y,
        grad_rendered_image.contiguous(),
        grad_auxiliary_image.contiguous(),
        spherical_harmonics,
        ctx.sh_degree,
        camera_position,
        geometry_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
        ctx.debug,
    )
    del grad_projected_means
    if precomputed_colors.numel() == 0:
        grad_precomputed_colors = torch.empty_like(precomputed_colors)
    return (
        None,
        grad_triangle_vertices,
        grad_triangle_sigma.reshape_as(triangle_sigma),
        None,
        None,
        grad_precomputed_colors,
        grad_triangle_opacities,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        grad_spherical_harmonics,
        None,
        None,
        None,
        None,
    )


@torch.library.custom_op(
    "triangle_splatting::rasterize_triangles",
    mutates_args=(),
)
def rasterize_triangles_op(
    background_color: Tensor,
    triangle_vertices: Tensor,
    triangle_sigma: Tensor,
    vertices_per_triangle: Tensor,
    triangle_vertex_offsets: Tensor,
    precomputed_colors: Tensor,
    triangle_opacities: Tensor,
    view_matrix: Tensor,
    projection_matrix: Tensor,
    num_triangles: int,
    tangent_fov_x: float,
    tangent_fov_y: float,
    image_height: int,
    image_width: int,
    spherical_harmonics: Tensor,
    sh_degree: int,
    camera_position: Tensor,
    prefiltered: bool,
    debug: bool,
) -> TriangleRasterizeForwardOutput:
    """Autograd-enabled Triangle Splatting rasterization stage."""
    return rasterize_triangles_fwd_op(
        background_color,
        triangle_vertices,
        triangle_sigma,
        vertices_per_triangle,
        triangle_vertex_offsets,
        precomputed_colors,
        triangle_opacities,
        view_matrix,
        projection_matrix,
        num_triangles,
        tangent_fov_x,
        tangent_fov_y,
        image_height,
        image_width,
        spherical_harmonics,
        sh_degree,
        camera_position,
        prefiltered,
        debug,
    )


rasterize_triangles_op.register_fake(rasterize_triangles_fwd_fake)
rasterize_triangles_op.register_autograd(
    rasterize_triangles_backward,
    setup_context=rasterize_triangles_setup_context,
)


@torch.library.custom_op("triangle_splatting::mark_visible", mutates_args=())
def mark_visible_op(
    triangle_centers: Tensor,
    view_matrix: Tensor,
    projection_matrix: Tensor,
) -> Tensor:
    """Run upstream Triangle Splatting frustum visibility marking."""
    return backend().mark_visible(
        triangle_centers,
        view_matrix,
        projection_matrix,
    )


@mark_visible_op.register_fake
def mark_visible_fake(
    triangle_centers: Tensor,
    view_matrix: Tensor,
    projection_matrix: Tensor,
) -> Tensor:
    """Return fake tensors for tracing visibility marking."""
    del view_matrix, projection_matrix
    return torch.empty(
        (triangle_centers.shape[0],),
        device=triangle_centers.device,
        dtype=torch.bool,
    )


@torch.library.custom_op(
    "triangle_splatting::compute_relocation",
    mutates_args=(),
)
def compute_relocation_op(
    old_opacity: Tensor,
    old_scale: Tensor,
    relocation_counts: Tensor,
    binomial_coefficients: Tensor,
    max_relocation_count: int,
) -> tuple[Tensor, Tensor]:
    """Run upstream Triangle Splatting relocation helper."""
    return backend().compute_relocation(
        old_opacity,
        old_scale,
        relocation_counts,
        binomial_coefficients,
        max_relocation_count,
    )


@compute_relocation_op.register_fake
def compute_relocation_fake(
    old_opacity: Tensor,
    old_scale: Tensor,
    relocation_counts: Tensor,
    binomial_coefficients: Tensor,
    max_relocation_count: int,
) -> tuple[Tensor, Tensor]:
    """Return fake tensors for tracing the relocation helper."""
    del relocation_counts, binomial_coefficients, max_relocation_count
    return torch.empty_like(old_opacity), torch.empty_like(old_scale)


def rasterize_triangles(
    background_color: Tensor,
    triangle_vertices: Tensor,
    triangle_sigma: Tensor,
    vertices_per_triangle: Tensor,
    triangle_vertex_offsets: Tensor,
    precomputed_colors: Tensor,
    triangle_opacities: Tensor,
    view_matrix: Tensor,
    projection_matrix: Tensor,
    *,
    num_triangles: int,
    tangent_fov_x: float,
    tangent_fov_y: float,
    image_height: int,
    image_width: int,
    spherical_harmonics: Tensor,
    sh_degree: int,
    camera_position: Tensor,
    prefiltered: bool = False,
    debug: bool = False,
) -> TriangleRasterizationResult:
    """Rasterize triangle primitives and wrap raw outputs in a dataclass."""
    return TriangleRasterizationResult.from_tensors(
        *rasterize_triangles_op(
            background_color,
            triangle_vertices,
            triangle_sigma,
            vertices_per_triangle,
            triangle_vertex_offsets,
            precomputed_colors,
            triangle_opacities,
            view_matrix,
            projection_matrix,
            num_triangles,
            tangent_fov_x,
            tangent_fov_y,
            image_height,
            image_width,
            spherical_harmonics,
            sh_degree,
            camera_position,
            prefiltered,
            debug,
        )
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
