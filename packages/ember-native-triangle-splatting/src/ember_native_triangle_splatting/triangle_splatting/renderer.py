"""Ember-native Triangle Splatting backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from beartype import beartype
from ember_core.core.capabilities import HasAlpha, HasDepth, HasNormals
from ember_core.core.contracts import (
    CameraState,
    RenderOptions,
    RenderOutput,
    TriangleSplattingScene,
)
from ember_core.core.registry import output_set, register_backend
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ember_native_triangle_splatting.triangle_splatting.runtime import (
    rasterize_triangles,
)

SUPPORTED_OUTPUTS = output_set("alpha", "depth", "normals")

AlphaTensor = Float[Tensor, "num_cams height width"]
DepthTensor = Float[Tensor, "num_cams height width"]
NormalsTensor = Float[Tensor, "num_cams height width 3"]
RadiiTensor = Int[Tensor, "num_cams num_triangles"]
VisibilityTensor = Bool[Tensor, "num_cams num_triangles"]


@beartype
@dataclass(frozen=True)
class TriangleSplattingNativeRenderOutput(
    RenderOutput,
    HasAlpha,
    HasDepth,
    HasNormals,
):
    """Triangle Splatting render output."""

    alphas: AlphaTensor
    depth: DepthTensor
    normals: NormalsTensor
    median_depth: DepthTensor
    expected_depth: DepthTensor
    distortion: DepthTensor
    radii: RadiiTensor
    visibility_filter: VisibilityTensor
    screen_space_scale: Float[Tensor, "num_cams num_triangles"]
    density_factor: Float[Tensor, "num_cams num_triangles"]
    max_blending: Float[Tensor, "num_cams num_triangles"]


@beartype
@dataclass(frozen=True)
class TriangleSplattingNativeRenderOptions(RenderOptions):
    """Render configuration for Triangle Splatting."""

    color_source: Literal["spherical_harmonics", "direct_rgb"] = (
        "spherical_harmonics"
    )
    depth_mode: Literal["median", "expected"] = "median"
    near_plane: float = 0.01
    far_plane: float = 100.0
    clamp_output: bool = True
    prefiltered: bool = False
    debug: bool = False


@dataclass(frozen=True)
class CameraStageParams:
    """Camera tensors and scalars in the upstream CUDA layout."""

    view_matrix: Float[Tensor, "4 4"]
    projection_matrix: Float[Tensor, "4 4"]
    camera_position: Float[Tensor, " 3"]
    tangent_fov_x: float
    tangent_fov_y: float
    image_width: int
    image_height: int


@beartype
def validate_inputs(
    scene: TriangleSplattingScene,
    camera: CameraState,
) -> None:
    """Validate the CUDA, shape, and camera convention requirements."""
    if scene.triangle_vertices.device.type != "cuda":
        raise ValueError("Triangle Splatting requires scene tensors on CUDA.")
    if camera.cam_to_world.device.type != "cuda":
        raise ValueError("Triangle Splatting requires camera tensors on CUDA.")
    if scene.triangle_vertices.device != camera.cam_to_world.device:
        raise ValueError(
            "Triangle Splatting requires scene and camera tensors on the same "
            "device."
        )
    if camera.camera_convention != "opencv":
        raise ValueError(
            "triangle_splatting.core expects opencv cameras; got "
            f"{camera.camera_convention!r}."
        )


def projection_matrix(
    *,
    tangent_fov_x: float,
    tangent_fov_y: float,
    near_plane: float,
    far_plane: float,
    reference: Tensor,
) -> Float[Tensor, "4 4"]:
    """Build the transposed projection matrix expected by the CUDA kernels."""
    projection = torch.zeros(
        (4, 4),
        dtype=reference.dtype,
        device=reference.device,
    )
    projection[0, 0] = 1.0 / tangent_fov_x
    projection[1, 1] = 1.0 / tangent_fov_y
    projection[3, 2] = 1.0
    projection[2, 2] = far_plane / (far_plane - near_plane)
    projection[2, 3] = -(far_plane * near_plane) / (far_plane - near_plane)
    return projection.mT.contiguous()


def camera_stage_params(
    camera: CameraState,
    camera_index: int,
    options: TriangleSplattingNativeRenderOptions,
) -> CameraStageParams:
    """Pack one camera into the upstream stage parameter layout."""
    intrinsics = camera.get_intrinsics()[camera_index]
    image_width = int(camera.width[camera_index].item())
    image_height = int(camera.height[camera_index].item())
    focal_x = float(intrinsics[0, 0].item())
    focal_y = float(intrinsics[1, 1].item())
    tangent_fov_x = image_width / (2.0 * focal_x)
    tangent_fov_y = image_height / (2.0 * focal_y)
    world_to_camera = torch.linalg.inv(camera.cam_to_world[camera_index])
    view_matrix = world_to_camera.mT.contiguous()
    camera_projection = projection_matrix(
        tangent_fov_x=tangent_fov_x,
        tangent_fov_y=tangent_fov_y,
        near_plane=options.near_plane,
        far_plane=options.far_plane,
        reference=camera.cam_to_world,
    )
    return CameraStageParams(
        view_matrix=view_matrix,
        projection_matrix=(view_matrix[None].bmm(camera_projection[None]))
        .squeeze(0)
        .contiguous(),
        camera_position=camera.cam_to_world[camera_index, :3, 3].contiguous(),
        tangent_fov_x=tangent_fov_x,
        tangent_fov_y=tangent_fov_y,
        image_width=image_width,
        image_height=image_height,
    )


def empty_float_tensor(
    scene: TriangleSplattingScene,
) -> Float[Tensor, " empty"]:
    """Return an empty float tensor on the scene device."""
    return torch.empty(
        (0,),
        dtype=scene.triangle_vertices.dtype,
        device=scene.triangle_vertices.device,
    )


def color_inputs(
    scene: TriangleSplattingScene,
    options: TriangleSplattingNativeRenderOptions,
) -> tuple[Tensor, Tensor]:
    """Split scene color into direct RGB and spherical harmonics inputs."""
    if options.color_source == "direct_rgb":
        return scene.features_dc[:, 0, :].contiguous(), empty_float_tensor(
            scene
        )
    return empty_float_tensor(scene), scene.features.contiguous()


def depth_outputs(
    auxiliary_image: Float[Tensor, "7 height width"],
) -> tuple[
    Float[Tensor, "height width"],
    Float[Tensor, "height width"],
    Float[Tensor, "height width"],
]:
    """Extract expected, median, and distortion depth maps."""
    alpha = auxiliary_image[1]
    expected_depth = torch.nan_to_num(
        auxiliary_image[0] / alpha.clamp_min(1e-8),
        0.0,
        0.0,
    )
    median_depth = torch.nan_to_num(auxiliary_image[5], 0.0, 0.0)
    distortion = torch.nan_to_num(auxiliary_image[6], 0.0, 0.0)
    return expected_depth, median_depth, distortion


def world_normals(
    auxiliary_image: Float[Tensor, "7 height width"],
    stage_params: CameraStageParams,
) -> Float[Tensor, "height width 3"]:
    """Transform rasterized normals from view space to world space."""
    view_normals = auxiliary_image[2:5].permute(1, 2, 0)
    return (view_normals @ stage_params.view_matrix[:3, :3].T).contiguous()


def render_single_camera(
    scene: TriangleSplattingScene,
    camera: CameraState,
    camera_index: int,
    options: TriangleSplattingNativeRenderOptions,
) -> tuple[
    Float[Tensor, "height width 3"],
    Float[Tensor, "height width"],
    Float[Tensor, "height width"],
    Float[Tensor, "height width"],
    Float[Tensor, "height width"],
    Float[Tensor, "height width 3"],
    Int[Tensor, " num_triangles"],
    Float[Tensor, " num_triangles"],
    Float[Tensor, " num_triangles"],
    Float[Tensor, " num_triangles"],
]:
    """Render one camera with the native Triangle Splatting rasterizer."""
    stage_params = camera_stage_params(camera, camera_index, options)
    precomputed_colors, spherical_harmonics = color_inputs(scene, options)
    background_color = options.background_color.to(
        device=scene.triangle_vertices.device,
        dtype=scene.triangle_vertices.dtype,
    )
    result = rasterize_triangles(
        background_color,
        scene.flattened_triangle_vertices.contiguous(),
        scene.sigma.contiguous(),
        scene.vertices_per_triangle.contiguous().to(torch.int32),
        scene.triangle_vertex_offsets.contiguous().to(torch.int32),
        precomputed_colors,
        scene.masked_opacity.contiguous(),
        stage_params.view_matrix,
        stage_params.projection_matrix,
        num_triangles=scene.num_triangles,
        tangent_fov_x=stage_params.tangent_fov_x,
        tangent_fov_y=stage_params.tangent_fov_y,
        image_height=stage_params.image_height,
        image_width=stage_params.image_width,
        spherical_harmonics=spherical_harmonics,
        sh_degree=scene.active_sh_degree,
        camera_position=stage_params.camera_position,
        prefiltered=options.prefiltered,
        debug=options.debug,
    )
    image = result.image.permute(1, 2, 0).contiguous()
    if options.clamp_output:
        image = image.clamp(0.0, 1.0)
    expected_depth, median_depth, distortion = depth_outputs(
        result.auxiliary_image
    )
    normal = world_normals(result.auxiliary_image, stage_params)
    return (
        image,
        result.auxiliary_image[1].contiguous(),
        expected_depth,
        median_depth,
        distortion,
        normal,
        result.radii.contiguous(),
        result.screen_space_scale.contiguous(),
        result.density_factor.contiguous(),
        result.max_blending.contiguous(),
    )


@beartype
def render_triangle_splatting_native(
    scene: TriangleSplattingScene,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: TriangleSplattingNativeRenderOptions | None = None,
) -> TriangleSplattingNativeRenderOutput:
    """Render a scene with the native Triangle Splatting runtime."""
    del return_alpha, return_depth, return_normals
    if return_gaussian_impact_score:
        raise ValueError(
            "The Triangle Splatting backend does not expose Gaussian impact "
            "scores."
        )
    if return_2d_projections:
        raise ValueError(
            "The Triangle Splatting backend does not expose 2D Gaussian "
            "projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "The Triangle Splatting backend does not expose projective "
            "intersection transforms."
        )

    options = options or TriangleSplattingNativeRenderOptions()
    validate_inputs(scene, camera)
    per_camera_outputs = [
        render_single_camera(scene, camera, camera_index, options)
        for camera_index in range(camera.cam_to_world.shape[0])
    ]
    (
        rendered_images,
        alphas,
        expected_depths,
        median_depths,
        distortions,
        normals,
        radii,
        screen_space_scale,
        density_factor,
        max_blending,
    ) = zip(*per_camera_outputs, strict=True)
    expected_depth = torch.stack(expected_depths, dim=0)
    median_depth = torch.stack(median_depths, dim=0)
    depth = median_depth if options.depth_mode == "median" else expected_depth
    radii_tensor = torch.stack(radii, dim=0)
    return TriangleSplattingNativeRenderOutput(
        render=torch.stack(rendered_images, dim=0),
        alphas=torch.stack(alphas, dim=0),
        depth=depth,
        normals=torch.stack(normals, dim=0),
        median_depth=median_depth,
        expected_depth=expected_depth,
        distortion=torch.stack(distortions, dim=0),
        radii=radii_tensor,
        visibility_filter=radii_tensor > 0,
        screen_space_scale=torch.stack(screen_space_scale, dim=0),
        density_factor=torch.stack(density_factor, dim=0),
        max_blending=torch.stack(max_blending, dim=0),
    )


def register() -> None:
    """Register the native Triangle Splatting backend."""
    register_backend(
        name="triangle_splatting.core",
        default_options=TriangleSplattingNativeRenderOptions(),
        accepted_scene_types=(TriangleSplattingScene,),
        supported_outputs=SUPPORTED_OUTPUTS,
    )(render_triangle_splatting_native)


__all__ = [
    "TriangleSplattingNativeRenderOptions",
    "TriangleSplattingNativeRenderOutput",
    "register",
    "render_triangle_splatting_native",
]
