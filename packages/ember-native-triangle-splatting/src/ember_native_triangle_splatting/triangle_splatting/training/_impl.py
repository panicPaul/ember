"""Training helpers for Triangle Splatting scenes."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from ember_core.core.contracts import CameraState, TriangleSplattingScene
from ember_core.data import PointCloudState, SceneRecord
from ember_core.densification import (
    BaseDensificationMethod,
    DensificationContext,
    DensificationLifecycleContext,
    register_family_ops,
)
from ember_core.initialization import InitializedModel
from ember_core.training import (
    CallableSpec,
    LossResult,
    OptimizationConfig,
    ParameterGroupConfig,
    ParameterTargetSpec,
    TrainState,
)
from jaxtyping import Bool, Float, Int
from torch import Tensor

SPHERICAL_HARMONIC_DC_SCALE = 0.28209479177387814
TRIANGLE_SPLATTING_TOPOLOGY_PARAMETER_FIELD_NAMES = (
    "triangle_vertices",
    "features_dc",
    "features_rest",
    "logit_opacity",
    "raw_sigma",
    "mask_logits",
)


def triangle_splatting_rgb_to_sh(
    rgb: Float[Tensor, "... 3"],
) -> Float[Tensor, "... 3"]:
    """Convert normalized RGB colors to SH DC coefficients."""
    return (rgb - 0.5) / SPHERICAL_HARMONIC_DC_SCALE


def scene_topology_parameter_tensor(
    scene: TriangleSplattingScene,
    name: str,
) -> Tensor:
    """Return a topology parameter by its public scene field name."""
    if name == "triangle_vertices":
        return scene.triangle_vertices
    if name == "features_dc":
        return scene.features_dc
    if name == "features_rest":
        return scene.features_rest
    if name == "logit_opacity":
        return scene.logit_opacity
    if name == "raw_sigma":
        return scene.raw_sigma
    if name == "mask_logits":
        return scene.mask_logits
    raise KeyError(f"Unknown Triangle Splatting topology field {name!r}.")


def inverse_sigmoid(
    values: Float[Tensor, ...],
) -> Float[Tensor, ...]:
    """Return logits for probabilities clamped away from the boundary."""
    return torch.logit(values.clamp(1e-5, 1.0 - 1e-5))


def triangle_splatting_root_mean_squared_knn_distances(
    positions: Float[Tensor, "num_points 3"],
    *,
    torch_chunk_size: int = 512,
) -> Float[Tensor, " num_points"]:
    """Compute upstream Triangle Splatting initial radius distances."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "Triangle Splatting KNN distances expect positions with shape "
            f"(num_points, 3), got {tuple(positions.shape)}."
        )
    if torch_chunk_size < 1:
        raise ValueError("torch_chunk_size must be at least 1.")
    num_points = int(positions.shape[0])
    if num_points == 0:
        return torch.empty(
            (0,),
            dtype=positions.dtype,
            device=positions.device,
        )
    if positions.device.type == "cuda":
        try:
            from simple_knn._C import distCUDA2

            mean_squared = distCUDA2(positions.contiguous())
            return mean_squared.clamp_min(1e-7).sqrt()
        except Exception:
            pass
    if num_points == 1:
        return torch.full(
            (1,),
            1e-3,
            dtype=positions.dtype,
            device=positions.device,
        )
    k_nearest = min(3, num_points - 1)
    nearest_distances = []
    for start_index in range(0, num_points, torch_chunk_size):
        stop_index = min(start_index + torch_chunk_size, num_points)
        distances = torch.cdist(positions[start_index:stop_index], positions)
        row_indices = torch.arange(
            stop_index - start_index,
            dtype=torch.long,
            device=positions.device,
        )
        col_indices = torch.arange(
            start_index,
            stop_index,
            dtype=torch.long,
            device=positions.device,
        )
        distances[row_indices, col_indices] = math.inf
        nearest_distances.append(
            distances.topk(k_nearest, largest=False).values
        )
    mean_squared = torch.cat(nearest_distances, dim=0).square().mean(dim=1)
    return mean_squared.clamp_min(1e-7).sqrt()


def fibonacci_directions(
    num_directions: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "num_directions 3"]:
    """Generate unit directions on a sphere with Fibonacci spacing."""
    if num_directions < 2:
        raise ValueError("num_directions must be at least 2.")
    indices = torch.arange(num_directions, device=device, dtype=dtype)
    z_coord = 1.0 - (2.0 * indices / float(num_directions - 1))
    radius_xy = torch.sqrt((1.0 - z_coord.square()).clamp_min(0.0))
    theta = math.pi * (3.0 - math.sqrt(5.0)) * indices
    return torch.stack(
        [
            radius_xy * torch.cos(theta),
            radius_xy * torch.sin(theta),
            z_coord,
        ],
        dim=-1,
    )


def random_rotation_matrices(
    num_matrices: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Float[Tensor, "num_matrices 3 3"]:
    """Generate random 3D rotation matrices."""
    axis = torch.randn(
        (num_matrices, 3),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angles = (
        2.0
        * math.pi
        * torch.rand(
            (num_matrices,),
            device=device,
            dtype=dtype,
            generator=generator,
        )
    )
    sine = torch.sin(angles)
    cosine = torch.cos(angles)
    skew = torch.zeros((num_matrices, 3, 3), device=device, dtype=dtype)
    x_axis, y_axis, z_axis = axis[:, 0], axis[:, 1], axis[:, 2]
    skew[:, 0, 1] = -z_axis
    skew[:, 0, 2] = y_axis
    skew[:, 1, 0] = z_axis
    skew[:, 1, 2] = -x_axis
    skew[:, 2, 0] = -y_axis
    skew[:, 2, 1] = x_axis
    identity = torch.eye(3, device=device, dtype=dtype).expand(
        num_matrices,
        -1,
        -1,
    )
    return (
        identity
        + sine[:, None, None] * skew
        + (1.0 - cosine)[:, None, None] * skew.bmm(skew)
    )


def generate_triangle_vertices(
    centers: Float[Tensor, "num_triangles 3"],
    radii: Float[Tensor, " num_triangles"],
    *,
    vertices_per_triangle: int = 3,
    generator: torch.Generator | None = None,
    chunk_size: int = 2000,
) -> Float[Tensor, "num_triangles vertices_per_triangle 3"]:
    """Generate randomly oriented triangle vertices around point centers."""
    device = centers.device
    dtype = centers.dtype
    num_triangles = int(centers.shape[0])
    base_directions = fibonacci_directions(
        vertices_per_triangle,
        device=device,
        dtype=dtype,
    )
    triangle_vertices = torch.empty(
        (num_triangles, vertices_per_triangle, 3),
        device=device,
        dtype=dtype,
    )
    for start_index in range(0, num_triangles, chunk_size):
        stop_index = min(start_index + chunk_size, num_triangles)
        chunk_centers = centers[start_index:stop_index]
        chunk_radii = radii[start_index:stop_index]
        rotations = random_rotation_matrices(
            int(chunk_centers.shape[0]),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        rotated_directions = torch.einsum(
            "nij,kj->nki",
            rotations,
            base_directions,
        )
        triangle_vertices[start_index:stop_index] = (
            chunk_centers[:, None, :]
            + rotated_directions * chunk_radii[:, None, None]
        )
    return triangle_vertices


def fibonacci_sphere(
    num_points: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "num_points 3"]:
    """Generate sky-dome points on a sphere like the upstream initializer."""
    if num_points <= 0:
        return torch.empty((0, 3), device=device, dtype=dtype)
    indices = torch.arange(num_points, device=device, dtype=dtype)
    phi = math.pi * (math.sqrt(5.0) - 1.0)
    y_coord = 1.0 - (indices / max(float(num_points - 1), 1.0)) * 2.0
    radius = torch.sqrt((1.0 - y_coord.square()).clamp_min(0.0))
    theta = phi * indices
    return torch.stack(
        [torch.cos(theta) * radius, y_coord, torch.sin(theta) * radius],
        dim=-1,
    )


def require_point_cloud(scene_record: SceneRecord) -> PointCloudState:
    """Return the SfM point cloud required by Triangle Splatting."""
    if scene_record.point_cloud is None:
        raise ValueError(
            "Triangle Splatting initialization requires an SfM point cloud."
        )
    return scene_record.point_cloud


def triangle_splatting_is_large_scene(
    points: Float[Tensor, "num_points 3"],
) -> bool:
    """Return the upstream large-scene flag used by the image loss."""
    if int(points.shape[0]) == 0:
        return False
    extents = points.max(dim=0).values - points.min(dim=0).values
    return bool((extents.max() > 300.0).item())


def initialize_triangle_splatting_scene_from_scene_record(
    scene_record: SceneRecord,
    *,
    device: torch.device | str = torch.device("cuda"),
    sh_degree: int = 3,
    vertices_per_triangle: int = 3,
    initial_opacity: float = 0.28,
    initial_triangle_size: float = 2.23,
    initial_sigma: float = 1.16,
    include_sky_dome: bool = True,
    seed: int = 0,
) -> TriangleSplattingScene:
    """Build a TriangleSplattingScene from an SfM point cloud."""
    target_device = torch.device(device)
    point_cloud = require_point_cloud(scene_record)
    centers = point_cloud.points.to(device=target_device, dtype=torch.float32)
    colors = (
        torch.ones_like(centers)
        if point_cloud.colors is None
        else point_cloud.colors.to(device=target_device, dtype=torch.float32)
    )
    if include_sky_dome:
        sky_point_count = int(int(centers.shape[0]) * 0.05)
        scene_radius = torch.max(torch.abs(centers)).clamp_min(1e-8)
        sky_points = (
            fibonacci_sphere(
                sky_point_count,
                device=target_device,
                dtype=centers.dtype,
            )
            * scene_radius
        )
        centers = torch.cat([sky_points, centers], dim=0)
        colors = torch.cat([torch.ones_like(sky_points), colors], dim=0)

    generator = torch.Generator(device=str(target_device))
    generator.manual_seed(seed)
    distances = triangle_splatting_root_mean_squared_knn_distances(centers)
    radii = initial_triangle_size * distances
    triangle_vertices = generate_triangle_vertices(
        centers,
        radii,
        vertices_per_triangle=vertices_per_triangle,
        generator=generator,
    )
    num_triangles = int(triangle_vertices.shape[0])
    sh_coeffs = (sh_degree + 1) ** 2
    features = torch.zeros(
        (num_triangles, sh_coeffs, 3),
        dtype=torch.float32,
        device=target_device,
    )
    features[:, 0, :] = triangle_splatting_rgb_to_sh(colors)
    opacity = torch.full(
        (num_triangles, 1),
        initial_opacity,
        dtype=torch.float32,
        device=target_device,
    )
    sigma = torch.full(
        (num_triangles, 1),
        initial_sigma,
        dtype=torch.float32,
        device=target_device,
    )
    raw_sigma = torch.log((sigma - 0.01).clamp_min(1e-8))
    return TriangleSplattingScene(
        triangle_vertices=triangle_vertices.requires_grad_(True),
        raw_sigma=raw_sigma.requires_grad_(True),
        logit_opacity=inverse_sigmoid(opacity).requires_grad_(True),
        features_dc=features[:, :1, :].contiguous().requires_grad_(True),
        features_rest=features[:, 1:, :].contiguous().requires_grad_(True),
        sh_degree=sh_degree,
        active_sh_degree=0,
    )


def initialize_triangle_splatting_model_from_scene_record(
    scene_record: SceneRecord,
    *,
    modules: dict[str, torch.nn.Module] | None = None,
    parameters: dict[str, torch.nn.Parameter] | None = None,
    buffers: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
    device: torch.device | str = torch.device("cuda"),
    sh_degree: int = 3,
    vertices_per_triangle: int = 3,
    initial_opacity: float = 0.28,
    initial_triangle_size: float = 2.23,
    initial_sigma: float = 1.16,
    include_sky_dome: bool = True,
    seed: int = 0,
) -> InitializedModel:
    """Build a Triangle Splatting training payload from scene geometry."""
    scene = initialize_triangle_splatting_scene_from_scene_record(
        scene_record,
        device=device,
        sh_degree=sh_degree,
        vertices_per_triangle=vertices_per_triangle,
        initial_opacity=initial_opacity,
        initial_triangle_size=initial_triangle_size,
        initial_sigma=initial_sigma,
        include_sky_dome=include_sky_dome,
        seed=seed,
    )
    point_cloud = require_point_cloud(scene_record)
    resolved_metadata = dict(metadata or {})
    resolved_metadata.setdefault(
        "triangle_splatting_large_scene",
        triangle_splatting_is_large_scene(point_cloud.points.to(torch.float32)),
    )
    return InitializedModel(
        scene=scene,
        modules=dict(modules or {}),
        parameters=dict(parameters or {}),
        buffers=dict(buffers or {}),
        metadata=resolved_metadata,
    )


@dataclass(frozen=True)
class TriangleSplattingOptimizationRecipe:
    """Triangle Splatting optimizer group defaults."""

    optimizer: str = "torch.optim.Adam"
    feature_lr: float = 0.0025
    opacity_lr: float = 0.014
    sigma_lr: float = 0.0008
    triangle_vertices_lr_init: float = 0.0018
    triangle_vertices_lr_final_factor: float = 0.01
    triangle_vertices_lr_max_steps: int = 30_000
    adam_eps: float = 1e-15


def scene_parameter_target(name: str) -> ParameterTargetSpec:
    """Build a scene-parameter target spec."""
    return ParameterTargetSpec(scope="scene", name=name)


def triangle_splatting_parameter_groups(
    recipe: TriangleSplattingOptimizationRecipe | dict[str, Any],
) -> list[ParameterGroupConfig]:
    """Build Triangle Splatting optimizer groups."""
    if isinstance(recipe, dict):
        recipe = TriangleSplattingOptimizationRecipe(**recipe)
    optimizer_kwargs = {"eps": recipe.adam_eps}
    return [
        ParameterGroupConfig(
            target=scene_parameter_target("features_dc"),
            optimizer=recipe.optimizer,
            lr=recipe.feature_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
        ParameterGroupConfig(
            target=scene_parameter_target("features_rest"),
            optimizer=recipe.optimizer,
            lr=recipe.feature_lr / 20.0,
            optimizer_kwargs=optimizer_kwargs,
        ),
        ParameterGroupConfig(
            target=scene_parameter_target("logit_opacity"),
            optimizer=recipe.optimizer,
            lr=recipe.opacity_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
        ParameterGroupConfig(
            target=scene_parameter_target("triangle_vertices"),
            optimizer=recipe.optimizer,
            lr=recipe.triangle_vertices_lr_init,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=CallableSpec(
                target="ember_core.training.exponential_decay_to",
                kwargs={
                    "final_lr": recipe.triangle_vertices_lr_init
                    * recipe.triangle_vertices_lr_final_factor,
                    "max_steps": recipe.triangle_vertices_lr_max_steps,
                },
            ),
        ),
        ParameterGroupConfig(
            target=scene_parameter_target("raw_sigma"),
            optimizer=recipe.optimizer,
            lr=recipe.sigma_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
    ]


def triangle_splatting_optimization_config(
    recipe: TriangleSplattingOptimizationRecipe | dict[str, Any],
) -> OptimizationConfig:
    """Build an OptimizationConfig from Triangle Splatting optimizer groups."""
    return OptimizationConfig(
        parameter_groups=triangle_splatting_parameter_groups(recipe)
    )


def equilateral_area(
    triangle_vertices: Float[Tensor, "num_triangles 3 3"],
) -> Float[Tensor, " num_triangles"]:
    """Return per-triangle areas used by the upstream size regularizer."""
    edge_a = triangle_vertices[:, 1, :] - triangle_vertices[:, 0, :]
    edge_b = triangle_vertices[:, 2, :] - triangle_vertices[:, 0, :]
    return 0.5 * torch.cross(edge_a, edge_b, dim=1).norm(dim=1)


def camera_depth_normals(
    camera: CameraState,
    depth: Float[Tensor, "height width"],
) -> Float[Tensor, "height width 3"]:
    """Estimate world-space normals from a depth map."""
    intrinsics = camera.get_intrinsics()[0].to(
        device=depth.device,
        dtype=depth.dtype,
    )
    cam_to_world = camera.cam_to_world[0].to(
        device=depth.device,
        dtype=depth.dtype,
    )
    height, width = int(depth.shape[0]), int(depth.shape[1])
    xs = torch.arange(width, device=depth.device, dtype=depth.dtype)
    ys = torch.arange(height, device=depth.device, dtype=depth.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x_camera = (grid_x - intrinsics[0, 2]) / intrinsics[0, 0] * depth
    y_camera = (grid_y - intrinsics[1, 2]) / intrinsics[1, 1] * depth
    camera_points = torch.stack([x_camera, y_camera, depth], dim=-1)
    world_points = camera_points @ cam_to_world[:3, :3].T + cam_to_world[
        :3, 3
    ].reshape(1, 1, 3)
    vertical_delta = world_points[2:, 1:-1] - world_points[:-2, 1:-1]
    horizontal_delta = world_points[1:-1, 2:] - world_points[1:-1, :-2]
    normal = torch.zeros_like(world_points)
    normal[1:-1, 1:-1] = torch.cross(
        vertical_delta,
        horizontal_delta,
        dim=-1,
    )
    return (
        normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    ).contiguous()


def triangle_splatting_dssim_loss(
    prediction: Float[Tensor, "batch height width 3"],
    target: Float[Tensor, "batch height width 3"],
) -> Tensor:
    """Compute DSSIM through the shared splatting-training helper."""
    from ember_splatting_training.losses import dssim_loss

    return dssim_loss(prediction, target)


def triangle_splatting_loss(
    state: TrainState,
    batch: Any,
    render_output: Any,
    *,
    weights: dict[str, float],
    lambda_dssim: float = 0.2,
    lambda_opacity: float = 0.0055,
    lambda_size: float = 1e-8,
    lambda_normals: float = 0.0001,
    lambda_dist: float = 0.0,
    iteration_mesh: int = 5000,
    densify_until_iter: int = 25_000,
    outdoor: bool = True,
    large_scene: bool | None = None,
) -> LossResult:
    """Triangle Splatting paper loss."""
    del weights
    prediction = render_output.render
    target = batch.images
    if prediction.shape != target.shape:
        raise ValueError(
            "Triangle Splatting loss expects prediction and target images to "
            f"share shape, got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}."
        )
    model_metadata = state.model.metadata
    resolved_large_scene = bool(
        model_metadata.get("triangle_splatting_large_scene", False)
        if large_scene is None
        else large_scene
    )
    if outdoor and resolved_large_scene:
        pixel_loss = (prediction - target).square().mean()
    else:
        pixel_loss = (prediction - target).abs().mean()
    image_loss = (1.0 - lambda_dssim) * pixel_loss + lambda_dssim * (
        1.0 - (1.0 - 2.0 * triangle_splatting_dssim_loss(prediction, target))
    )
    scene = state.model.scene
    if not isinstance(scene, TriangleSplattingScene):
        raise TypeError(
            "triangle_splatting_loss expects a TriangleSplattingScene model."
        )
    opacity_loss = scene.opacity.abs().mean() * lambda_opacity
    area = equilateral_area(scene.triangle_vertices).clamp_min(1e-8)
    size_loss = (1.0 / area).mean() * lambda_size
    normal_loss = torch.zeros(
        (), dtype=prediction.dtype, device=prediction.device
    )
    dist_loss = torch.zeros_like(normal_loss)
    step = int(state.step) + 1
    if step > iteration_mesh:
        dist_loss = lambda_dist * render_output.distortion.mean()
        surface_normals = torch.stack(
            [
                camera_depth_normals(
                    batch.camera.to(prediction.device),
                    render_output.depth[camera_index],
                )
                for camera_index in range(prediction.shape[0])
            ],
            dim=0,
        )
        surface_normals = (
            surface_normals * render_output.alphas[..., None].detach()
        )
        normal_error = 1.0 - (render_output.normals * surface_normals).sum(
            dim=-1
        )
        normal_loss = lambda_normals * normal_error.mean()
    loss = image_loss + opacity_loss + normal_loss + dist_loss
    if step < densify_until_iter:
        loss = loss + size_loss
    return LossResult(
        loss=loss,
        metrics={
            "pixel_loss": float(pixel_loss.detach().item()),
            "image_loss": float(image_loss.detach().item()),
            "opacity_loss": float(opacity_loss.detach().item()),
            "size_loss": float(size_loss.detach().item()),
            "normal_loss": float(normal_loss.detach().item()),
            "dist_loss": float(dist_loss.detach().item()),
        },
    )


class SceneOptimizerStateAdapter:
    """Update scene fields and optimizer state after topology edits."""

    def __init__(self, optimizers: Sequence[Any]) -> None:
        """Collect optimizer bindings that target scene fields."""
        self.bindings: dict[str, list[Any]] = {}
        for binding in optimizers:
            if not hasattr(binding, "matches_target") or not hasattr(
                binding,
                "field_name",
            ):
                continue
            matches_target = binding.matches_target
            field_name = binding.field_name
            if not callable(matches_target) or field_name is None:
                continue
            if not matches_target("scene", field_name):
                continue
            self.bindings.setdefault(field_name, []).append(binding)

    def replace_scene_fields(
        self,
        scene: TriangleSplattingScene,
        updates: dict[str, Tensor],
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]],
    ) -> None:
        """Replace scene fields and update optimizer state."""
        installed = scene.replace_fields_(**updates)
        for name, value in installed.items():
            for binding in self.bindings.get(name, ()):
                if not hasattr(binding, "replace_parameter"):
                    continue
                replace_parameter = binding.replace_parameter
                if callable(replace_parameter):
                    replace_parameter(value, state_transforms[name])


class TriangleSplattingFamilyOps:
    """Topology edits for Triangle Splatting scenes."""

    def __init__(self, state: Any, optimizers: Sequence[Any]) -> None:
        """Bind mutable training state and optimizer references."""
        self.state = state
        self.optimizer_adapter = SceneOptimizerStateAdapter(optimizers)

    @property
    def scene(self) -> TriangleSplattingScene:
        """Return the bound Triangle Splatting scene."""
        scene = self.state.model.scene
        if not isinstance(scene, TriangleSplattingScene):
            raise TypeError(
                "TriangleSplattingFamilyOps requires a TriangleSplattingScene."
            )
        return scene

    def replace_fields(
        self,
        updates: dict[str, Tensor],
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]],
    ) -> None:
        """Replace scene tensor fields while preserving optimizer state."""
        self.optimizer_adapter.replace_scene_fields(
            self.scene,
            updates,
            state_transforms,
        )


@dataclass
class TriangleSplattingDensification(BaseDensificationMethod):
    """Upstream-style Triangle Splatting split/clone/prune refinement."""

    densification_interval: int = 500
    densify_from_iter: int = 500
    densify_until_iter: int = 25_000
    max_shapes: int = 5_200_000
    add_shape: float = 1.3
    split_size: float = 24.0
    max_noise_factor: float = 1.5
    opacity_dead: float = 0.014
    importance_threshold: float = 0.022
    proba_distr: Literal[0, 1, 2] = 2
    outdoor: bool = True
    expected_scene_families: tuple[str, ...] = ("triangle_splatting",)
    family_ops: TriangleSplattingFamilyOps | None = field(
        default=None,
        init=False,
        repr=False,
    )
    triangle_area: Tensor | None = field(default=None, init=False, repr=False)
    image_size: Tensor | None = field(default=None, init=False, repr=False)
    importance_score: Tensor | None = field(
        default=None, init=False, repr=False
    )
    opacity_probability_next: bool = field(default=True, init=False, repr=False)
    new_round: bool = field(default=False, init=False, repr=False)
    removed_points: bool = field(default=False, init=False, repr=False)
    visibility_statistics_ready: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    number_of_training_views: int = field(default=0, init=False, repr=False)
    round_seen_frame_keys: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Bind Triangle Splatting topology operations."""
        del optimizers
        if not isinstance(family_ops, TriangleSplattingFamilyOps):
            raise TypeError(
                "TriangleSplattingDensification requires "
                "TriangleSplattingFamilyOps."
            )
        if state.model.scene.scene_family not in self.expected_scene_families:
            raise TypeError(
                "TriangleSplattingDensification expects Triangle Splatting "
                f"scenes, got {state.model.scene.scene_family!r}."
            )
        self.family_ops = family_ops
        self.reset_statistics()

    def before_training(
        self,
        context: DensificationLifecycleContext,
    ) -> None:
        """Initialize refinement statistics."""
        self.number_of_training_views = self.resolve_number_of_training_views(
            context
        )
        self.new_round = False
        self.removed_points = False
        self.visibility_statistics_ready = False
        self.round_seen_frame_keys.clear()
        self.reset_statistics()

    def post_backward(self, context: DensificationContext) -> None:
        """Accumulate per-triangle image-size and importance statistics."""
        if self.family_ops is None:
            return
        output = context.render_output
        scene = self.family_ops.scene
        self.ensure_statistics(scene)
        assert self.triangle_area is not None
        assert self.image_size is not None
        assert self.importance_score is not None
        density_factor = output.density_factor.detach().max(dim=0).values
        screen_space_scale = (
            output.screen_space_scale.detach().max(dim=0).values
        )
        max_blending = output.max_blending.detach().max(dim=0).values
        if self.new_round:
            self.triangle_area[density_factor > 1.0] += 1.0
        self.image_size = torch.maximum(self.image_size, screen_space_scale)
        self.importance_score = torch.maximum(
            self.importance_score,
            max_blending,
        )
        self.observe_training_round(context)

    def pre_optimizer_step(self, context: DensificationContext) -> None:
        """Run scheduled Triangle Splatting refinement."""
        if self.family_ops is None:
            return
        scene = self.family_ops.scene
        step = int(context.step) + 1
        if step % 1000 == 0 and scene.active_sh_degree < scene.sh_degree:
            scene.active_sh_degree += 1
        if (
            step < self.densify_until_iter
            and step % self.densification_interval == 0
            and step > self.densify_from_iter
        ):
            self.add_new_triangles(self.dead_mask(before_final=True, step=step))
            return
        if (
            step > self.densify_until_iter
            and step % self.densification_interval == 0
        ):
            self.remove_final_points(
                self.dead_mask(before_final=False, step=step)
            )

    def resolve_number_of_training_views(
        self,
        context: DensificationLifecycleContext | DensificationContext,
    ) -> int:
        """Return the number of training cameras when the runtime exposes it."""
        runtime = context.runtime
        if runtime is None:
            return self.number_of_training_views
        try:
            return len(runtime.all_cameras())
        except Exception:
            return self.number_of_training_views

    def training_frame_keys(
        self,
        context: DensificationContext,
    ) -> tuple[str, ...]:
        """Return stable keys for the frames in the current training batch."""
        frames = context.batch.frames
        return tuple(
            f"{frame.sensor_id}:{frame.frame_id}:{frame.camera_index}"
            for frame in frames
        )

    def observed_complete_training_round(self) -> bool:
        """Return whether the current frame-key set covers a camera round."""
        if self.number_of_training_views <= 0:
            return len(self.round_seen_frame_keys) > 0
        return len(self.round_seen_frame_keys) >= self.number_of_training_views

    def observe_training_round(self, context: DensificationContext) -> None:
        """Advance the upstream post-edit camera-round state machine."""
        if not self.removed_points and not self.new_round:
            return
        self.number_of_training_views = self.resolve_number_of_training_views(
            context
        )
        frame_keys = self.training_frame_keys(context)
        if not frame_keys:
            return

        # Upstream starts a fresh visibility-counting round only after the
        # camera stack has been exhausted following a topology edit.
        self.round_seen_frame_keys.update(frame_keys)
        if not self.observed_complete_training_round():
            return

        if self.removed_points:
            self.removed_points = False
            self.new_round = True
            self.round_seen_frame_keys.clear()
            return

        self.new_round = False
        self.visibility_statistics_ready = True
        self.round_seen_frame_keys.clear()

    def ensure_statistics(self, scene: TriangleSplattingScene) -> None:
        """Ensure refinement accumulators match the current scene size."""
        num_triangles = scene.num_triangles
        needs_reset = (
            self.triangle_area is None
            or int(self.triangle_area.shape[0]) != num_triangles
        )
        if needs_reset:
            self.triangle_area = torch.zeros(
                (num_triangles,),
                dtype=scene.triangle_vertices.dtype,
                device=scene.triangle_vertices.device,
            )
            self.image_size = torch.zeros_like(self.triangle_area)
            self.importance_score = torch.zeros_like(self.triangle_area)

    def reset_statistics(self) -> None:
        """Reset refinement statistics for the bound scene."""
        if self.family_ops is None:
            self.triangle_area = None
            self.image_size = None
            self.importance_score = None
            return
        self.triangle_area = None
        self.ensure_statistics(self.family_ops.scene)

    def dead_mask(
        self, *, before_final: bool, step: int
    ) -> Bool[Tensor, " num_triangles"]:
        """Compute upstream-style dead-triangle pruning mask."""
        assert self.family_ops is not None
        assert self.triangle_area is not None
        assert self.image_size is not None
        assert self.importance_score is not None
        scene = self.family_ops.scene
        opacity = scene.opacity.squeeze(-1)
        if self.number_of_training_views < 250 or not self.new_round:
            dead_mask = (self.importance_score < self.importance_threshold) | (
                opacity <= self.opacity_dead
            )
        else:
            dead_mask = opacity <= self.opacity_dead
        can_prune_by_visibility = (
            self.visibility_statistics_ready and not self.new_round
        )
        if before_final and step > 1000 and can_prune_by_visibility:
            dead_mask |= self.triangle_area < 2.0
            if not self.outdoor:
                dead_mask |= self.image_size > 1400.0
        if not before_final and can_prune_by_visibility:
            dead_mask |= self.triangle_area < 2.0
        return dead_mask

    def begin_post_edit_visibility_round(self) -> None:
        """Start waiting for a fresh post-topology camera round."""
        self.new_round = False
        self.removed_points = True
        self.visibility_statistics_ready = False
        self.round_seen_frame_keys.clear()

    def probability_group(self) -> bool:
        """Return whether the next alive sampling pass is opacity-based."""
        if self.proba_distr == 0:
            return True
        if self.proba_distr == 1:
            return False
        result = self.opacity_probability_next
        self.opacity_probability_next = not self.opacity_probability_next
        return result

    def sample_alive(
        self,
        probabilities: Tensor,
        num_requested: int,
        big_mask: Bool[Tensor, " num_triangles"],
    ) -> Int[Tensor, " num_selected"]:
        """Sample alive triangles with upstream split/clone costs."""
        probabilities = torch.nan_to_num(probabilities, 0.0, 0.0, 0.0)
        probabilities = probabilities.clamp_min(0.0)
        positive_count = int((probabilities > 0).sum().item())
        if positive_count <= 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=probabilities.device,
            )
        probabilities = probabilities / probabilities.sum().clamp_min(
            torch.finfo(probabilities.dtype).eps
        )
        sampled = torch.multinomial(
            probabilities,
            min(num_requested, positive_count),
            replacement=False,
        )
        costs = torch.where(
            big_mask[sampled],
            torch.tensor(3, dtype=torch.long, device=probabilities.device),
            torch.tensor(1, dtype=torch.long, device=probabilities.device),
        )
        cumulative_costs = torch.cumsum(costs, dim=0)
        cutoff_indices = (cumulative_costs >= num_requested).nonzero(
            as_tuple=True
        )[0]
        if cutoff_indices.numel() == 0:
            return sampled
        return sampled[: int(cutoff_indices[0].item()) + 1]

    def split_fields(self, indices: Tensor) -> dict[str, Tensor]:
        """Subdivide selected triangles into four child triangles."""
        assert self.family_ops is not None
        scene = self.family_ops.scene
        selected = scene.triangle_vertices[indices]
        vertex_a = selected[:, 0, :]
        vertex_b = selected[:, 1, :]
        vertex_c = selected[:, 2, :]
        midpoint_ab = (vertex_a + vertex_b) / 2.0
        midpoint_ac = (vertex_a + vertex_c) / 2.0
        midpoint_bc = (vertex_b + vertex_c) / 2.0
        split_vertices = torch.cat(
            [
                torch.stack([vertex_a, midpoint_ab, midpoint_ac], dim=1),
                torch.stack([vertex_b, midpoint_ab, midpoint_bc], dim=1),
                torch.stack([vertex_c, midpoint_ac, midpoint_bc], dim=1),
                torch.stack([midpoint_ab, midpoint_ac, midpoint_bc], dim=1),
            ],
            dim=0,
        )
        return {
            "triangle_vertices": split_vertices,
            "features_dc": scene.features_dc[indices].repeat(4, 1, 1),
            "features_rest": scene.features_rest[indices].repeat(4, 1, 1),
            "logit_opacity": scene.logit_opacity[indices].repeat(4, 1),
            "raw_sigma": scene.raw_sigma[indices].repeat(4, 1),
            "mask_logits": torch.ones_like(
                scene.mask_logits[indices].repeat(4, 1)
            ),
        }

    def clone_fields(self, indices: Tensor) -> dict[str, Tensor]:
        """Clone selected small triangles with in-plane noise."""
        assert self.family_ops is not None
        scene = self.family_ops.scene
        selected = scene.triangle_vertices[indices]
        normals = torch.cross(
            selected[:, 1, :] - selected[:, 0, :],
            selected[:, 2, :] - selected[:, 0, :],
            dim=1,
        )
        normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-9)
        shape_sizes = selected.max(dim=1).values - selected.min(dim=1).values
        noise = (
            torch.rand(
                (int(selected.shape[0]), 1, 3),
                dtype=selected.dtype,
                device=selected.device,
            )
            - 0.5
        ) * (shape_sizes * self.max_noise_factor)[:, None, :]
        dot_products = (noise * normals[:, None, :]).sum(dim=-1, keepdim=True)
        noisy_selected = selected + noise - dot_products * normals[:, None, :]
        opacity_old = scene.opacity[indices]
        opacity_new = inverse_sigmoid(1.0 - torch.pow(1.0 - opacity_old, 0.5))
        return {
            "triangle_vertices": torch.cat([selected, noisy_selected], dim=0),
            "features_dc": torch.cat(
                [scene.features_dc[indices], scene.features_dc[indices]],
                dim=0,
            ),
            "features_rest": torch.cat(
                [scene.features_rest[indices], scene.features_rest[indices]],
                dim=0,
            ),
            "logit_opacity": torch.cat([opacity_new, opacity_new], dim=0),
            "raw_sigma": torch.cat(
                [scene.raw_sigma[indices], scene.raw_sigma[indices]],
                dim=0,
            ),
            "mask_logits": torch.cat(
                [scene.mask_logits[indices], torch.ones_like(opacity_new)],
                dim=0,
            ),
        }

    def add_new_triangles(
        self, dead_mask: Bool[Tensor, " num_triangles"]
    ) -> None:
        """Append sampled children/clones and remove sources plus dead rows."""
        assert self.family_ops is not None
        assert self.image_size is not None
        scene = self.family_ops.scene
        current_count = scene.num_triangles
        target_count = min(self.max_shapes, int(self.add_shape * current_count))
        num_new = max(0, target_count - current_count) + int(dead_mask.sum())
        if num_new <= 0:
            return
        probabilities = (
            scene.opacity.squeeze(-1)
            if self.probability_group()
            else 1.0
            / scene.sigma.squeeze(-1).clamp_min(
                torch.finfo(scene.sigma.dtype).eps
            )
        )
        probabilities = probabilities.clone()
        probabilities[dead_mask] = 0.0
        big_mask = self.image_size > self.split_size
        selected_indices = self.sample_alive(probabilities, num_new, big_mask)
        if selected_indices.numel() == 0:
            return
        selected_big_mask = big_mask[selected_indices]
        split_indices = selected_indices[selected_big_mask]
        clone_indices = selected_indices[~selected_big_mask]
        appended = self.empty_field_values(scene)
        if split_indices.numel() > 0:
            appended = self.concat_field_values(
                appended,
                self.split_fields(split_indices),
            )
        if clone_indices.numel() > 0:
            appended = self.concat_field_values(
                appended,
                self.clone_fields(clone_indices),
            )
        self.append_and_prune(appended, selected_indices, dead_mask)
        self.removed_points = True
        self.new_round = False

    def remove_final_points(
        self,
        dead_mask: Bool[Tensor, " num_triangles"],
    ) -> None:
        """Remove final-stage dead triangles."""
        assert self.family_ops is not None
        self.replace_with_keep_mask(~dead_mask)
        self.removed_points = True
        self.new_round = False

    def empty_field_values(
        self,
        scene: TriangleSplattingScene,
    ) -> dict[str, Tensor]:
        """Return empty tensors for each topology field."""
        return {
            "triangle_vertices": torch.empty(
                (0, *scene.triangle_vertices.shape[1:]),
                dtype=scene.triangle_vertices.dtype,
                device=scene.triangle_vertices.device,
            ),
            "features_dc": torch.empty(
                (0, *scene.features_dc.shape[1:]),
                dtype=scene.features_dc.dtype,
                device=scene.features_dc.device,
            ),
            "features_rest": torch.empty(
                (0, *scene.features_rest.shape[1:]),
                dtype=scene.features_rest.dtype,
                device=scene.features_rest.device,
            ),
            "logit_opacity": torch.empty(
                (0, 1),
                dtype=scene.logit_opacity.dtype,
                device=scene.logit_opacity.device,
            ),
            "raw_sigma": torch.empty(
                (0, 1),
                dtype=scene.raw_sigma.dtype,
                device=scene.raw_sigma.device,
            ),
            "mask_logits": torch.empty(
                (0, 1),
                dtype=scene.mask_logits.dtype,
                device=scene.mask_logits.device,
            ),
        }

    def concat_field_values(
        self,
        left: dict[str, Tensor],
        right: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Concatenate two topology-field dictionaries."""
        return {
            name: torch.cat([left[name], right[name]], dim=0) for name in left
        }

    def append_and_prune(
        self,
        appended: dict[str, Tensor],
        selected_indices: Tensor,
        dead_mask: Tensor,
    ) -> None:
        """Append new rows, then remove source/dead rows in one replacement."""
        assert self.family_ops is not None
        scene = self.family_ops.scene
        grown_count = scene.num_triangles + int(
            appended["triangle_vertices"].shape[0]
        )
        remove_mask = torch.zeros(
            (grown_count,),
            dtype=torch.bool,
            device=scene.triangle_vertices.device,
        )
        remove_mask[selected_indices] = True
        remove_mask[: scene.num_triangles][dead_mask] = True
        keep_mask = ~remove_mask
        updates: dict[str, Tensor] = {}
        transforms: dict[str, Callable[[str, Tensor], Tensor]] = {}
        for name in TRIANGLE_SPLATTING_TOPOLOGY_PARAMETER_FIELD_NAMES:
            value = scene_topology_parameter_tensor(scene, name)
            grown_value = torch.cat([value, appended[name]], dim=0)
            updates[name] = (
                grown_value[keep_mask]
                .detach()
                .requires_grad_(value.requires_grad)
            )
            transforms[name] = (
                lambda _key, old_value, local_keep=keep_mask, local_append=appended[name]: (
                    torch.cat(
                        [old_value, torch.zeros_like(local_append)],
                        dim=0,
                    )[local_keep]
                )
            )
        self.install_updates(updates, transforms)

    def replace_with_keep_mask(self, keep_mask: Tensor) -> None:
        """Prune all topology fields with one keep mask."""
        assert self.family_ops is not None
        scene = self.family_ops.scene
        updates: dict[str, Tensor] = {}
        transforms: dict[str, Callable[[str, Tensor], Tensor]] = {}
        for name in TRIANGLE_SPLATTING_TOPOLOGY_PARAMETER_FIELD_NAMES:
            value = scene_topology_parameter_tensor(scene, name)
            updates[name] = (
                value[keep_mask].detach().requires_grad_(value.requires_grad)
            )
            transforms[name] = lambda _key, old_value, local_keep=keep_mask: (
                old_value[local_keep]
            )
        self.install_updates(updates, transforms)

    def install_updates(
        self,
        updates: dict[str, Tensor],
        transforms: dict[str, Callable[[str, Tensor], Tensor]] | None = None,
    ) -> None:
        """Install topology updates and refresh derived buffers/statistics."""
        assert self.family_ops is not None
        scene = self.family_ops.scene
        vertices_per_triangle = (
            TriangleSplattingScene.full_vertices_per_triangle(
                updates["triangle_vertices"]
            )
        )
        updates["vertices_per_triangle"] = vertices_per_triangle
        updates["triangle_vertex_offsets"] = (
            TriangleSplattingScene.vertex_offsets(vertices_per_triangle)
        )
        resolved_transforms = transforms or {
            name: (lambda _key, old_value: old_value) for name in updates
        }
        resolved_transforms.setdefault(
            "vertices_per_triangle",
            lambda _key, old_value: old_value,
        )
        resolved_transforms.setdefault(
            "triangle_vertex_offsets",
            lambda _key, old_value: old_value,
        )
        self.family_ops.replace_fields(updates, resolved_transforms)
        self.reset_statistics()
        self.begin_post_edit_visibility_round()


def register_triangle_splatting_family_ops() -> None:
    """Register Triangle Splatting topology operations with ember-core."""
    register_family_ops("triangle_splatting", TriangleSplattingFamilyOps)


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
