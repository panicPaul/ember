from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from ember_core.core import BACKEND_REGISTRY, CameraState, render
from ember_core.core.contracts import TriangleSplattingScene
from ember_core.training import ParameterTargetSpec
from ember_core.training.runtime import OptimizerBinding

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIANGLE_NATIVE_SRC = (
    REPO_ROOT / "packages" / "ember-native-triangle-splatting" / "src"
)
if str(TRIANGLE_NATIVE_SRC) not in sys.path:
    sys.path.insert(0, str(TRIANGLE_NATIVE_SRC))

from ember_native_triangle_splatting import register
from ember_native_triangle_splatting.triangle_splatting import (
    TriangleSplattingDensification,
    TriangleSplattingFamilyOps,
    TriangleSplattingNativeRenderOptions,
    TriangleSplattingNativeRenderOutput,
    render_triangle_splatting_native,
)


def triangle_scene(device: torch.device) -> TriangleSplattingScene:
    colors = torch.tensor(
        [
            [0.9, 0.2, 0.2],
            [0.2, 0.9, 0.2],
            [0.2, 0.2, 0.9],
        ],
        dtype=torch.float32,
        device=device,
    )
    features_dc = ((colors - 0.5) / 0.28209479177387814)[:, None, :]
    return TriangleSplattingScene(
        triangle_vertices=torch.tensor(
            [
                [[-0.35, -0.35, 5.0], [0.35, -0.35, 5.0], [0.0, 0.35, 5.0]],
                [[-0.2, -0.2, 5.4], [0.4, -0.15, 5.4], [0.1, 0.45, 5.4]],
                [[-0.45, 0.05, 4.8], [0.15, 0.1, 4.8], [-0.2, 0.55, 4.8]],
            ],
            dtype=torch.float32,
            device=device,
        ),
        raw_sigma=torch.full(
            (3, 1),
            torch.log(torch.tensor(1.15)).item(),
            dtype=torch.float32,
            device=device,
        ),
        logit_opacity=torch.full(
            (3, 1),
            torch.logit(torch.tensor(0.8)).item(),
            dtype=torch.float32,
            device=device,
        ),
        features_dc=features_dc,
        features_rest=torch.empty(
            (3, 0, 3), dtype=torch.float32, device=device
        ),
        mask_logits=torch.ones((3, 1), dtype=torch.float32, device=device),
        sh_degree=0,
        active_sh_degree=0,
    )


def test_triangle_splatting_backend_registers() -> None:
    register()

    assert "triangle_splatting.core" in BACKEND_REGISTRY


def test_triangle_splatting_scene_contract_on_cpu() -> None:
    scene = triangle_scene(torch.device("cpu"))

    assert scene.scene_family == "triangle_splatting"
    assert scene.num_triangles == 3
    assert scene.flattened_triangle_vertices.shape == (9, 3)
    assert scene.features.shape == (3, 1, 3)
    assert TriangleSplattingScene.vertex_offsets(
        torch.empty((0,), dtype=torch.int32)
    ).shape == (0,)
    torch.testing.assert_close(
        scene.sigma,
        torch.full((3, 1), 1.16),
    )


def test_triangle_splatting_visibility_prune_waits_for_ready_round() -> None:
    scene = triangle_scene(torch.device("cpu"))
    state = SimpleNamespace(model=SimpleNamespace(scene=scene))
    densification = TriangleSplattingDensification()
    densification.family_ops = TriangleSplattingFamilyOps(state, [])
    densification.number_of_training_views = 100
    densification.ensure_statistics(scene)
    assert densification.triangle_area is not None
    assert densification.image_size is not None
    assert densification.importance_score is not None
    densification.importance_score.fill_(1.0)
    densification.image_size.fill_(1.0)

    early_mask = densification.dead_mask(before_final=True, step=1000)
    waiting_mask = densification.dead_mask(before_final=True, step=1500)
    densification.visibility_statistics_ready = True
    ready_mask = densification.dead_mask(before_final=True, step=1500)

    assert not early_mask.any()
    assert not waiting_mask.any()
    assert ready_mask.all()


def test_triangle_splatting_round_state_tracks_post_edit_views() -> None:
    scene = triangle_scene(torch.device("cpu"))
    state = SimpleNamespace(model=SimpleNamespace(scene=scene))
    densification = TriangleSplattingDensification()
    densification.family_ops = TriangleSplattingFamilyOps(state, [])
    densification.number_of_training_views = 2
    runtime = SimpleNamespace(all_cameras=lambda: (object(), object()))
    first_context = SimpleNamespace(
        runtime=runtime,
        batch=SimpleNamespace(
            frames=(
                SimpleNamespace(
                    sensor_id="camera",
                    frame_id="first",
                    camera_index=0,
                ),
            )
        ),
    )
    second_context = SimpleNamespace(
        runtime=runtime,
        batch=SimpleNamespace(
            frames=(
                SimpleNamespace(
                    sensor_id="camera",
                    frame_id="second",
                    camera_index=1,
                ),
            )
        ),
    )

    densification.begin_post_edit_visibility_round()
    densification.observe_training_round(first_context)
    densification.observe_training_round(second_context)
    assert densification.new_round
    assert not densification.visibility_statistics_ready

    densification.observe_training_round(first_context)
    densification.observe_training_round(second_context)
    assert not densification.new_round
    assert densification.visibility_statistics_ready


def test_triangle_splatting_append_and_prune_resizes_optimizer_state() -> None:
    scene = triangle_scene(torch.device("cpu"))
    scene.triangle_vertices.requires_grad_(True)
    state = SimpleNamespace(model=SimpleNamespace(scene=scene))
    optimizer = torch.optim.Adam([scene.triangle_vertices], lr=0.1)
    binding = OptimizerBinding(
        target=ParameterTargetSpec(scope="scene", name="triangle_vertices"),
        optimizer=optimizer,
        base_parameter=scene.triangle_vertices,
        field_name="triangle_vertices",
    )
    scene.triangle_vertices.sum().backward()
    binding.step()
    binding.zero_grad()

    densification = TriangleSplattingDensification()
    densification.family_ops = TriangleSplattingFamilyOps(state, [binding])
    appended = densification.split_fields(torch.tensor([0]))
    densification.append_and_prune(
        appended,
        selected_indices=torch.tensor([0]),
        dead_mask=torch.zeros(scene.num_triangles, dtype=torch.bool),
    )
    updated_parameter = optimizer.param_groups[0]["params"][0]
    updated_state = optimizer.state[updated_parameter]["exp_avg"]

    assert updated_parameter.shape[0] == scene.num_triangles
    assert updated_state.shape == updated_parameter.shape
    scene.triangle_vertices.sum().backward()
    binding.step()


def test_triangle_splatting_rejects_cpu_scene(cpu_camera: CameraState) -> None:
    register()
    scene = triangle_scene(torch.device("cpu"))

    with pytest.raises(ValueError, match="scene tensors on CUDA"):
        render_triangle_splatting_native(scene, cpu_camera)


@pytest.mark.backend
@pytest.mark.cuda
def test_triangle_splatting_render_outputs(cpu_camera: CameraState) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triangle Splatting native tests.")
    register()
    cuda_scene = triangle_scene(torch.device("cuda"))
    cuda_camera = cpu_camera.to(torch.device("cuda"))

    output = cast(
        TriangleSplattingNativeRenderOutput,
        render(
            cuda_scene,
            cuda_camera,
            backend="triangle_splatting.core",
            return_alpha=True,
            return_depth=True,
            return_normals=True,
            options=TriangleSplattingNativeRenderOptions(clamp_output=False),
        ),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.alphas.shape == (1, 32, 32)
    assert output.depth.shape == (1, 32, 32)
    assert output.normals.shape == (1, 32, 32, 3)
    assert output.radii.shape == (1, 3)
    assert output.screen_space_scale.shape == (1, 3)
    assert torch.isfinite(output.render).all()


@pytest.mark.backend
@pytest.mark.cuda
def test_triangle_splatting_render_backward(cpu_camera: CameraState) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triangle Splatting native tests.")
    register()
    cuda_scene = triangle_scene(torch.device("cuda"))
    cuda_camera = cpu_camera.to(torch.device("cuda"))
    for parameter in cuda_scene.parameters():
        parameter.requires_grad_(True)

    output = render_triangle_splatting_native(
        cuda_scene,
        cuda_camera,
        options=TriangleSplattingNativeRenderOptions(clamp_output=False),
    )
    loss = (
        output.render.sum()
        + output.alphas.sum()
        + output.depth.sum()
        + output.normals.sum()
    )

    loss.backward()

    assert cuda_scene.triangle_vertices.grad is not None
    assert cuda_scene.raw_sigma.grad is not None
    assert cuda_scene.logit_opacity.grad is not None
    assert cuda_scene.features_dc.grad is not None
    assert torch.isfinite(cuda_scene.triangle_vertices.grad).all()
    assert torch.isfinite(cuda_scene.raw_sigma.grad).all()
    assert torch.isfinite(cuda_scene.logit_opacity.grad).all()
    assert torch.isfinite(cuda_scene.features_dc.grad).all()
