from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from marimo_config_gui.api import load_script_config
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "triangle_splatting" / "notebook.py"
TRIANGLE_NATIVE_SRC = (
    REPO_ROOT / "packages" / "ember-native-triangle-splatting" / "src"
)
if str(TRIANGLE_NATIVE_SRC) not in sys.path:
    sys.path.insert(0, str(TRIANGLE_NATIVE_SRC))


def load_triangle_splatting_preset(triangle_splatting_config_module, name: str):
    return load_preset_config(
        triangle_splatting_config_module.triangle_splatting_preset_catalog(),
        name,
    )


def load_triangle_splatting_script_config(
    triangle_splatting_config_module,
    args: list[str],
):
    return load_script_config(
        triangle_splatting_config_module.TriangleSplattingExperimentConfig,
        args=args,
        presets=(
            triangle_splatting_config_module.triangle_splatting_preset_catalog()
        ),
    )


def load_triangle_splatting_notebook():
    spec = importlib.util.spec_from_file_location(
        "papers.triangle_splatting.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_triangle_splatting_garden_preset_matches_upstream_defaults() -> None:
    module = load_triangle_splatting_notebook()
    config = load_triangle_splatting_preset(module, "garden")

    assert config.scene.path.name == "garden"
    assert config.scene.path.parent.name == "360"
    assert config.scene.image_directory == "images_4"
    assert config.training.runtime.max_steps == 30_000
    assert config.training.initialization.initial_opacity == 0.28
    assert config.training.initialization.initial_triangle_size == 2.23
    assert config.training.initialization.initial_sigma == 1.16
    assert config.training.optimization.feature_lr == 0.0025
    assert config.training.optimization.opacity_lr == 0.014
    assert config.training.optimization.sigma_lr == 0.0008
    assert config.training.optimization.triangle_vertices_lr_init == 0.0018
    assert config.training.loss.lambda_dssim == 0.2
    assert config.training.loss.lambda_opacity == 0.0055
    assert config.training.loss.lambda_normals == 0.0001
    assert config.training.loss.lambda_dist == 0.0
    assert config.training.loss.lambda_size == 1e-8
    assert config.training.densification.densification_interval == 500
    assert config.training.densification.densify_from_iter == 500
    assert config.training.densification.densify_until_iter == 25_000
    assert config.training.densification.max_shapes == 5_200_000
    assert config.training.densification.split_size == 24.0
    assert config.training.densification.proba_distr == 2
    assert config.training.densification.outdoor is True


def test_triangle_splatting_resolved_training_config_targets_native_runtime() -> (
    None
):
    module = load_triangle_splatting_notebook()
    config = load_triangle_splatting_preset(module, "garden")
    training_config = module.resolve_training_config(config)

    assert training_config.render.backend == "triangle_splatting.core"
    assert training_config.render.backend_options == {
        "near_plane": 0.01,
        "far_plane": 100.0,
        "background_color": [0.0, 0.0, 0.0],
        "clamp_output": False,
        "depth_mode": "median",
    }
    assert training_config.render.return_alpha is True
    assert training_config.render.return_depth is True
    assert training_config.render.return_normals is True
    assert (
        training_config.initialization.initializer.target
        == "ember_native_triangle_splatting.triangle_splatting."
        "initialize_triangle_splatting_model_from_scene_record"
    )
    assert training_config.initialization.initializer.context_kwargs == {
        "device": "device"
    }
    assert [
        group.target.name
        for group in training_config.optimization.parameter_groups
    ] == [
        "features_dc",
        "features_rest",
        "logit_opacity",
        "triangle_vertices",
        "raw_sigma",
    ]
    assert (
        training_config.loss.target.target
        == "ember_native_triangle_splatting.triangle_splatting."
        "triangle_splatting_loss"
    )
    assert (
        training_config.densification.builders[0].target
        == "ember_native_triangle_splatting.triangle_splatting."
        "TriangleSplattingDensification"
    )


def test_triangle_splatting_scene_loader_uses_images_4() -> None:
    module = load_triangle_splatting_notebook()
    config = load_triangle_splatting_preset(module, "garden")
    scene_config = module.build_scene_load_config(config)

    assert scene_config.image_root.name == "images_4"
    assert scene_config.image_root.parent.name == "garden"


def test_triangle_splatting_script_loader_applies_cli_overrides() -> None:
    module = load_triangle_splatting_notebook()
    loaded = load_triangle_splatting_script_config(
        module,
        args=[
            "--preset",
            "garden_debug_val",
            "--training.runtime.max-steps",
            "5",
            "--training.profiler.enabled",
            "True",
            "--training.densification.max-shapes",
            "1234",
        ],
    )

    assert isinstance(loaded, module.TriangleSplattingExperimentConfig)
    assert loaded.preset == "garden_debug_val"
    assert loaded.training.runtime.max_steps == 5
    assert loaded.training.profiler.enabled is True
    assert loaded.training.densification.max_shapes == 1234


def test_triangle_splatting_script_loader_replays_json_config(
    tmp_path: Path,
) -> None:
    module = load_triangle_splatting_notebook()
    config = load_triangle_splatting_preset(module, "garden")
    json_path = tmp_path / "triangle_splatting_config.json"
    json_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))

    loaded = load_triangle_splatting_script_config(
        module, args=[str(json_path)]
    )

    assert isinstance(loaded, module.TriangleSplattingExperimentConfig)
    assert loaded == config


def test_triangle_splatting_marimo_graph_initializes() -> None:
    module = load_triangle_splatting_notebook()

    module.app._maybe_initialize()
