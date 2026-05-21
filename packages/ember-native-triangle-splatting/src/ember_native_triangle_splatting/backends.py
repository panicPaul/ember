"""Typed backend refs for ember-native-triangle-splatting."""

from __future__ import annotations

from typing import Final

from ember_core.core.backend_refs import BackendRef
from ember_core.core.contracts import TriangleSplattingScene
from ember_core.core.keys import BackendId

from ember_native_triangle_splatting.triangle_splatting.renderer import (
    TriangleSplattingNativeRenderOptions,
    TriangleSplattingNativeRenderOutput,
)

TRIANGLE_SPLATTING_CORE: Final[
    BackendRef[
        TriangleSplattingScene,
        TriangleSplattingNativeRenderOptions,
        TriangleSplattingNativeRenderOutput,
    ]
] = BackendRef(
    id=BackendId("triangle_splatting", "core"),
    scene_type=TriangleSplattingScene,
    options_type=TriangleSplattingNativeRenderOptions,
    output_type=TriangleSplattingNativeRenderOutput,
)

__all__ = ["TRIANGLE_SPLATTING_CORE"]
