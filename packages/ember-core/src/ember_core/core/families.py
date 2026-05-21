"""Core scene-family keys."""

from __future__ import annotations

from typing import Final

from ember_core.core.keys import SceneFamilyKey, serialized_id

GAUSSIAN: Final = SceneFamilyKey("core", "gaussian")
SPARSE_VOXEL: Final = SceneFamilyKey("core", "sparse_voxel")
FOAM: Final = SceneFamilyKey("core", "foam")
RADFOAM: Final = SceneFamilyKey("core", "radfoam")
POWERFOAM: Final = SceneFamilyKey("core", "powerfoam")
TRIANGLE_SPLATTING: Final = SceneFamilyKey("core", "triangle_splatting")


def scene_family_id(value: str | SceneFamilyKey) -> str:
    """Return the runtime scene-family string."""
    serialized = serialized_id(value)
    if serialized == GAUSSIAN.serialized:
        return "gaussian"
    if serialized == SPARSE_VOXEL.serialized:
        return "sparse_voxel"
    if serialized == TRIANGLE_SPLATTING.serialized:
        return "triangle_splatting"
    if serialized in {
        FOAM.serialized,
        RADFOAM.serialized,
        POWERFOAM.serialized,
        "foam",
        "radfoam",
        "powerfoam",
    }:
        return "foam"
    return serialized
