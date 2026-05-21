"""Native Triangle Splatting backend for Ember."""

from ember_native_triangle_splatting._version import __version__


def register() -> None:
    """Register all native Triangle Splatting backends."""
    from ember_native_triangle_splatting.triangle_splatting import (
        register as register_core,
    )
    from ember_native_triangle_splatting.triangle_splatting.training import (
        register_triangle_splatting_family_ops,
    )

    register_triangle_splatting_family_ops()
    register_core()


__all__ = ["__version__", "register"]
