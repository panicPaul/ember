"""JIT extension loader for Triangle Splatting CUDA stages."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ember_core.native.torch_extensions import load_torch_extension


@lru_cache(maxsize=1)
def load_extension() -> Any:
    """Compile and load the Triangle Splatting CUDA extension."""
    native_root = Path(__file__).resolve().parent.parent / "native"
    upstream_root = native_root / "upstream"
    rasterizer_root = upstream_root / "cuda_rasterizer"
    return load_torch_extension(
        name="ember_triangle_splatting_native_ext",
        sources=[
            str(native_root / "bindings.cpp"),
            str(upstream_root / "rasterize_points.cu"),
            str(rasterizer_root / "rasterizer_impl.cu"),
            str(rasterizer_root / "forward.cu"),
            str(rasterizer_root / "backward.cu"),
            str(rasterizer_root / "utils.cu"),
        ],
        extra_include_paths=[
            str(native_root),
            str(upstream_root),
            str(rasterizer_root),
            str(upstream_root / "third_party" / "glm"),
        ],
        extra_cflags=[
            "-O3",
            "-std=c++17",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "-std=c++17",
            "--extended-lambda",
            "-Xcompiler=-fvisibility=hidden",
            "-Xcompiler=-fvisibility-inlines-hidden",
        ],
        with_cuda=True,
        verbose=False,
    )
