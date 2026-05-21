Triangle Splatting Native Staging
=================================

This directory stages the CUDA rasterizer used by Triangle Splatting into the
Ember-owned `ember_native_triangle_splatting.triangle_splatting` runtime.

The staged source comes from the pinned reference:

- `third_party/triangle-splatting`
- `submodules/diff-triangle-rasterization`
- commit `6d61f4c0a571c635e37ffb740250706fa30a5541`

Do not import the upstream Python package at runtime. Package-local bindings
expose descriptive FasterGS-style stage wrappers around the staged C++/CUDA
functions, and Python runtime modules register thin `torch.library.custom_op`
dispatchers.
