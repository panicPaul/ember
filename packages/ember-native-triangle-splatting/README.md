# ember-native-triangle-splatting

Native Ember backend for Triangle Splatting.

The package-local CUDA runtime stages the rasterizer from the pinned
`third_party/triangle-splatting` reference and exposes an Ember-owned renderer,
scene contract, and paper-training helpers.
