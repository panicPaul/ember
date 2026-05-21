# Triangle Splatting

This notebook ports the official Triangle Splatting Garden training recipe into
Ember. It uses the pinned upstream repository in `third_party/triangle-splatting`
for parity checks and the package-local native rasterizer in
`ember-native-triangle-splatting` for runtime training.

The default Garden preset mirrors the upstream command:

```bash
python train.py -s <MipNeRF360>/garden -i images_4 --eval --outdoor --max_shapes 5200000
```
