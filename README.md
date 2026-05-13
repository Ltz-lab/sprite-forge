---
title: sprite-forge
emoji: 🎮
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.0
app_file: app.py
pinned: false
license: mit
---

# 🎮 sprite-forge

Type a character description → get a 4-direction walk-cycle sprite sheet.

```
"armored knight with red sash" → 🗡️ knight_sheet.png + knight.json
```

Powered by **Flux.1-schnell** (Apache-2.0) + **ControlNet Union** on HuggingFace ZeroGPU.

### Features

- 5 art styles: pixel-art, painted, chibi, realistic, cel-shaded
- 4 or 8 directions × 4-8 frames per row
- Single-file preview + downloadable JSON atlas with frame coordinates
- Ready for Phaser, Godot, Unity, Aseprite

### Usage

1. Type a character prompt
2. Choose style and grid size
3. Click Generate
4. Download your sprite sheet + atlas

### License

MIT. Model weights are under their own licenses (Flux.1-schnell is Apache-2.0).
