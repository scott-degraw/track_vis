# Browser renderer (JavaScript)

This folder is the JavaScript half of track_vis. You do not need to read it to
use or change the Python code.

- `event_display.js`: draws one scene in a web page with three.js.
- `vendor/`: three.js r147 (MIT licence, `LICENSE-three.txt`), embedded in
  pages made with `lib="inline"`.

## How it connects to Python

The only Python module that touches this folder is `track_vis/browser.py`. It
puts `event_display.js` in the page and calls

```js
TrackVis.mount(divId, scene, scriptUrls)
```

where `scene` is the JSON from `Scene.to_json()` in `track_vis/scene.py`. That
file documents every field and how arrays are encoded, so it is the contract
between the two halves:

| To change... | Edit |
| --- | --- |
| what is drawn (colours, grouping, hover text, new physics inputs) | Python only: `display.py` |
| a field in the scene description | `scene.py`, and the matching reader in `event_display.js` |
| how something looks or behaves in the page (controls, axes, shaders) | `event_display.js` only |

## Layout of `event_display.js`

1. Shaders (GPU programs, written in GLSL) for volumes and markers.
2. The light and dark themes: `THEMES` (colours WebGL draws) and `STYLE`, one
   stylesheet for the page elements (classes `tv-*`, colours as CSS variables),
   added to the page once.
3. Small helpers (base64 decoding, colour scales, tick spacing).
4. Layer builders `addVolume`, `addLines`, `addPoints`: one per list in the scene.
5. `addAxes`, `addPanel`, `addTools`, `addColorbars`, `addHover`: axes and floor grid, controls, buttons, colour bars and hover read-outs.
6. `render(root, scene)`: puts the pieces together.
7. `mount(...)`: loads three.js if needed and calls `render` once the view scrolls into sight.

three.js is pinned to r147 because it is the last release with plain
(non-module) script builds, which load without import maps in restricted pages
such as claude.ai artifacts. Changing the version means switching to ES modules.
