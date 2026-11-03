# Changelog

## 0.2.0

A rewrite on three.js. **Breaking:** the Plotly functions are gone.

- New `EventDisplay` API (`add_volume`, `add_deposits`, `add_tracks`,
  `add_vertices`, `add_line`, `add_points`, `to_html`, `write_html`). Replaces
  `plot_tracks`, `plot_truth_vertices`, `plot_voxels` and the Plotly figures.
- Voxel images are ray-marched on the GPU (sharp cubes, smooth, or
  maximum-intensity projection) with shared threshold/density/gamma sliders.
- Axes with mm tick labels, per-layer toggles, hover read-outs, colour bars.
- Pages embed in Jupyter, Quarto and claude.ai artifacts; `lib="inline"`
  makes a page that works offline.
- `DisplayMenu` pages through several displays (e.g. events) in one view, and
  `EventDisplay(bounds=...)` fixes the view box.
- Plotly dependency removed.
- CI: ruff, pytest and a headless-browser smoke test.

## 0.1.0

Plotly event display.
