# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (core deps only)
uv sync

# Install with the optional PDG-name lookup dependency
uv sync --extra particle

# Install with dev tooling (jupyter, pre-commit, nbstripout, pytest)
uv sync --group dev

# Lint + format
ruff check . && ruff format .

# Run tests (Python side: spec building, binning, colour scales, HTML output)
uv run pytest

# Run the example notebook end-to-end and write outputs in place
uv run jupyter nbconvert --to notebook --execute --inplace examples/synthetic_event.ipynb

# Open the example notebook interactively
uv run jupyter notebook examples/synthetic_event.ipynb
```

Pre-commit is configured with a single hook, `nbstripout`, which strips notebook
outputs from `examples/*.ipynb` at commit time. Run `uv run pre-commit install`
once per clone to activate it. If you re-execute a notebook to check its outputs
render correctly, expect the working-tree file to shrink back down when it's
next committed — that's the hook doing its job, not a bug.

## Architecture

A small three.js event-display library for 3D SNO+ events. Python builds a JSON
scene description; the browser draws it.

- `track_vis/display.py`: `EventDisplay`, the only scene type. `add_volume`,
  `add_deposits`, `add_tracks`, `add_vertices`, `add_line`, `add_points` append
  layers (each returns `self` for chaining); `spec()` is the JSON scene;
  `to_html(lib=)` wraps it in a fragment of `<script>`s + a container div;
  `_repr_html_` makes it render in Jupyter/Quarto; `write_html` writes a page.
  Also `voxelize` / `bin_deposits` (sparse/points -> dense `(grid, origin)`).
- `track_vis/static/event_display.js`: the renderer (`window.TrackVis.render(el, spec)`).
  Volumes are `uint8` 3D textures ray-marched in a GLSL3 fragment shader: exact
  voxel traversal (Amanatides & Woo) in sharp mode, quarter-voxel jittered
  sampling in smooth (trilinear) mode, plus a max-intensity mode. All volumes
  share one set of style uniforms (threshold/density/gamma sliders). Lines are
  `LineSegments2` (one per layer); markers are `THREE.Points` with a round
  shader. Hover is done in JS (nearest projected hover point within 10 px, else
  a CPU march for the brightest voxel). Axis tick labels are HTML placed in
  screen space every frame. Rendering is on demand (no animation loop).
- `track_vis/static/vendor/`: three.js r147 classic (non-module) builds, vendored
  for `lib="inline"`; `lib="cdn"` loads the same files from pinned jsDelivr URLs.
  r147 is the last release with `examples/js` classic scripts, which avoids ES
  modules and import maps (kept simple for CSP-restricted pages such as
  claude.ai artifacts). Don't bump it without switching to module loading.
- `track_vis/colors.py`: built-in colour scales as hex stops (no Plotly).
- `track_vis/io.py`: `load_event`, `particle_name`.

Volume data is quantised to `uint8` relative to the volume's `e_max` and sent
x-fastest (`tobytes(order="F")` of an `(nx, ny, nz)` array). Lines, markers and
hover points are base64 `float32`.

**Data shape is columnar, not a nested record schema**, and it directly mirrors
the branch layout of the SNO+ `imaging_reco` ntuple (see `load_event`):
- Tracks: a flat `pdg` array (one entry per track) plus jagged awkward arrays
  `pos_x`, `pos_y`, `pos_z`, `time`, `deposited_energy` — outer index = track,
  inner index = step. `add_tracks` and `add_deposits` both take these same
  columns (mirrors ROOT's `trackPDG`/`trackPosX/Y/Z`/`trackTime`/`trackDep`
  branches, which are `vector<vector<T>>` aligned by track index with no
  separate track-id needed).
- Vertices: flat arrays `pdg`, `x`, `y`, `z`, `t`, `energy`, one entry per vertex
  (mirrors ROOT's plural `mcpdgs`/`mcxs`/`mcys`/`mczs`/`mcts`/`mckes` branches,
  which support multiple primaries per event).
- Units throughout: position in mm, time in ns, energy in MeV.

`load_event(path, index)` is the bridge from a ROOT file to this format: it opens
one event with `uproot` and returns `(tracks, vertices)` as two kwargs dicts, so
call sites look like `disp.add_tracks(**tracks, t_min=..., t_max=...)`.

`add_tracks` and `add_vertices` colour by time over the caller's `t_min`/`t_max`
and share one "Time (ns)" colour bar (a later call replaces it), so pass the same
bounds to both.

`particle_name(pdg)` degrades gracefully: it uses the optional `particle` package
for PDG-to-name lookups if installed (`uv sync --extra particle`), otherwise
falls back to `f"PDG {pdg}"`. Don't assume `particle` is present.

`examples/synthetic_event.ipynb` is the canonical worked example and doubles as
a manual integration test: it builds a fake event with the columnar schema above,
then loads and plots one real event via `load_event` from a SNO+ MC ntuple at
`/data/snoplus2/degraw/imaging_reco/mc/electron_uniform_fiducial/root/job_0.ntuple.root`,
and ends with a toy reconstructed volume. When changing the `EventDisplay` API, the
spec format or the columnar schema, update this notebook and re-execute it to
confirm the change works end-to-end; check the JS side in a real browser (WebGL)
since pytest only covers the Python side.
