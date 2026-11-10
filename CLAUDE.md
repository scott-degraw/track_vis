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

# Run tests (Python side: display, scene encoding, HTML output, voxels, colours)
uv run pytest

# Browser smoke test of the JS renderer (skipped unless Chromium is installed)
uv run playwright install chromium
TRACK_VIS_REQUIRE_BROWSER=1 uv run pytest tests/test_render_in_browser.py

# Write a gallery of synthetic event pages (light or dark) to look at the renderer
uv run python examples/gallery.py /tmp/gallery --theme dark

# Render the interactive tutorial to one HTML page (live 3D views)
uv run jupyter nbconvert --to html --execute examples/tutorial.ipynb --output-dir /tmp

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

A small three.js event-display library for 3D SNO+ events. The Python and
JavaScript halves are kept apart so the Python can be read on its own:

- `track_vis/display.py`: `EventDisplay`, the user API, and `DisplayMenu` (a
  drop-down of displays sharing one view). Turns physics inputs
  into `Scene` layers (colours, species grouping, hover text). No JSON or JS.
- `track_vis/scene.py`: `Scene`, `VolumeLayer`, `LineLayer`, `PointLayer`,
  `Colorbar`, `VolumeStyle` dataclasses holding NumPy data, plus `to_json()`
  and the array encodings. This is the contract with the browser: every field
  the JS reads is documented here.
- `track_vis/browser.py`: the only Python module that touches JavaScript.
  Wraps a scene in an HTML fragment (`scene_html`) or page (`page_html`):
  the renderer script, a div, and `TrackVis.mount(id, scene_json, urls)`
  (`menu_html` and `TrackVis.mountMenu` for a `DisplayMenu`).
- `track_vis/voxels.py` (`voxelize`, `bin_deposits`), `colors.py` (colour
  scales, no Plotly), `io.py` (`load_event`, `particle_name`).
- `track_vis/js/`: `event_display.js` (the renderer) and `vendor/` (three.js
  r147 classic builds, MIT). `js/README.md` explains its layout and which file
  to edit for which change.

Volumes are `uint8` 3D textures ray-marched in a GLSL3 fragment shader: exact
voxel traversal (Amanatides & Woo) in sharp mode, quarter-voxel jittered
sampling in smooth mode, plus maximum-intensity projection. All volumes share
the threshold/density/gamma sliders. Page elements are styled by one stylesheet
(`STYLE`, classes `tv-*`) with light and dark CSS variables; `EventDisplay(theme=...)`
picks one, and `THEMES` holds the matching WebGL colours. Clicking a colour bar
swaps its scale in the page: line and marker layers carry a `ScaleLink` (their
positions in [0, 1] on a named colour bar) so the JS can resample their colours. three.js stays on r147, the last release
with classic (non-module) scripts, which load without import maps in
CSP-restricted pages such as claude.ai artifacts.

Tests (`tests/`) mirror the modules: `test_display.py` checks Scene layers
without any HTML, `test_scene.py` the encodings, `test_browser.py` the HTML.

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

`examples/tutorial.ipynb` is the user tutorial (synthetic data only; `tests/test_tutorial.py`
runs its code cells). When changing the `EventDisplay` API, update it too.

`examples/synthetic_event.ipynb` is the canonical worked example and doubles as
a manual integration test: it builds a fake event with the columnar schema above,
then loads and plots one real event via `load_event` from a SNO+ MC ntuple at
`/data/snoplus2/degraw/imaging_reco/mc/electron_uniform_fiducial/root/job_0.ntuple.root`,
and ends with a toy reconstructed volume. When changing the `EventDisplay` API, the
spec format or the columnar schema, update this notebook and re-execute it to
confirm the change works end-to-end; check the JS side in a real browser (WebGL)
since pytest only covers the Python side.
