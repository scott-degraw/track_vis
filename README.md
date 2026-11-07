# track_vis

Interactive 3D event displays for SNO+ imaging MC, drawn in the browser with three.js:
particle tracks, truth vertices, and voxel images (binned deposits or reconstructed
MLEM images). Reads the SNO+ `imaging_reco` ntuple track/vertex format via `uproot`.

A display is one self-contained HTML fragment. It renders inline in Jupyter, in a
rendered Quarto page, or as a standalone file. Voxel images are uploaded to the GPU as
3D textures and ray-marched per pixel, so every voxel has its own opacity, nothing
pops or z-fights while rotating, and an image costs one byte per voxel.

## Install

```bash
uv sync
```

`particle_name` uses the optional `particle` package for PDG-code-to-name lookups
(falls back to `f"PDG {pdg}"` if not installed). Install it with:

```bash
uv sync --extra particle
```

## Usage

```python
from track_vis import EventDisplay, load_event

tracks, vertices = load_event(path, index=0)
t = ak.to_numpy(ak.flatten(tracks["time"]))

disp = (
    EventDisplay(title="Event 0")
    .add_deposits(tracks["pos_x"], tracks["pos_y"], tracks["pos_z"], tracks["deposited_energy"], voxel_size=50)
    .add_tracks(**tracks, t_min=t.min(), t_max=t.max())
    .add_vertices(**vertices, t_min=t.min(), t_max=t.max())
)
disp  # renders in Jupyter / Quarto
disp.write_html("event.html")  # standalone page, works offline
```

A reconstructed image goes in with `add_volume`, either dense or sparse:

```python
disp.add_volume(grid, origin=origin, voxel_size=100)  # grid[ix, iy, iz], origin = min corner
disp.add_volume(centers=centers, energies=energies, voxel_size=100)  # regular-grid voxel centres
disp.add_line(axis_points, "reconstructed axis", color="#2ca02c", dashed=True)
```

In the page you can drag to rotate, scroll to zoom and right-drag to pan. The title sits
in the top-left corner; the controls start hidden: click "Options" next to it to show them.
The layer list in the panel doubles as the legend. Buttons in the bottom-right corner
reset the camera and go full screen. `EventDisplay(theme="dark")` gives a dark
background, on which faint voxels and the bright end of the colour scales stand out. Sliders set
the volume threshold, opacity (density) and opacity curve (gamma). Checkboxes switch
on trilinear smoothing or a maximum-intensity projection, and toggle each layer.
Hovering shows the time and deposit of the nearest track step or vertex, or else the
brightest voxel under the cursor. Three box edges carry labelled mm ticks, with a faint
grid on the floor of the box at the same spacing.

## API

Tracks and vertices are columnar, mirroring the ntuple's branch layout: `pdg` is a
flat array (one entry per track/vertex), and per-step fields (`pos_x`, `pos_y`,
`pos_z`, `time`, `deposited_energy`) are jagged awkward arrays (outer index = track,
inner index = step). Vertex fields (`x`, `y`, `z`, `t`, `energy`) are flat, one entry
per vertex. Units: mm, ns, MeV.

| Method / function | Description |
| --- | --- |
| `EventDisplay(title, width, height, threshold, density, gamma, smooth, bounds, theme)` | A scene. `theme` is `"light"` (default) or `"dark"`. The volume style arguments set the initial slider values; `bounds=[[x0, y0, z0], [x1, y1, z1]]` fixes the view box (default: fit all layers). Every `add_*` returns the display, so calls chain. |
| `.add_tracks(pdg, pos_x, pos_y, pos_z, time, deposited_energy, t_min, t_max)` | Truth tracks coloured by step time, one toggleable layer per species. |
| `.add_vertices(pdg, x, y, z, t, energy, t_min, t_max)` | Truth vertex markers coloured by time, one layer per species. Use the same `t_min`/`t_max` as the tracks to share the time colour bar. |
| `.add_deposits(pos_x, pos_y, pos_z, deposited_energy, voxel_size=100)` | Bins per-step deposits into cubes aligned to multiples of `voxel_size` and adds them as a volume. |
| `.add_volume(grid, origin, voxel_size)` or `.add_volume(centers=, energies=, voxel_size=)` | A voxel image, ray-marched on the GPU. `e_max` fixes the colour/opacity scale for comparing events. |
| `.add_line(points, name, color= / values=, dashed=, width=)` | A polyline (or a list of them in one layer), in one colour or coloured by per-point values. |
| `.add_points(points, name, color= / values=, size=, colorbar_title=)` | Round markers; with `values` and `colorbar_title` they get their own colour bar. |
| `.to_html(lib="cdn")`, `.write_html(path, lib="inline")` | HTML output. `lib="cdn"` loads three.js r147 from a pinned jsDelivr URL; `lib="inline"` embeds the copy vendored in `track_vis/js/vendor` (MIT). |
| `.scene` | The `Scene` built so far: plain dataclasses of NumPy arrays (see `scene.py`). |
| `DisplayMenu(displays, labels)` | Several displays in one view with a drop-down and previous / next buttons to pick one (e.g. one per event). Only the shown one holds a WebGL context. Same `to_html`/`write_html`. |
| `load_event(path, index)` | Opens an `imaging_reco` ntuple and returns `(tracks, vertices)` kwargs dicts. |
| `voxelize(centers, values, voxel_size)`, `bin_deposits(points, energies, voxel_size)` | Dense `(grid, origin)` from sparse voxels or point deposits. |
| `sample_colorscale(name, u)`, `COLORSCALES` | Built-in colour scales: plasma, viridis, inferno, magma, cividis, turbo. |
| `particle_name(pdg)` | PDG code to name, via the optional `particle` package. |

Browsers allow roughly 16 live WebGL contexts per page, so each display initialises
only when it scrolls near the viewport.

## Code layout

The Python and JavaScript halves are separate; you can read and change the
Python without knowing JavaScript.

| Module | Role |
| --- | --- |
| `display.py` | `EventDisplay`: turns tracks, vertices, deposits and images into scene layers. `DisplayMenu`: a drop-down of displays. |
| `scene.py` | The scene dataclasses and their JSON encoding: the contract with the browser. |
| `browser.py` | Wraps a scene in HTML. The only Python module that touches JavaScript. |
| `voxels.py`, `colors.py`, `io.py` | Grids, colour scales, ntuple reading. |
| `js/` | The JavaScript renderer and vendored three.js; see `js/README.md`. |

## Example

`examples/synthetic_event.ipynb` builds a fake event (electron, proton, and gamma
from a common vertex) and displays it with deposits, tracks and vertices, then does
the same for a real event read via `load_event` from a SNO+ `imaging_reco` MC
ntuple, and ends with a toy reconstructed image. Run it with:

```bash
uv sync --group dev
uv run jupyter notebook examples/synthetic_event.ipynb
```

## Tests

```bash
uv run pytest
```

`tests/test_render_in_browser.py` also opens a page in headless Chromium and
checks that the JavaScript renderer draws without errors. It is skipped unless
Playwright's browser is installed:

```bash
uv run playwright install chromium
TRACK_VIS_REQUIRE_BROWSER=1 uv run pytest tests/test_render_in_browser.py
```

CI (`.github/workflows/ci.yml`) runs ruff, pytest on Python 3.10 and 3.13, and
this browser test.
