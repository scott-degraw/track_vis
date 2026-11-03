# track_vis

Plotly-based 3D event display utilities: particle tracks, energy voxels, and truth vertices.
Reads the SNO+ `imaging_reco` ntuple track/vertex format via `uproot`.

## Install

```bash
uv sync
```

## Usage

```python
from track_vis import plot_tracks, plot_voxels, plot_truth_vertices, voxel_mesh
```

`particle_name` uses the optional `particle` package for PDG-code-to-name lookups
(falls back to `f"PDG {pdg}"` if not installed). Install it with:

```bash
uv sync --extra particle
```

## Plotting functions

Tracks and vertices are columnar, mirroring the ntuple's branch layout: `pdg` is a
flat array (one entry per track/vertex), and per-step fields (`pos_x`, `pos_y`,
`pos_z`, `time`, `deposited_energy`) are jagged awkward arrays — outer index = track,
inner index = step. Vertex fields (`x`, `y`, `z`, `t`, `energy`) are flat, one entry
per vertex.

| Function | Returns | Description |
| --- | --- | --- |
| `load_event(path, index)` | `tuple[dict, dict]` | Opens an `imaging_reco` ntuple with `uproot` and returns `(tracks, vertices)` kwargs dicts for one event, ready to unpack into the functions below. |
| `plot_tracks(pdg, pos_x, pos_y, pos_z, time, deposited_energy, t_min, t_max)` | `list[go.Scatter3d]` | One line trace per track, colour-coded along its length by step time. |
| `plot_truth_vertices(pdg, x, y, z, t, energy, t_min, t_max)` | `list[go.Scatter3d]` | One marker trace per particle species, colour-coded by vertex time. |
| `plot_voxels(pos_x, pos_y, pos_z, deposited_energy, voxel_size=100.0)` | `list[go.Mesh3d]` | Bins deposited energy into a 3D grid of cubes of edge `voxel_size` and returns a single batched mesh trace coloured by per-voxel energy. |
| `voxel_mesh(centers, energies, voxel_size, ...)` | `go.Mesh3d` | Lower-level helper used by `plot_voxels`: builds one batched `Mesh3d` cube per row of `centers`/`energies`. |
| `time_colorscale(times, t_min, t_max, colorscale=COLORSCALE)` | `list[str]` | Maps an array of times to CSS colour strings via a Plotly colorscale. |
| `particle_name(pdg)` | `str` | PDG code to particle name, via the optional `particle` package (falls back to `f"PDG {pdg}"`). |

All trace lists are meant to be combined and passed to `go.Figure(data=...)`, e.g.:

```python
tracks, vertices = load_event(path, index=0)
fig = go.Figure(data=plot_tracks(**tracks, t_min=t_min, t_max=t_max) + plot_truth_vertices(**vertices, t_min=t_min, t_max=t_max))
```

## Volume viewer (three.js)

`track_vis.volume` renders a reconstructed voxel image as a GPU ray-marched volume
with truth tracks on top, as an HTML fragment for Jupyter or a Quarto page. Each
voxel is composited with its own opacity along every pixel's ray (exact voxel
traversal, or trilinear with the "smooth" toggle), so there is no triangle sorting,
and the image costs one byte per voxel instead of a mesh. Threshold, opacity,
max-intensity projection and track visibility are adjustable in the page, and
hovering reports the brightest voxel under the cursor.

```python
from IPython.display import HTML
from track_vis import track_spec, volume_html

tracks = [track_spec(points, "primary track", values=times), track_spec(axis, "reco axis", dashed=True)]
HTML(volume_html(centers=centers, energies=energies, voxel_size=100.0, tracks=tracks))
# or a dense image: volume_html(grid, origin, voxel_size, tracks=tracks)
```

three.js r147 comes from a pinned jsDelivr URL by default; `lib="inline"` embeds
the copy vendored in `track_vis/static/vendor` (MIT) for pages with no network.

## Example

`examples/synthetic_event.ipynb` builds a fake event (electron, proton, and gamma
from a common vertex) and renders it with `plot_tracks`, `plot_truth_vertices`, and
`plot_voxels`, then does the same for a real event read via `load_event` from a
SNO+ `imaging_reco` MC ntuple. Run it with:

```bash
uv sync --group dev
uv run jupyter notebook examples/synthetic_event.ipynb
```
