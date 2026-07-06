# track_vis

Plotly-based 3D event display utilities: particle tracks, energy voxels, and truth vertices.

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
