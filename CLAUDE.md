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

# Run tests (no test suite exists yet)
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

This is a small, single-module Plotly plotting library (`track_vis/viz.py`) for
3D SNO+ event displays. There is no class hierarchy — every plotting function is a
free function that returns a `list` of Plotly trace objects (`go.Scatter3d` /
`go.Mesh3d`), meant to be concatenated and passed into `go.Figure(data=...)`. There
is no `Figure`-building helper in the library itself — layout (titles, axis
labels, `aspectmode`) is left to the caller/notebook.

**Data shape is columnar, not a nested record schema**, and it directly mirrors
the branch layout of the SNO+ `imaging_reco` ntuple (see `load_event`):
- Tracks: a flat `pdg` array (one entry per track) plus jagged awkward arrays
  `pos_x`, `pos_y`, `pos_z`, `time`, `deposited_energy` — outer index = track,
  inner index = step. `plot_tracks` and `plot_voxels` both take these same
  columns (mirrors ROOT's `trackPDG`/`trackPosX/Y/Z`/`trackTime`/`trackDep`
  branches, which are `vector<vector<T>>` aligned by track index with no
  separate track-id needed).
- Vertices: flat arrays `pdg`, `x`, `y`, `z`, `t`, `energy`, one entry per vertex
  (mirrors ROOT's plural `mcpdgs`/`mcxs`/`mcys`/`mczs`/`mcts`/`mckes` branches,
  which support multiple primaries per event).
- Units throughout: position in mm, time in ns, energy in MeV.

`load_event(path, index)` is the bridge from a ROOT file to this format: it opens
one event with `uproot` and returns `(tracks, vertices)` as two kwargs dicts, so
call sites look like `plot_tracks(**tracks, t_min=..., t_max=...)`.

Both `plot_tracks` and `plot_truth_vertices` color by time using the same
`t_min`/`t_max` bounds passed in by the caller (computed once from the flattened
step/vertex times) — this keeps their colorbars on a shared scale. `plot_voxels`
bins per-step deposited energy into a 3D grid and renders it as a *single*
batched `go.Mesh3d` (via the lower-level `voxel_mesh` helper) rather than one
mesh per voxel, for render performance.

`particle_name(pdg)` degrades gracefully: it uses the optional `particle` package
for PDG-to-name lookups if installed (`uv sync --extra particle`), otherwise
falls back to `f"PDG {pdg}"`. Don't assume `particle` is present.

`examples/synthetic_event.ipynb` is the canonical worked example and doubles as
a manual integration test: it builds a fake event with the columnar schema above,
then loads and plots one real event via `load_event` from a SNO+ MC ntuple at
`/data/snoplus2/degraw/imaging_reco/mc/electron_uniform_fiducial/root/job_0.ntuple.root`.
When changing any plotting function's signature or the columnar schema, update
this notebook and re-execute it to confirm the change works end-to-end.
