"""
track_vis

Interactive three.js event displays for SNO+ imaging MC.

`EventDisplay` draws truth tracks, vertices and voxel images (binned deposits
or reconstructed images) as a self-contained HTML fragment for Jupyter,
Quarto or a standalone page. `load_event` reads events from ``imaging_reco``
ntuples.

Package layout (Python unless noted):

- ``display.py``: `EventDisplay`, the user-facing API.
- ``scene.py``: `Scene` and its layers, the data handed to the browser.
- ``browser.py``: wraps a scene in HTML; the only module that touches JavaScript.
- ``voxels.py``: dense grids from sparse voxels or point deposits.
- ``colors.py``: built-in colour scales.
- ``io.py``: reading ntuples and naming PDG codes.
- ``js/``: the JavaScript renderer and vendored three.js (see ``js/README.md``).
"""

from track_vis.colors import COLORSCALES, colorscale_stops, sample_colorscale
from track_vis.display import EventDisplay
from track_vis.io import load_event, particle_name
from track_vis.scene import Scene
from track_vis.voxels import bin_deposits, voxelize

__all__ = [
    "COLORSCALES",
    "EventDisplay",
    "Scene",
    "bin_deposits",
    "colorscale_stops",
    "load_event",
    "particle_name",
    "sample_colorscale",
    "voxelize",
]
