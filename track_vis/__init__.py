"""
track_vis

Interactive three.js event displays for SNO+ imaging MC.

`EventDisplay` draws truth tracks, vertices and voxel images (binned deposits
or reconstructed images) as a self-contained HTML fragment for Jupyter,
Quarto or a standalone page. `load_event` reads events from ``imaging_reco``
ntuples.
"""

from track_vis.colors import COLORSCALES, colorscale_stops, sample_colorscale
from track_vis.display import EventDisplay, bin_deposits, voxelize
from track_vis.io import load_event, particle_name

__all__ = [
    "COLORSCALES",
    "EventDisplay",
    "bin_deposits",
    "colorscale_stops",
    "load_event",
    "particle_name",
    "sample_colorscale",
    "voxelize",
]
