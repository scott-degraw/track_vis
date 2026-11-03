"""Interactive three.js event displays for SNO+ imaging MC: tracks, vertices and voxel images."""

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
