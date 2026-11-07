"""
scene.py

The scene description that Python hands to the browser renderer.

This module is the contract between the Python and JavaScript halves of
track_vis. `EventDisplay` fills a `Scene` with plain NumPy data; `Scene.to_json`
turns it into the JSON object that ``js/event_display.js`` draws, and
``browser.py`` wraps that in HTML. No other Python module touches the
JavaScript side.

All positions are in mm. Colours are RGB in [0, 1] per point, or colour scale
stops ``[[position, "#rrggbb"], ...]`` (see `colors.colorscale_stops`).

Encoding
--------
Large arrays are sent as base64 strings rather than JSON number lists, which
keeps pages small and fast to load:

- float arrays (positions, colours) as little-endian ``float32``;
- volume images as ``uint8``, each voxel ``round(255 * value / e_max)``
  clipped to [0, 255], flattened with x varying fastest (Fortran order of an
  ``(nx, ny, nz)`` array), as WebGL 3D textures expect.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np

from track_vis.colors import COLORSCALES, colorscale_stops


def encode_float32(a: np.ndarray) -> str:
    """
    Encode an array as base64 little-endian ``float32`` bytes.

    Parameters
    ----------
    a : array_like
        Any numeric array; it is flattened in C order.

    Returns
    -------
    str
        Base64 text.
    """
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def encode_volume(grid: np.ndarray, e_max: float) -> str:
    """
    Quantise a 3D image to ``uint8`` relative to `e_max` and encode it as base64.

    Parameters
    ----------
    grid : numpy.ndarray, shape (nx, ny, nz)
        Image indexed by x, y, z voxel index.
    e_max : float
        Value mapped to 255; larger values are clipped.

    Returns
    -------
    str
        Base64 text of the bytes in x-fastest order.
    """
    q = np.clip(np.rint(np.asarray(grid, dtype=float) / e_max * 255), 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes(order="F")).decode("ascii")


@dataclass
class VolumeStyle:
    """
    Initial volume rendering controls, shared by all volumes in a scene.

    Parameters
    ----------
    threshold : float
        Fraction of each volume's `e_max` below which voxels are transparent.
    density : float
        Optical depth of one voxel pitch at `e_max`; higher is more opaque.
    gamma : float
        Exponent of the opacity curve on ``value / e_max``; above 1 fades
        faint voxels.
    smooth : bool
        Trilinear interpolation instead of sharp voxel cubes.
    """

    threshold: float = 0.05
    density: float = 2.0
    gamma: float = 2.0
    smooth: bool = False

    def to_json(self) -> dict:
        """Return the JSON form of the style."""
        return {
            "threshold": float(self.threshold),
            "density": float(self.density),
            "gamma": float(self.gamma),
            "smooth": bool(self.smooth),
        }


@dataclass
class VolumeLayer:
    """
    A voxel image on a regular grid, ray-marched on the GPU.

    Parameters
    ----------
    name : str
        Layer name in the control panel and hover text.
    grid : numpy.ndarray, shape (nx, ny, nz)
        Image values indexed by x, y, z voxel index.
    origin : numpy.ndarray, shape (3,)
        Min corner (mm) of voxel ``[0, 0, 0]``.
    voxel_size : float
        Voxel pitch (mm).
    e_max : float
        Top of the colour and opacity scale.
    colorscale : list of list
        Colour scale stops.
    colorbar_title : str
        Title of the colour bar.
    visible : bool
        Whether the layer starts visible.
    """

    name: str
    grid: np.ndarray
    origin: np.ndarray
    voxel_size: float
    e_max: float
    colorscale: list
    colorbar_title: str
    visible: bool = True

    def to_json(self) -> dict:
        """Return the JSON form of the layer, with the image encoded."""
        return {
            "name": self.name,
            "shape": [int(n) for n in self.grid.shape],
            "origin": [float(o) for o in self.origin],
            "voxel_size": float(self.voxel_size),
            "data": encode_volume(self.grid, self.e_max),
            "e_max": float(self.e_max),
            "colorscale": self.colorscale,
            "colorbar_title": self.colorbar_title,
            "visible": bool(self.visible),
        }


@dataclass
class ScaleLink:
    """
    Ties a line or marker layer's colours to a colour bar.

    Parameters
    ----------
    colorbar : str
        Title of the colour bar (in `Scene.colorbars`) whose scale coloured
        the layer.
    u : numpy.ndarray, shape (N,)
        Position in [0, 1] on that scale of each colour in the layer (each
        segment end point, or each marker).
    """

    colorbar: str
    u: np.ndarray

    def to_json(self) -> dict:
        """Return the JSON form, with `u` encoded."""
        return {"colorbar": self.colorbar, "u": encode_float32(self.u)}


@dataclass
class LineLayer:
    """
    Line segments drawn with a fixed pixel width.

    Polylines are stored as independent segments so that many tracks fit in
    one layer.

    Parameters
    ----------
    name : str
        Layer name in the control panel.
    segments : numpy.ndarray, shape (2 * n_segments, 3)
        Segment end points (mm), two rows per segment.
    colors : numpy.ndarray, shape (2 * n_segments, 3)
        RGB colour in [0, 1] at each end point.
    width : float
        Line width in pixels.
    dashed : bool
        Whether to draw dashes.
    dash : float
        Dash length (mm); gaps are 0.6 of this.
    hover_points : numpy.ndarray, shape (N, 3), optional
        Positions (mm) that show `hover_text` when the cursor is near them.
    hover_text : list of str, optional
        One string per hover point.
    visible : bool
        Whether the layer starts visible.
    scale : ScaleLink, optional
        For lines coloured from a colour bar's scale: their positions on it,
        so the page can recolour them when the scale is changed there.
    """

    name: str
    segments: np.ndarray
    colors: np.ndarray
    width: float = 3.0
    dashed: bool = False
    dash: float = 20.0
    hover_points: np.ndarray | None = None
    hover_text: list[str] | None = None
    visible: bool = True
    scale: ScaleLink | None = None

    def to_json(self) -> dict:
        """Return the JSON form of the layer, with arrays encoded."""
        hover = None
        if self.hover_points is not None:
            hover = {"points": encode_float32(self.hover_points), "text": list(self.hover_text)}
        return {
            "name": self.name,
            "segments": encode_float32(self.segments),
            "colors": encode_float32(self.colors),
            "width": float(self.width),
            "dashed": bool(self.dashed),
            "dash": float(self.dash),
            "visible": bool(self.visible),
            "hover": hover,
            "scale": None if self.scale is None else self.scale.to_json(),
        }


@dataclass
class PointLayer:
    """
    Round markers with a fixed pixel size.

    Parameters
    ----------
    name : str
        Layer name in the control panel.
    positions : numpy.ndarray, shape (N, 3)
        Marker positions (mm).
    colors : numpy.ndarray, shape (N, 3)
        RGB colour in [0, 1] of each marker.
    size : float
        Marker diameter in pixels.
    hover_text : list of str
        One hover string per marker.
    visible : bool
        Whether the layer starts visible.
    scale : ScaleLink, optional
        For markers coloured from a colour bar's scale (see `LineLayer`).
    """

    name: str
    positions: np.ndarray
    colors: np.ndarray
    size: float = 12.0
    hover_text: list[str] = field(default_factory=list)
    visible: bool = True
    scale: ScaleLink | None = None

    def to_json(self) -> dict:
        """Return the JSON form of the layer, with arrays encoded."""
        return {
            "name": self.name,
            "positions": encode_float32(self.positions),
            "colors": encode_float32(self.colors),
            "size": float(self.size),
            "hover": list(self.hover_text),
            "visible": bool(self.visible),
            "scale": None if self.scale is None else self.scale.to_json(),
        }


@dataclass
class Colorbar:
    """
    A colour bar that is not tied to a volume (e.g. the shared time scale).

    Parameters
    ----------
    title : str
        Text above the bar.
    colorscale : list of list
        Colour scale stops.
    vmin, vmax : float
        Values at the bottom and top of the bar.
    """

    title: str
    colorscale: list
    vmin: float
    vmax: float

    def to_json(self) -> dict:
        """Return the JSON form of the colour bar."""
        return {"title": self.title, "stops": self.colorscale, "min": float(self.vmin), "max": float(self.vmax)}


@dataclass
class Scene:
    """
    Everything the browser needs to draw one event display.

    Volumes get their own colour bars; `colorbars` holds the others.

    Parameters
    ----------
    title : str or None
        Text in the top-left corner of the view.
    width : int
        Maximum view width in CSS pixels.
    height : int
        View height in CSS pixels.
    style : VolumeStyle
        Initial volume rendering controls.
    volumes, lines, points : list
        The layers, drawn and listed in the control panel in this order.
    colorbars : list of Colorbar
        Extra colour bars.
    bounds : numpy.ndarray, shape (2, 3), optional
        Fixed view box ``[[x_min, y_min, z_min], [x_max, y_max, z_max]]``
        (mm) for the axes and camera. None fits the box to all layers.
    theme : {"light", "dark"}
        Starting colours of the background, axes and controls.

    The JSON also carries ``colorscales``, the built-in scales (name to
    stops) offered by the colour-bar menu in the page.
    """

    title: str | None = None
    width: int = 900
    height: int = 600
    style: VolumeStyle = field(default_factory=VolumeStyle)
    volumes: list[VolumeLayer] = field(default_factory=list)
    lines: list[LineLayer] = field(default_factory=list)
    points: list[PointLayer] = field(default_factory=list)
    colorbars: list[Colorbar] = field(default_factory=list)
    bounds: np.ndarray | None = None
    theme: str = "light"

    def to_json(self) -> dict:
        """
        Return the JSON object read by ``js/event_display.js``.

        Returns
        -------
        dict
            JSON-serialisable scene with all arrays encoded.
        """
        return {
            "title": self.title,
            "width": int(self.width),
            "height": int(self.height),
            "volume_style": self.style.to_json(),
            "volumes": [v.to_json() for v in self.volumes],
            "lines": [ln.to_json() for ln in self.lines],
            "points": [p.to_json() for p in self.points],
            "colorbars": [c.to_json() for c in self.colorbars],
            "bounds": None if self.bounds is None else np.asarray(self.bounds, dtype=float).tolist(),
            "theme": self.theme,
            "colorscales": {name: colorscale_stops(name) for name in COLORSCALES},
        }
