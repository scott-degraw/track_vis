"""
display.py

Interactive 3D event displays rendered with three.js in the browser.

``EventDisplay`` collects layers (voxel volumes, polylines, markers) and turns
them into one self-contained HTML fragment. Voxel images are uploaded to the
GPU as 3D textures and ray-marched per pixel, so every voxel has its own
opacity, there is no triangle sorting, and an image costs one byte per voxel.

In Jupyter an ``EventDisplay`` renders itself when it is the last expression
of a cell (``_repr_html_``); the same output works in a rendered Quarto page.
``to_html`` returns the fragment and ``write_html`` writes a standalone page.

three.js r147 is loaded from a pinned jsDelivr URL (``lib="cdn"``, default)
or embedded from the copy vendored in ``track_vis/static/vendor``
(``lib="inline"``, ~660 kB, no network needed).

Units: positions in mm, times in ns, energies in MeV.
"""

from __future__ import annotations

import base64
import html
import json
import uuid
from functools import cache
from importlib import resources

import awkward as ak
import numpy as np

from track_vis.colors import colorscale_stops, hex_to_rgb, sample_colorscale
from track_vis.io import particle_name

THREE_VERSION = "0.147.0"
_VENDOR_FILES = [
    ("three.min.js", "build/three.min.js"),
    ("OrbitControls.js", "examples/js/controls/OrbitControls.js"),
    ("LineSegmentsGeometry.js", "examples/js/lines/LineSegmentsGeometry.js"),
    ("LineMaterial.js", "examples/js/lines/LineMaterial.js"),
    ("LineSegments2.js", "examples/js/lines/LineSegments2.js"),
]
CDN_URLS = [f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/{path}" for _, path in _VENDOR_FILES]

TIME_COLORSCALE = "plasma"
VOXEL_COLORSCALE = "plasma"
TIME_COLORBAR_TITLE = "Time (ns)"


@cache
def _static(name: str) -> str:
    return resources.files("track_vis").joinpath("static", name).read_text(encoding="utf-8")


def _inline_script(name: str, once_key: str | None = None) -> str:
    """A <script> holding a static file; with *once_key* it runs once per page."""
    body = _static(name)
    if once_key is not None:
        flag = f"window[{json.dumps('__trackvis_' + once_key)}]"
        body = f"if (!{flag}) {{ {flag} = true;\n{body}\n}}"
    return f"<script>{body}</script>"


def _script_json(value) -> str:
    """JSON safe to embed in an HTML <script> (no "</" that could close the tag)."""
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def _b64(a: np.ndarray, dtype=np.float32) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _to_numpy(a) -> np.ndarray:
    return np.asarray(ak.to_numpy(a) if isinstance(a, ak.Array) else a, dtype=float)


def _group_by_species(pdg) -> dict[str, list[int]]:
    """Map particle name to the indices of entries with that PDG code, in first-seen order."""
    groups: dict[str, list[int]] = {}
    for i, code in enumerate(_to_numpy(pdg).astype(int)):
        groups.setdefault(particle_name(int(code)), []).append(i)
    return groups


def _colors(values, vmin, vmax, colorscale, color: str, n: int) -> np.ndarray:
    """(n, 3) RGB: *values* sampled on *colorscale* over [vmin, vmax], else *color* for all."""
    if values is None:
        return np.tile(hex_to_rgb(color), (n, 1))
    values = np.asarray(values, dtype=float)
    lo = float(values.min()) if vmin is None else vmin
    hi = float(values.max()) if vmax is None else vmax
    return sample_colorscale(colorscale, (values - lo) / ((hi - lo) or 1.0))


def voxelize(centers: np.ndarray, values: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Scatter per-voxel values at regular-grid centres into a dense array.

    Values of repeated centres are summed.

    Parameters
    ----------
    centers : array_like, shape (N, 3)
        Voxel centre positions (mm), all on one grid of pitch `voxel_size`.
    values : array_like, shape (N,)
        Value of each voxel, e.g. reconstructed energy (MeV).
    voxel_size : float
        Grid pitch (mm).

    Returns
    -------
    grid : numpy.ndarray, shape (nx, ny, nz)
        Dense ``float32`` array indexed by x, y, z voxel index.
    origin : numpy.ndarray, shape (3,)
        Min corner (mm) of voxel ``[0, 0, 0]``.
    """
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    lo = centers.min(axis=0)
    idx = np.rint((centers - lo) / voxel_size).astype(np.int64)
    grid = np.zeros(idx.max(axis=0) + 1, dtype=np.float32)
    np.add.at(grid, tuple(idx.T), np.asarray(values, dtype=np.float32))
    return grid, lo - voxel_size / 2


def bin_deposits(points: np.ndarray, energies: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum point energy deposits into cubes aligned to multiples of `voxel_size`.

    Deposits with zero or negative energy are ignored.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Deposit positions (mm).
    energies : array_like, shape (N,)
        Deposited energy at each point (MeV).
    voxel_size : float
        Cube edge length (mm).

    Returns
    -------
    grid : numpy.ndarray, shape (nx, ny, nz)
        Summed energy per cube, as ``float32``.
    origin : numpy.ndarray, shape (3,)
        Min corner (mm) of cube ``[0, 0, 0]``.

    Raises
    ------
    ValueError
        If no deposit has positive energy.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    energies = np.asarray(energies, dtype=float)
    keep = energies > 0
    if not keep.any():
        raise ValueError("no positive deposits to bin")
    keys = np.floor(points[keep] / voxel_size).astype(np.int64)
    k0 = keys.min(axis=0)
    idx = keys - k0
    grid = np.zeros(idx.max(axis=0) + 1, dtype=np.float32)
    np.add.at(grid, tuple(idx.T), energies[keep].astype(np.float32))
    return grid, k0 * float(voxel_size)


def _is_single_path(points) -> bool:
    """True for one (N, 3) polyline, False for a list of them."""
    if isinstance(points, np.ndarray):
        return points.ndim == 2
    first = next(iter(points), None)
    return first is None or np.ndim(first) == 1


def _polyline_segments(paths: list[np.ndarray], colors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Turn polylines (each (N, 3)) with per-point RGB into segment endpoint arrays."""
    seg, col = [], []
    for p, c in zip(paths, colors):
        if len(p) < 2:
            continue
        seg.append(np.stack([p[:-1], p[1:]], axis=1).reshape(-1, 3))
        col.append(np.stack([c[:-1], c[1:]], axis=1).reshape(-1, 3))
    if not seg:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(seg), np.concatenate(col)


class EventDisplay:
    """
    Build a 3D scene of voxel volumes, polylines and markers for three.js.

    Every ``add_*`` method returns the display itself, so calls chain::

        disp = EventDisplay(title="Event 3")
        disp.add_volume(centers=c, energies=e, voxel_size=100)
        disp.add_tracks(**tracks, t_min=t_min, t_max=t_max)
        disp.add_vertices(**vertices, t_min=t_min, t_max=t_max)
        disp  # renders in Jupyter / Quarto

    The volume style arguments set the initial values of controls in the
    page and are shared by all volumes in the scene.

    Parameters
    ----------
    title : str, optional
        Text shown at the top of the control panel. Default is None.
    width : int, optional
        Maximum width of the view in CSS pixels. Default is 900.
    height : int, optional
        Height of the view in CSS pixels. Default is 600.
    threshold : float, optional
        Fraction of each volume's `e_max` below which voxels are fully
        transparent. Default is 0.05.
    density : float, optional
        Optical depth of one voxel pitch at `e_max`; higher is more opaque.
        Default is 2.0.
    gamma : float, optional
        Exponent of the opacity curve on ``value / e_max``; values above 1
        fade faint voxels. Default is 2.0.
    smooth : bool, optional
        If True, start with trilinear interpolation instead of sharp voxel
        cubes. Default is False.

    Attributes
    ----------
    title : str or None
        Title shown in the control panel.
    width : int
        Maximum view width in CSS pixels.
    height : int
        View height in CSS pixels.
    volume_style : dict
        Initial ``threshold``, ``density``, ``gamma`` and ``smooth`` values.
    volumes : list of dict
        Volume layers, as sent to the browser.
    lines : list of dict
        Polyline layers, as sent to the browser.
    points : list of dict
        Marker layers, as sent to the browser.
    colorbars : list of dict
        Extra colour bars (the shared time bar); volume bars come from
        `volumes`.
    """

    def __init__(
        self,
        title: str | None = None,
        width: int = 900,
        height: int = 600,
        threshold: float = 0.05,
        density: float = 2.0,
        gamma: float = 2.0,
        smooth: bool = False,
    ):
        self.title = title
        self.width = int(width)
        self.height = int(height)
        self.volume_style = {
            "threshold": float(threshold),
            "density": float(density),
            "gamma": float(gamma),
            "smooth": bool(smooth),
        }
        self.volumes: list[dict] = []
        self.lines: list[dict] = []
        self.points: list[dict] = []
        self.colorbars: list[dict] = []

    # ------------------------------------------------------------------ volumes
    def add_volume(
        self,
        grid: np.ndarray | None = None,
        origin: np.ndarray | None = None,
        voxel_size: float | None = None,
        *,
        centers: np.ndarray | None = None,
        energies: np.ndarray | None = None,
        e_max: float | None = None,
        name: str = "voxels",
        colorscale: str | list = VOXEL_COLORSCALE,
        colorbar_title: str = "Energy (MeV)",
        pad: int = 1,
        visible: bool = True,
    ) -> EventDisplay:
        """
        Add a voxel image, ray-marched on the GPU.

        Pass either a dense `grid` with its `origin`, or sparse `centers`
        and `energies` on a regular grid; `voxel_size` is needed for both.
        The image is quantised to 8 bits relative to `e_max`.

        Parameters
        ----------
        grid : array_like, shape (nx, ny, nz), optional
            Dense image indexed by x, y, z voxel index.
        origin : array_like, shape (3,) or float, optional
            Min corner (mm) of voxel ``[0, 0, 0]``; a scalar applies to all
            three axes. Required with `grid`.
        voxel_size : float
            Voxel pitch (mm).
        centers : array_like, shape (N, 3), optional
            Voxel centres (mm) on a regular grid, used when `grid` is None.
        energies : array_like, shape (N,), optional
            Value of each voxel in `centers`.
        e_max : float, optional
            Top of the colour and opacity scale. Default is the image
            maximum; pass a common value to compare events.
        name : str, optional
            Layer name shown in the control panel. Default is "voxels".
        colorscale : str or list, optional
            Colour scale name or stops (see `colors.colorscale_stops`).
            Default is "plasma".
        colorbar_title : str, optional
            Title of this volume's colour bar. Default is "Energy (MeV)".
        pad : int, optional
            Empty voxels added on every side so the image does not touch
            its box. Default is 1.
        visible : bool, optional
            Whether the layer starts visible. Default is True.

        Returns
        -------
        EventDisplay
            This display, for chaining.

        Raises
        ------
        ValueError
            If `voxel_size` is missing, if neither `grid` and `origin` nor
            `centers` and `energies` are given, or if `grid` is not 3D.
        """
        if voxel_size is None:
            raise ValueError("voxel_size is required")
        if grid is None:
            if centers is None or energies is None:
                raise ValueError("pass grid and origin, or centers and energies")
            grid, origin = voxelize(centers, energies, voxel_size)
        elif origin is None:
            raise ValueError("origin is required with grid")
        grid = np.asarray(grid, dtype=np.float32)
        if grid.ndim != 3:
            raise ValueError(f"grid must be 3D, got shape {grid.shape}")
        origin = np.broadcast_to(np.asarray(origin, dtype=float), (3,)).copy()
        if pad:
            grid = np.pad(grid, pad)
            origin = origin - pad * voxel_size
        if e_max is None:
            e_max = float(grid.max())
        e_max = float(e_max) or 1.0
        q = np.clip(np.rint(grid / e_max * 255), 0, 255).astype(np.uint8)
        self.volumes.append(
            {
                "name": name,
                "shape": [int(n) for n in q.shape],
                "origin": [float(o) for o in origin],
                "voxel_size": float(voxel_size),
                # WebGL 3D textures are x-fastest: Fortran order of an (x, y, z) array.
                "data": base64.b64encode(q.tobytes(order="F")).decode("ascii"),
                "e_max": e_max,
                "colorscale": colorscale_stops(colorscale),
                "colorbar_title": colorbar_title,
                "visible": bool(visible),
            }
        )
        return self

    def add_deposits(
        self,
        pos_x: ak.Array,
        pos_y: ak.Array,
        pos_z: ak.Array,
        deposited_energy: ak.Array,
        voxel_size: float = 100.0,
        name: str | None = None,
        colorbar_title: str = "Voxel dep. energy (MeV)",
        **kwargs,
    ) -> EventDisplay:
        """
        Bin per-step deposited energy into a voxel volume.

        Cubes are aligned to multiples of `voxel_size` (see `bin_deposits`).
        Inputs are jagged arrays indexed by track and step, as returned by
        `io.load_event`.

        Parameters
        ----------
        pos_x, pos_y, pos_z : awkward.Array
            Step positions (mm), shape (track, step).
        deposited_energy : awkward.Array
            Energy deposited at each step (MeV), shape (track, step).
        voxel_size : float, optional
            Cube edge length (mm). Default is 100.0.
        name : str, optional
            Layer name. Default is "deposits (<voxel_size> mm)".
        colorbar_title : str, optional
            Title of the colour bar. Default is "Voxel dep. energy (MeV)".
        **kwargs
            Passed on to `add_volume` (e.g. `e_max`, `colorscale`, `pad`).

        Returns
        -------
        EventDisplay
            This display, for chaining.

        Raises
        ------
        ValueError
            If no step has positive deposited energy.
        """
        pts = np.stack([_to_numpy(ak.flatten(a, axis=None)) for a in (pos_x, pos_y, pos_z)], axis=-1)
        dep = _to_numpy(ak.flatten(deposited_energy, axis=None))
        grid, origin = bin_deposits(pts, dep, voxel_size)
        return self.add_volume(
            grid,
            origin,
            voxel_size,
            name=name or f"deposits ({voxel_size:g} mm)",
            colorbar_title=colorbar_title,
            **kwargs,
        )

    # ------------------------------------------------------------------ lines
    def add_line(
        self,
        points: np.ndarray,
        name: str = "line",
        *,
        values: np.ndarray | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        colorscale: str | list = TIME_COLORSCALE,
        color: str = "#1f77b4",
        width: float = 3.0,
        dashed: bool = False,
        dash: float = 20.0,
        hover_text: list[str] | None = None,
        visible: bool = True,
    ) -> EventDisplay:
        """
        Add one polyline, or several polylines as one layer.

        Lines are drawn in one hex `color`, or coloured per point by
        sampling `colorscale` over ``[vmin, vmax]`` when `values` is given.

        Parameters
        ----------
        points : array_like, shape (N, 3), or list of such arrays
            Vertices of one polyline, or a list of polylines (mm).
        name : str, optional
            Layer name shown in the control panel. Default is "line".
        values : array_like, optional
            Per-point values with the same nesting as `points`.
        vmin, vmax : float, optional
            Colour scale range for `values`. Default is their min and max.
        colorscale : str or list, optional
            Colour scale used with `values`. Default is "plasma".
        color : str, optional
            Hex colour used when `values` is None. Default is "#1f77b4".
        width : float, optional
            Line width in pixels. Default is 3.0.
        dashed : bool, optional
            If True, draw a dashed line. Default is False.
        dash : float, optional
            Dash length (mm). Default is 20.0.
        hover_text : list of str, optional
            One hover string per point, flattened over all polylines.
        visible : bool, optional
            Whether the layer starts visible. Default is True.

        Returns
        -------
        EventDisplay
            This display, for chaining.

        Raises
        ------
        ValueError
            If `hover_text` does not have one entry per point.
        """
        if _is_single_path(points):
            points = [points]
            values = None if values is None else [values]
        paths = [np.asarray(p, dtype=float).reshape(-1, 3) for p in points]
        lengths = [len(p) for p in paths]
        flat_values = None if values is None else np.concatenate([np.asarray(v, float) for v in values])
        rgb = _colors(flat_values, vmin, vmax, colorscale, color, sum(lengths))
        seg, col = _polyline_segments(paths, np.split(rgb, np.cumsum(lengths)[:-1]))
        layer = {
            "name": name,
            "segments": _b64(seg),
            "colors": _b64(col),
            "width": float(width),
            "dashed": bool(dashed),
            "dash": float(dash),
            "visible": bool(visible),
            "hover": None,
        }
        if hover_text is not None:
            pts = np.concatenate(paths) if paths else np.zeros((0, 3))
            if len(hover_text) != len(pts):
                raise ValueError("hover_text needs one entry per point")
            layer["hover"] = {"points": _b64(pts), "text": list(map(str, hover_text))}
        self.lines.append(layer)
        return self

    def add_tracks(
        self,
        pdg: np.ndarray,
        pos_x: ak.Array,
        pos_y: ak.Array,
        pos_z: ak.Array,
        time: ak.Array,
        deposited_energy: ak.Array,
        t_min: float | None = None,
        t_max: float | None = None,
        colorscale: str | list = TIME_COLORSCALE,
        width: float = 3.0,
    ) -> EventDisplay:
        """
        Add MC truth tracks coloured by step time, one layer per species.

        Columns mirror the ``imaging_reco`` ntuple branches. Hovering over a
        step shows its time and deposited energy. Use the same `t_min` and
        `t_max` for `add_vertices` so both share the time colour bar.

        Parameters
        ----------
        pdg : array_like, shape (n_tracks,)
            PDG code of each track.
        pos_x, pos_y, pos_z : awkward.Array or list of lists
            Step positions (mm), shape (track, step).
        time : awkward.Array or list of lists
            Step times (ns), shape (track, step).
        deposited_energy : awkward.Array or list of lists
            Energy deposited at each step (MeV), shape (track, step).
        t_min, t_max : float, optional
            Time colour scale range. Default is the range of step times.
        colorscale : str or list, optional
            Time colour scale. Default is "plasma".
        width : float, optional
            Line width in pixels. Default is 3.0.

        Returns
        -------
        EventDisplay
            This display, for chaining.
        """
        columns = [ak.Array(c) for c in (pos_x, pos_y, pos_z, time, deposited_energy)]
        if t_min is None or t_max is None:
            all_t = _to_numpy(ak.flatten(columns[3], axis=None))
            t_min = float(all_t.min()) if t_min is None else t_min
            t_max = float(all_t.max()) if t_max is None else t_max
        for species, idx in _group_by_species(pdg).items():
            paths, times, texts = [], [], []
            dep_total = 0.0
            for i in idx:
                x, y, z, t, d = (_to_numpy(c[i]) for c in columns)
                paths.append(np.stack([x, y, z], axis=-1))
                times.append(t)
                dep_total += float(d.sum())
                texts += [f"t = {tt:.2f} ns, dep = {dd:.3f} MeV" for tt, dd in zip(t, d)]
            n = len(idx)
            label = f"{species} ({n} track{'s' if n > 1 else ''}, {dep_total:.2f} MeV dep.)"
            self.add_line(
                paths,
                label,
                values=times,
                vmin=t_min,
                vmax=t_max,
                colorscale=colorscale,
                width=width,
                hover_text=texts,
            )
        self._time_colorbar(t_min, t_max, colorscale)
        return self

    # ------------------------------------------------------------------ markers
    def add_points(
        self,
        points: np.ndarray,
        name: str = "points",
        *,
        values: np.ndarray | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        colorscale: str | list = TIME_COLORSCALE,
        color: str = "#1f77b4",
        size: float = 12.0,
        hover_text: list[str] | None = None,
        visible: bool = True,
    ) -> EventDisplay:
        """
        Add round markers as one layer.

        Markers are drawn in one hex `color`, or coloured per point from
        `colorscale` over ``[vmin, vmax]`` when `values` is given.

        Parameters
        ----------
        points : array_like, shape (N, 3)
            Marker positions (mm).
        name : str, optional
            Layer name shown in the control panel. Default is "points".
        values : array_like, shape (N,), optional
            Per-marker values to colour by.
        vmin, vmax : float, optional
            Colour scale range for `values`. Default is their min and max.
        colorscale : str or list, optional
            Colour scale used with `values`. Default is "plasma".
        color : str, optional
            Hex colour used when `values` is None. Default is "#1f77b4".
        size : float, optional
            Marker diameter in pixels. Default is 12.0.
        hover_text : list of str, optional
            One hover string per marker.
        visible : bool, optional
            Whether the layer starts visible. Default is True.

        Returns
        -------
        EventDisplay
            This display, for chaining.

        Raises
        ------
        ValueError
            If `hover_text` does not have one entry per marker.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        rgb = _colors(values, vmin, vmax, colorscale, color, len(pts))
        text = list(map(str, hover_text)) if hover_text is not None else [""] * len(pts)
        if len(text) != len(pts):
            raise ValueError("hover_text needs one entry per point")
        self.points.append(
            {
                "name": name,
                "positions": _b64(pts),
                "colors": _b64(rgb),
                "size": float(size),
                "hover": text,
                "visible": bool(visible),
            }
        )
        return self

    def add_vertices(
        self,
        pdg: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: np.ndarray,
        energy: np.ndarray,
        t_min: float | None = None,
        t_max: float | None = None,
        colorscale: str | list = TIME_COLORSCALE,
        size: float = 12.0,
    ) -> EventDisplay:
        """
        Add MC truth vertices coloured by time, one layer per species.

        Arrays are flat, one entry per vertex, mirroring the ntuple's
        ``mcpdgs``, ``mcxs``, ``mcys``, ``mczs``, ``mcts`` and ``mckes``.
        Hovering over a vertex shows its energy and time.

        Parameters
        ----------
        pdg : array_like, shape (N,)
            PDG code of each vertex particle.
        x, y, z : array_like, shape (N,)
            Vertex positions (mm).
        t : array_like, shape (N,)
            Vertex times (ns).
        energy : array_like, shape (N,)
            Kinetic energy of each particle (MeV).
        t_min, t_max : float, optional
            Time colour scale range. Default is the range of `t`.
        colorscale : str or list, optional
            Time colour scale. Default is "plasma".
        size : float, optional
            Marker diameter in pixels. Default is 12.0.

        Returns
        -------
        EventDisplay
            This display, for chaining.
        """
        pos = np.stack([_to_numpy(x), _to_numpy(y), _to_numpy(z)], axis=-1)
        t = _to_numpy(t)
        energy = _to_numpy(energy)
        t_min = float(t.min()) if t_min is None else t_min
        t_max = float(t.max()) if t_max is None else t_max
        for species, idx in _group_by_species(pdg).items():
            idx = np.array(idx)
            self.add_points(
                pos[idx],
                f"vertex: {species}",
                values=t[idx],
                vmin=t_min,
                vmax=t_max,
                colorscale=colorscale,
                size=size,
                hover_text=[f"E = {energy[i]:.3f} MeV, t = {t[i]:.2f} ns" for i in idx],
            )
        self._time_colorbar(t_min, t_max, colorscale)
        return self

    def _time_colorbar(self, t_min: float, t_max: float, colorscale) -> None:
        """Set the shared time colour bar, replacing any earlier one."""
        bar = {
            "title": TIME_COLORBAR_TITLE,
            "stops": colorscale_stops(colorscale),
            "min": float(t_min),
            "max": float(t_max),
        }
        self.colorbars = [b for b in self.colorbars if b["title"] != TIME_COLORBAR_TITLE] + [bar]

    # ------------------------------------------------------------------ output
    def spec(self) -> dict:
        """
        Return the JSON-serialisable scene description passed to the browser.

        Returns
        -------
        dict
            Title, size, volume style, and the volume, line, marker and
            colour bar layers.
        """
        return {
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "volume_style": self.volume_style,
            "volumes": self.volumes,
            "lines": self.lines,
            "points": self.points,
            "colorbars": self.colorbars,
        }

    def to_html(self, lib: str = "cdn") -> str:
        """
        Return an HTML fragment (scripts and a container div) for this scene.

        The display initialises when it scrolls near the viewport, since
        browsers cap live WebGL contexts at about 16 per page.

        Parameters
        ----------
        lib : {"cdn", "inline"}, optional
            "cdn" loads three.js from pinned jsDelivr URLs, once per page;
            "inline" embeds the vendored copy so no network is needed.
            Default is "cdn".

        Returns
        -------
        str
            The HTML fragment.

        Raises
        ------
        ValueError
            If `lib` is not "cdn" or "inline".
        """
        if lib == "inline":
            # Each vendored file runs only once per page, however many displays embed it.
            parts = [_inline_script(f"vendor/{fname}", once_key=fname) for fname, _ in _VENDOR_FILES]
            urls = None
        elif lib == "cdn":
            parts = []
            urls = CDN_URLS
        else:
            raise ValueError("lib must be 'cdn' or 'inline'")
        div_id = "track-vis-" + uuid.uuid4().hex[:12]
        args = ", ".join(_script_json(a) for a in (div_id, self.spec(), urls))
        parts += [
            _inline_script("event_display.js"),
            f'<div id="{div_id}" class="track-vis" style="min-height:{self.height}px"></div>',
            f"<script>TrackVis.mount({args});</script>",
        ]
        return "\n".join(parts)

    def _repr_html_(self) -> str:
        """Render in Jupyter and Quarto (three.js from the CDN)."""
        return self.to_html()

    def write_html(self, path: str, lib: str = "inline", page_title: str | None = None) -> None:
        """
        Write the display to a standalone HTML page.

        Parameters
        ----------
        path : str
            Output file path.
        lib : {"inline", "cdn"}, optional
            How three.js is included (see `to_html`). Default is "inline",
            so the page works offline.
        page_title : str, optional
            Browser tab title. Default is the display title.

        Raises
        ------
        ValueError
            If `lib` is not "cdn" or "inline".
        """
        title = html.escape(page_title or self.title or "track_vis event display")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
                '<meta name="viewport" content="width=device-width, initial-scale=1"></head>'
                f'<body style="margin:16px;background:#fff">{self.to_html(lib=lib)}</body></html>'
            )
