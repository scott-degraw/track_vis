"""Interactive 3D event displays rendered with three.js in the browser.

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


@cache
def _static(name: str) -> str:
    return resources.files("track_vis").joinpath("static", name).read_text(encoding="utf-8")


def _b64(a: np.ndarray, dtype=np.float32) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _np(a) -> np.ndarray:
    return np.asarray(ak.to_numpy(a) if isinstance(a, ak.Array) else a, dtype=float)


def voxelize(centers: np.ndarray, values: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Scatter per-voxel values at regular-grid *centers* into a dense array.

    Returns ``(grid, origin)``: ``grid`` has shape ``(nx, ny, nz)`` and
    ``origin`` is the min corner (mm) of voxel ``[0, 0, 0]``. Values of
    repeated centres are summed.
    """
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    lo = centers.min(axis=0)
    idx = np.rint((centers - lo) / voxel_size).astype(np.int64)
    grid = np.zeros(idx.max(axis=0) + 1, dtype=np.float32)
    np.add.at(grid, tuple(idx.T), np.asarray(values, dtype=np.float32))
    return grid, lo - voxel_size / 2


def bin_deposits(points: np.ndarray, energies: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Sum point energy deposits into cubes aligned to multiples of *voxel_size*.

    Returns ``(grid, origin)`` like ``voxelize``.
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
    """A 3D scene of voxel volumes, polylines and markers, rendered with three.js.

    Every ``add_*`` method returns ``self`` so calls can be chained::

        disp = EventDisplay(title="Event 3")
        disp.add_volume(centers=c, energies=e, voxel_size=100)
        disp.add_tracks(**tracks, t_min=t_min, t_max=t_max)
        disp.add_vertices(**vertices, t_min=t_min, t_max=t_max)
        disp  # renders in Jupyter / Quarto

    Parameters
    ----------
    title:
        Shown at the top of the control panel.
    width, height:
        Maximum width and fixed height of the view, in CSS pixels.
    threshold, density, gamma, smooth:
        Initial volume rendering style, shared by all volumes and adjustable
        in the page. ``threshold`` is the fraction of each volume's ``e_max``
        below which voxels are transparent; ``density`` the optical depth of
        one voxel pitch at ``e_max``; ``gamma`` the exponent of the opacity
        curve (>1 fades faint voxels); ``smooth`` trilinear interpolation
        instead of sharp voxel cubes.
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
        """Add a voxel image, ray-marched on the GPU.

        Pass either a dense ``grid`` (shape ``(nx, ny, nz)``, indexed x, y, z)
        with its min-corner ``origin`` and ``voxel_size``, or sparse
        ``centers`` / ``energies`` on a regular grid plus ``voxel_size``.

        ``e_max`` is the top of the colour and opacity scale (default: the
        image maximum); pass a common value to compare events. ``pad`` adds
        empty voxels on every side so the image does not touch its box.
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
        """Bin per-step deposited energy (jagged, track x step) into a voxel volume.

        Cubes are aligned to multiples of ``voxel_size``. Extra keyword
        arguments go to ``add_volume``.
        """
        pts = np.stack([_np(ak.flatten(a, axis=None)) for a in (pos_x, pos_y, pos_z)], axis=-1)
        dep = _np(ak.flatten(deposited_energy, axis=None))
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
        """Add one polyline, or several as a list of ``(N, 3)`` arrays in one layer.

        Colour is one hex ``color``, or sampled from ``colorscale`` over
        ``[vmin, vmax]`` when per-point ``values`` (same nesting as
        ``points``) are given. ``width`` is in pixels, ``dash`` in mm.
        ``hover_text`` gives one string per point (flattened over polylines).
        """
        single = _is_single_path(points)
        paths = [np.asarray(points, dtype=float).reshape(-1, 3)] if single else [np.asarray(p, float) for p in points]
        if values is not None:
            vals = [np.asarray(values, float)] if single else [np.asarray(v, float) for v in values]
            flat = np.concatenate(vals) if vals else np.zeros(0)
            lo = float(flat.min()) if vmin is None else vmin
            hi = float(flat.max()) if vmax is None else vmax
            cols = [sample_colorscale(colorscale, (v - lo) / ((hi - lo) or 1.0)) for v in vals]
        else:
            rgb = hex_to_rgb(color)
            cols = [np.tile(rgb, (len(p), 1)) for p in paths]
        seg, col = _polyline_segments(paths, cols)
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
        """Add MC truth tracks coloured by step time, one toggleable layer per species.

        Columns mirror the ntuple: flat ``pdg`` (one per track) and jagged
        ``pos_x/y/z``, ``time``, ``deposited_energy`` (track x step). Use the
        same ``t_min``/``t_max`` for ``add_vertices`` so both share the time
        colour bar (default: the range of the step times).
        """
        pdgs = _np(pdg).astype(int)
        cols = [
            ak.Array(c) if not isinstance(c, ak.Array) else c for c in (pos_x, pos_y, pos_z, time, deposited_energy)
        ]
        if t_min is None or t_max is None:
            all_t = _np(ak.flatten(cols[3], axis=None))
            t_min = float(all_t.min()) if t_min is None else t_min
            t_max = float(all_t.max()) if t_max is None else t_max
        by_species: dict[str, list[int]] = {}
        for i, p in enumerate(pdgs):
            by_species.setdefault(particle_name(int(p)), []).append(i)
        for species, idx in by_species.items():
            paths, times, texts = [], [], []
            dep_total = 0.0
            for i in idx:
                x, y, z, t, d = (_np(c[i]) for c in cols)
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
        """Add round markers (``size`` in pixels), coloured like ``add_line``."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if values is not None:
            v = np.asarray(values, dtype=float)
            lo = float(v.min()) if vmin is None else vmin
            hi = float(v.max()) if vmax is None else vmax
            rgb = sample_colorscale(colorscale, (v - lo) / ((hi - lo) or 1.0))
        else:
            rgb = np.tile(hex_to_rgb(color), (len(pts), 1))
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
        """Add MC truth vertices coloured by time, one layer per species.

        Flat arrays, one entry per vertex (``mcpdgs``, ``mcxs``... in the ntuple).
        """
        pos = np.stack([_np(x), _np(y), _np(z)], axis=-1)
        t = _np(t)
        energy = _np(energy)
        pdgs = _np(pdg).astype(int)
        t_min = float(t.min()) if t_min is None else t_min
        t_max = float(t.max()) if t_max is None else t_max
        by_species: dict[str, list[int]] = {}
        for i, p in enumerate(pdgs):
            by_species.setdefault(particle_name(int(p)), []).append(i)
        for species, idx in by_species.items():
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
        bar = {"title": "Time (ns)", "stops": colorscale_stops(colorscale), "min": float(t_min), "max": float(t_max)}
        self.colorbars = [b for b in self.colorbars if b["title"] != "Time (ns)"] + [bar]

    # ------------------------------------------------------------------ output
    def spec(self) -> dict:
        """The JSON-serialisable scene description passed to the browser."""
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
        """Return an HTML fragment (scripts + container div) for this scene.

        ``lib="cdn"`` loads three.js from jsDelivr once per page; ``"inline"``
        embeds it, so the fragment needs no network.
        """
        div_id = "tv-" + uuid.uuid4().hex[:12]
        parts = []
        if lib == "inline":
            for fname, _ in _VENDOR_FILES:
                parts.append(f"<script>if(!window.__tvThreeInline){{{_static(f'vendor/{fname}')}\n}}</script>")
            parts.append("<script>window.__tvThreeInline=true;</script>")
            loader = "Promise.resolve()"
        elif lib == "cdn":
            loader = (
                "(window.__tvThree = window.__tvThree || "
                + json.dumps(CDN_URLS)
                + ".reduce((p, u) => p.then(() => new Promise((ok, err) => { const s = document.createElement('script'); s.src = u;"
                " s.onload = ok; s.onerror = () => err(u); document.head.appendChild(s); })), Promise.resolve()))"
            )
        else:
            raise ValueError("lib must be 'cdn' or 'inline'")
        parts.append(f"<script>{_static('event_display.js')}</script>")
        parts.append(f'<div id="{div_id}" class="track-vis" style="min-height:{self.height}px"></div>')
        # Initialise when scrolled near view: browsers cap live WebGL contexts at ~16 per page.
        parts.append(
            "<script>(function(){"
            f"const el = document.getElementById({json.dumps(div_id)});"
            f"const spec = {json.dumps(self.spec(), separators=(',', ':'))};"
            "let done = false;"
            "const go = () => { if (done) return; done = true;"
            f"{loader}.then(() => TrackVis.render(el, spec))"
            ".catch(e => { el.textContent = 'track_vis: could not load three.js (' + e + ')'; }); };"
            "if ('IntersectionObserver' in window) {"
            "new IntersectionObserver((es, ob) => { if (es.some(e => e.isIntersecting)) { ob.disconnect(); go(); } },"
            "{ rootMargin: '300px' }).observe(el);"
            "} else { go(); }"
            "})();</script>"
        )
        return "\n".join(parts)

    def _repr_html_(self) -> str:
        return self.to_html()

    def write_html(self, path: str, lib: str = "inline", page_title: str | None = None) -> None:
        """Write a standalone page (three.js inlined by default, so it works offline)."""
        title = html.escape(page_title or self.title or "track_vis event display")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
                '<meta name="viewport" content="width=device-width, initial-scale=1"></head>'
                f'<body style="margin:16px;background:#fff">{self.to_html(lib=lib)}</body></html>'
            )
