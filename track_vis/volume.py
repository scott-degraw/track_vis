"""Ray-marched voxel image viewer (three.js) for static HTML pages.

Unlike the Plotly ``voxel_cubes`` traces, the image is uploaded once as a 3D
texture and composited front to back per pixel on the GPU, so every voxel gets
its own opacity, there is no triangle sorting (no popping or z-fighting while
rotating), and page size scales with the number of voxels (one byte each)
instead of with mesh vertices.

``volume_html`` returns an HTML fragment that works inside Jupyter (via
``IPython.display.HTML``) and in a rendered Quarto page. three.js r147 is
loaded either from a pinned jsDelivr URL (``lib="cdn"``, the default; ~200 kB
transferred, cached across figures) or inlined from the copy vendored in
``track_vis/static/vendor`` (``lib="inline"``, ~660 kB per page, no network).
"""

from __future__ import annotations

import base64
import html
import json
import uuid
from functools import cache
from importlib import resources

import numpy as np
import plotly.colors as pc

THREE_VERSION = "0.147.0"
_VENDOR_FILES = [
    ("three.min.js", "build/three.min.js"),
    ("OrbitControls.js", "examples/js/controls/OrbitControls.js"),
    ("LineSegmentsGeometry.js", "examples/js/lines/LineSegmentsGeometry.js"),
    ("LineGeometry.js", "examples/js/lines/LineGeometry.js"),
    ("LineMaterial.js", "examples/js/lines/LineMaterial.js"),
    ("LineSegments2.js", "examples/js/lines/LineSegments2.js"),
    ("Line2.js", "examples/js/lines/Line2.js"),
]
CDN_URLS = [f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/{path}" for _, path in _VENDOR_FILES]


@cache
def _static(name: str) -> str:
    return resources.files("track_vis").joinpath("static", name).read_text(encoding="utf-8")


def _colorscale_stops(colorscale) -> list:
    """Return ``[[pos, '#rrggbb'], ...]`` for a Plotly colorscale name or list."""
    scale = pc.get_colorscale(colorscale) if isinstance(colorscale, str) else colorscale
    out = []
    for pos, col in scale:
        if col.startswith("rgb"):
            r, g, b = (int(float(c)) for c in col[col.index("(") + 1 : col.index(")")].split(",")[:3])
            col = f"#{r:02x}{g:02x}{b:02x}"
        out.append([float(pos), col])
    return out


def voxelize(centers: np.ndarray, values: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Scatter per-voxel values at regular-grid *centers* into a dense array.

    Returns ``(grid, origin)`` where ``grid`` has shape ``(nx, ny, nz)`` and
    ``origin`` is the min corner (mm) of voxel ``[0, 0, 0]``.
    """
    centers = np.asarray(centers, dtype=float)
    lo = centers.min(axis=0)
    idx = np.rint((centers - lo) / voxel_size).astype(np.int64)
    grid = np.zeros(idx.max(axis=0) + 1, dtype=np.float32)
    np.add.at(grid, tuple(idx.T), np.asarray(values, dtype=np.float32))
    return grid, lo - voxel_size / 2


def track_spec(
    points: np.ndarray,
    name: str = "track",
    values: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorscale: str = "Plasma",
    color: str = "#1f77b4",
    width: float = 3.0,
    dashed: bool = False,
    visible: bool = True,
) -> dict:
    """Describe a polyline for ``volume_html``.

    Colour is either one CSS hex *color* or, if *values* (e.g. step times) are
    given, sampled from *colorscale* over ``[vmin, vmax]``. *width* is in pixels.
    """
    points = np.asarray(points, dtype=float)
    if values is not None:
        values = np.asarray(values, dtype=float)
        lo = values.min() if vmin is None else vmin
        hi = values.max() if vmax is None else vmax
        u = np.clip((values - lo) / ((hi - lo) or 1.0), 0, 1)
        rgb = [pc.unlabel_rgb(c) for c in pc.sample_colorscale(colorscale, u.tolist())]
        colors = np.asarray(rgb, dtype=float) / 255.0
    else:
        rgb = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255.0
        colors = np.tile(rgb, (len(points), 1))
    return {
        "name": name,
        "points": np.round(points, 2).tolist(),
        "colors": np.round(colors, 3).tolist(),
        "width": float(width),
        "dashed": bool(dashed),
        "visible": bool(visible),
    }


def volume_html(
    grid: np.ndarray | None = None,
    origin: np.ndarray | None = None,
    voxel_size: float | None = None,
    *,
    centers: np.ndarray | None = None,
    energies: np.ndarray | None = None,
    tracks: list[dict] | None = None,
    e_max: float | None = None,
    threshold: float = 0.05,
    density: float = 2.0,
    gamma: float = 2.0,
    smooth: bool = False,
    colorscale: str | list = "Plasma",
    colorbar_title: str = "Energy (MeV)",
    title: str | None = None,
    width: int = 900,
    height: int = 600,
    axis_length: float = 100.0,
    pad: int = 1,
    lib: str = "cdn",
) -> str:
    """Return a self-contained HTML fragment showing a voxel image and tracks.

    Pass either a dense ``grid`` (shape ``(nx, ny, nz)``, indexed x, y, z) with
    its min-corner ``origin`` and ``voxel_size``, or sparse ``centers`` /
    ``energies`` on a regular grid plus ``voxel_size``.

    Parameters
    ----------
    e_max:
        Top of the colour/opacity scale (default: image max). Use a common
        value to compare events.
    threshold:
        Fraction of ``e_max`` below which voxels are fully transparent.
    density:
        Optical depth of one voxel pitch at ``e_max``. Higher is more opaque.
    gamma:
        Opacity curve exponent on ``value / e_max``; >1 fades faint voxels.
    smooth:
        Trilinear interpolation instead of sharp voxel blocks (toggle in page).
    pad:
        Empty voxels added on every side so the image does not touch the box.
    lib:
        ``"cdn"`` loads three.js from jsDelivr (pinned); ``"inline"`` embeds it.

    All of threshold, density, gamma, smoothing and track visibility can also
    be changed with controls in the page.
    """
    if grid is None:
        if centers is None or energies is None or voxel_size is None:
            raise ValueError("pass grid, origin and voxel_size, or centers, energies and voxel_size")
        grid, origin = voxelize(centers, energies, voxel_size)
    grid = np.asarray(grid, dtype=np.float32)
    origin = np.asarray(origin, dtype=float)
    if pad:
        grid = np.pad(grid, pad)
        origin = origin - pad * voxel_size
    if e_max is None:
        e_max = float(grid.max()) or 1.0
    q = np.clip(np.rint(grid / e_max * 255), 0, 255).astype(np.uint8)
    # WebGL 3D textures are x-fastest: write in Fortran (x, y, z) order.
    data = base64.b64encode(q.tobytes(order="F")).decode("ascii")

    spec = {
        "volume": {
            "shape": [int(n) for n in q.shape],
            "origin": [float(o) for o in origin],
            "voxel_size": float(voxel_size),
            "data": data,
        },
        "tracks": tracks or [],
        "e_max": float(e_max),
        "threshold": float(threshold),
        "density": float(density),
        "gamma": float(gamma),
        "smooth": bool(smooth),
        "colorscale": _colorscale_stops(colorscale),
        "colorbar_title": colorbar_title,
        "title": title,
        "width": int(width),
        "height": int(height),
        "axis_length": float(axis_length),
    }

    div_id = "tv-" + uuid.uuid4().hex[:12]
    parts = []
    if lib == "inline":
        for fname, _ in _VENDOR_FILES:
            body = _static(f"vendor/{fname}")
            parts.append(f"<script>if(!window.__tvThreeInline){{{body}\n}}</script>")
        parts.append("<script>window.__tvThreeInline=true;</script>")
        loader = "Promise.resolve()"
    elif lib == "cdn":
        loader = (
            "(window.__tvThree = window.__tvThree || "
            + json.dumps(CDN_URLS)
            + ".reduce((p, u) => p.then(() => new Promise((ok, err) => {"
            "const s = document.createElement('script'); s.src = u; s.onload = ok; s.onerror = err;"
            "document.head.appendChild(s); })), Promise.resolve()))"
        )
    else:
        raise ValueError("lib must be 'cdn' or 'inline'")
    parts.append(f"<script>{_static('volume_view.js')}</script>")
    parts.append(f'<div id="{div_id}" class="track-vis-volume"></div>')
    # Initialise only when scrolled into view (browsers cap live WebGL contexts at ~16 per page).
    parts.append(
        "<script>(function(){"
        f"const el = document.getElementById({json.dumps(div_id)});"
        f"const spec = {json.dumps(spec, separators=(',', ':'))};"
        "let done = false;"
        "const go = () => { if (done) return; done = true;"
        f"{loader}.then(() => TrackVisVolume.render(el, spec))"
        ".catch(e => { el.textContent = 'track_vis: failed to load three.js (' + e + ')'; }); };"
        "if ('IntersectionObserver' in window) {"
        "new IntersectionObserver((es, ob) => { if (es.some(e => e.isIntersecting)) { ob.disconnect(); go(); } },"
        "{ rootMargin: '200px' }).observe(el);"
        f"el.style.minHeight = '{int(height)}px';"
        "} else { go(); }"
        "})();</script>"
    )
    return "\n".join(parts)


def write_volume_html(path: str, title: str = "track_vis volume", **kwargs) -> None:
    """Write ``volume_html(**kwargs)`` as a standalone page."""
    body = volume_html(**kwargs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            '<meta name="viewport" content="width=device-width, initial-scale=1"></head>'
            f'<body style="margin:16px;background:#fff">{body}</body></html>'
        )
