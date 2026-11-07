"""
Write a gallery of synthetic event displays, to check how the renderer looks.

    uv run python examples/gallery.py OUTDIR [--theme light|dark]

writes one standalone page per example to OUTDIR. The events are fake but
shaped like the SNO+ imaging studies: a reconstructed muon image with its
truth track, a two-electron image, binned truth deposits of a small shower,
and a drop-down menu of muon events.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import awkward as ak
import numpy as np

from track_vis import DisplayMenu, EventDisplay


def tube_image(start, end, voxel, half_width, sigma, rng, noise=0.08, bead=0.0):
    """A blurred line segment on a voxel grid, with optional MLEM-like beading."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    lo = np.minimum(start, end) - half_width
    hi = np.maximum(start, end) + half_width
    lo = np.floor(lo / voxel) * voxel
    hi = np.ceil(hi / voxel) * voxel
    axes = [np.arange(a, b, voxel) + voxel / 2 for a, b in zip(lo, hi)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    d = end - start
    s = np.clip(((pts - start) @ d) / (d @ d), 0, 1)
    dist = np.linalg.norm(pts - (start + s[..., None] * d), axis=-1)
    img = np.exp(-0.5 * (dist / sigma) ** 2) * (0.7 + 0.3 * s)
    if bead:
        img *= 1 + bead * np.cos(2 * np.pi * s * np.linalg.norm(d) / (3 * voxel)) ** 2
    img *= rng.gamma(1 / noise**2, noise**2, img.shape) if noise else 1
    img[img < 0.02 * img.max()] = 0
    return img, lo


def muon_tracks(start, direction, length, rng, t0=0.0):
    """A straight mu+ track plus a few short delta-ray electrons, in the columnar schema."""
    direction = np.asarray(direction, float) / np.linalg.norm(direction)
    n = 60
    s = np.linspace(0, length, n)
    mu = np.asarray(start) + s[:, None] * direction + rng.normal(0, 1.5, (n, 3)).cumsum(0) * 0.3
    tracks = [(-13, mu, t0 + s / 299.8, np.full(n, 0.2 * length / n / 10))]
    for k in rng.choice(np.arange(5, n - 5), 3, replace=False):
        m = 12
        dd = rng.normal(size=3)
        dd = 0.5 * dd / np.linalg.norm(dd) + direction
        step = rng.uniform(3, 6)
        e = mu[k] + np.cumsum(np.tile(dd / np.linalg.norm(dd), (m, 1)) * step + rng.normal(0, 2, (m, 3)), 0)
        tracks.append((11, e, tracks[0][2][k] + np.arange(m) * 0.02, np.full(m, 0.05)))
    tracks.append(
        (
            22,
            mu[-1] + np.outer(np.linspace(0, 380, 8), rng.normal(size=3) / 1.7),
            np.linspace(0, 1.3, 8) + tracks[0][2][-1],
            np.zeros(8),
        )
    )
    return {
        "pdg": np.array([t[0] for t in tracks]),
        "pos_x": ak.Array([t[1][:, 0].tolist() for t in tracks]),
        "pos_y": ak.Array([t[1][:, 1].tolist() for t in tracks]),
        "pos_z": ak.Array([t[1][:, 2].tolist() for t in tracks]),
        "time": ak.Array([t[2].tolist() for t in tracks]),
        "deposited_energy": ak.Array([t[3].tolist() for t in tracks]),
    }


def time_range(tracks):
    t = ak.to_numpy(ak.flatten(tracks["time"]))
    return float(t.min()), float(t.max())


def muon_display(seed, theme_kw, title=None):
    rng = np.random.default_rng(seed)
    start = rng.uniform(-300, 300, 3)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    length = rng.uniform(380, 620)
    end = start + direction * length
    voxel = 50.0
    img, origin = tube_image(start, end, voxel, 200, 45, rng, bead=0.6)
    tracks = muon_tracks(start, direction, length, rng)
    t_min, t_max = time_range(tracks)
    vertices = {
        "pdg": np.array([-13]),
        "x": start[:1],
        "y": start[1:2],
        "z": start[2:],
        "t": np.array([0.0]),
        "energy": np.array([140.0]),
    }
    fit = np.stack([start + 0.03 * length * direction, end - 0.05 * length * direction])
    return (
        EventDisplay(title=title or f"140 MeV mu+, 50 mm voxels (seed {seed})", **theme_kw)
        .add_volume(img, origin=origin, voxel_size=voxel, name="MLEM image", colorbar_title="Image (MeV)")
        .add_tracks(**tracks, t_min=t_min, t_max=t_max)
        .add_vertices(**vertices, t_min=t_min, t_max=t_max)
        .add_line(fit, "fitted axis", color="#3b82f6", dashed=True, width=2)
    )


def two_electron_display(theme_kw):
    rng = np.random.default_rng(7)
    voxel = 100.0
    g = np.arange(-800, 800, voxel) + voxel / 2
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1)
    a, b = np.array([-220.0, 80, 40]), np.array([260.0, -140, -60])
    img = 1.0 * np.exp(-0.5 * np.sum((pts - a) ** 2, -1) / 110**2) + 0.7 * np.exp(
        -0.5 * np.sum((pts - b) ** 2, -1) / 120**2
    )
    img *= rng.gamma(80, 1 / 80, img.shape)
    img[img < 0.03] = 0
    v = {
        "pdg": np.array([11, 11]),
        "x": np.array([a[0], b[0]]),
        "y": np.array([a[1], b[1]]),
        "z": np.array([a[2], b[2]]),
        "t": np.array([0.0, 0.0]),
        "energy": np.array([1.17, 1.33]),
    }
    return (
        EventDisplay(title="Two electrons, 1.17 + 1.33 MeV, 100 mm voxels", **theme_kw)
        .add_volume(img, origin=g[0] - voxel / 2, voxel_size=voxel, name="MLEM image", colorbar_title="Image (MeV)")
        .add_vertices(**v, t_min=0.0, t_max=1.0)
    )


def shower_display(theme_kw):
    rng = np.random.default_rng(42)

    def make(direction, n, step, t0, dedx, curl):
        direction = np.asarray(direction, float) / np.linalg.norm(direction)
        perp = np.cross(direction, [0, 0, 1])
        perp /= np.linalg.norm(perp)
        p = [np.zeros(3)]
        for i in range(1, n):
            p.append(p[-1] + direction * step + perp * np.sin(curl * i) * step)
        return np.array(p), t0 + np.arange(n) * step / 200, dedx * rng.uniform(0.7, 1.3, n)

    parts = [
        (11, [1, 0.3, 0.2], 40, 15, 0, 0.3, 0.15),
        (2212, [-0.5, 1, 0.1], 25, 20, 0.5, 1.2, 0),
        (22, [-0.2, -0.8, 0.9], 30, 25, 0.2, 0.05, 0),
    ]
    cols = {k: [] for k in ("pos_x", "pos_y", "pos_z", "time", "deposited_energy")}
    for _, d, n, st, t0, dedx, curl in parts:
        p, t, e = make(d, n, st, t0, dedx, curl)
        for k, a in zip(("pos_x", "pos_y", "pos_z"), p.T):
            cols[k].append(a.tolist())
        cols["time"].append(t.tolist())
        cols["deposited_energy"].append(e.tolist())
    tracks = {"pdg": np.array([p[0] for p in parts]), **{k: ak.Array(v) for k, v in cols.items()}}
    t_min, t_max = time_range(tracks)
    vertices = {
        "pdg": np.array([11, 2212]),
        "x": np.array([0.0, 5]),
        "y": np.array([0.0, -3]),
        "z": np.array([0.0, 2]),
        "t": np.array([0.0, 1.1]),
        "energy": np.array([2.5, 0.8]),
    }
    return (
        EventDisplay(title="Truth deposits: e- / p / gamma from a common vertex", **theme_kw)
        .add_deposits(tracks["pos_x"], tracks["pos_y"], tracks["pos_z"], tracks["deposited_energy"], voxel_size=10.0)
        .add_tracks(**tracks, t_min=t_min, t_max=t_max)
        .add_vertices(**vertices, t_min=t_min, t_max=t_max)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--theme", default=None, help="passed to EventDisplay(theme=...) if supported")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    theme_kw = {"theme": args.theme} if args.theme and "theme" in inspect.signature(EventDisplay).parameters else {}
    muon_display(3, theme_kw).write_html(str(out / "muon.html"))
    two_electron_display(theme_kw).write_html(str(out / "two_electrons.html"))
    shower_display(theme_kw).write_html(str(out / "deposits.html"))
    DisplayMenu([muon_display(s, theme_kw) for s in (1, 2, 4)], [f"muon event {s}" for s in (1, 2, 4)]).write_html(
        str(out / "menu.html")
    )


if __name__ == "__main__":
    main()
