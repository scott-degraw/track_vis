"""
display.py

`EventDisplay`: the user-facing API for building 3D event displays.

An `EventDisplay` turns physics inputs (truth tracks, vertices, deposits,
reconstructed voxel images, fitted lines) into the layers of a `Scene`
(see ``scene.py``), choosing colours, grouping by particle species and
writing hover text. It contains no JavaScript or HTML: drawing is delegated to
``browser.py``, which hands the scene to the browser renderer.

In Jupyter an `EventDisplay` renders itself when it is the last expression
of a cell; the same output works in a rendered Quarto page. `to_html` returns
the HTML fragment and `write_html` writes a standalone page.

Units: positions in mm, times in ns, energies in MeV.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from track_vis import browser
from track_vis.colors import colorscale_stops, hex_to_rgb, sample_colorscale
from track_vis.io import particle_name
from track_vis.scene import Colorbar, LineLayer, PointLayer, Scene, VolumeLayer, VolumeStyle
from track_vis.voxels import bin_deposits, voxelize

TIME_COLORSCALE = "plasma"
VOXEL_COLORSCALE = "plasma"
TIME_COLORBAR_TITLE = "Time (ns)"


def _to_numpy(a) -> np.ndarray:
    """Convert an awkward or array-like input to a float NumPy array."""
    return np.asarray(ak.to_numpy(a) if isinstance(a, ak.Array) else a, dtype=float)


def _group_by_species(pdg) -> dict[str, list[int]]:
    """Map particle name to the indices of entries with that PDG code, in first-seen order."""
    groups: dict[str, list[int]] = {}
    for i, code in enumerate(_to_numpy(pdg).astype(int)):
        groups.setdefault(particle_name(int(code)), []).append(i)
    return groups


def _colors(values, vmin, vmax, colorscale, color: str, n: int) -> np.ndarray:
    """Return (n, 3) RGB: `values` sampled on `colorscale` over [vmin, vmax], else `color` for all."""
    if values is None:
        return np.tile(hex_to_rgb(color), (n, 1))
    values = np.asarray(values, dtype=float)
    lo, hi = _value_range(values, vmin, vmax)
    return sample_colorscale(colorscale, (values - lo) / ((hi - lo) or 1.0))


def _value_range(values, vmin, vmax) -> tuple[float, float]:
    """Return (vmin, vmax), each defaulting to the min or max of `values`."""
    values = np.asarray(values, dtype=float)
    return (float(values.min()) if vmin is None else vmin, float(values.max()) if vmax is None else vmax)


def _is_single_path(points) -> bool:
    """Return True for one (N, 3) polyline, False for a list of them."""
    if isinstance(points, np.ndarray):
        return points.ndim == 2
    first = next(iter(points), None)
    return first is None or np.ndim(first) == 1


def _polyline_segments(paths: list[np.ndarray], colors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Split polylines (each (N, 3), with per-point RGB) into segment end points, two rows per segment."""
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
    Build an interactive 3D display of an event.

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
    bounds : array_like, shape (2, 3), optional
        Fixed view box ``[[x_min, y_min, z_min], [x_max, y_max, z_max]]``
        in mm, e.g. to give several displays the same view or to keep far
        away hidden layers from shrinking the event. Default is None, which
        fits the box to all layers.

    Attributes
    ----------
    scene : Scene
        The layers added so far and the view settings (see ``scene.py``).
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
        bounds=None,
    ):
        if bounds is not None:
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
                raise ValueError("bounds must be [[x_min, y_min, z_min], [x_max, y_max, z_max]] with max > min")
        self.scene = Scene(
            title=title,
            width=int(width),
            height=int(height),
            style=VolumeStyle(threshold=threshold, density=density, gamma=gamma, smooth=smooth),
            bounds=bounds,
        )

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
        self.scene.volumes.append(
            VolumeLayer(
                name=name,
                grid=grid,
                origin=origin,
                voxel_size=float(voxel_size),
                e_max=float(e_max) or 1.0,  # an all-zero image still needs a scale
                colorscale=colorscale_stops(colorscale),
                colorbar_title=colorbar_title,
                visible=visible,
            )
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
        segments, segment_colors = _polyline_segments(paths, np.split(rgb, np.cumsum(lengths)[:-1]))
        hover_points = None
        if hover_text is not None:
            hover_points = np.concatenate(paths)
            if len(hover_text) != len(hover_points):
                raise ValueError("hover_text needs one entry per point")
            hover_text = [str(t) for t in hover_text]
        self.scene.lines.append(
            LineLayer(
                name=name,
                segments=segments,
                colors=segment_colors,
                width=width,
                dashed=dashed,
                dash=dash,
                hover_points=hover_points,
                hover_text=hover_text,
                visible=visible,
            )
        )
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
        self._set_colorbar(TIME_COLORBAR_TITLE, colorscale, t_min, t_max)
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
        colorbar_title: str | None = None,
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
        colorbar_title : str, optional
            With `values`, also show a colour bar with this title (a later
            bar with the same title replaces it). Default is None, no bar.
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
        text = [str(t) for t in hover_text] if hover_text is not None else [""] * len(pts)
        if len(text) != len(pts):
            raise ValueError("hover_text needs one entry per point")
        self.scene.points.append(
            PointLayer(name=name, positions=pts, colors=rgb, size=size, hover_text=text, visible=visible)
        )
        if values is not None and colorbar_title is not None:
            lo, hi = _value_range(values, vmin, vmax)
            self._set_colorbar(colorbar_title, colorscale, lo, hi)
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
        self._set_colorbar(TIME_COLORBAR_TITLE, colorscale, t_min, t_max)
        return self

    def _set_colorbar(self, title: str, colorscale, vmin: float, vmax: float) -> None:
        """Add a colour bar, replacing any earlier one with the same title."""
        bar = Colorbar(title, colorscale_stops(colorscale), vmin, vmax)
        self.scene.colorbars = [b for b in self.scene.colorbars if b.title != title] + [bar]

    # ------------------------------------------------------------------ output
    def to_html(self, lib: str = "cdn") -> str:
        """
        Return an HTML fragment that draws this display.

        Parameters
        ----------
        lib : {"cdn", "inline"}, optional
            "cdn" loads three.js from pinned jsDelivr URLs; "inline" embeds
            it so no network is needed. Default is "cdn".

        Returns
        -------
        str
            The HTML fragment.

        Raises
        ------
        ValueError
            If `lib` is not "cdn" or "inline".
        """
        return browser.scene_html(self.scene, lib)

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
        with open(path, "w", encoding="utf-8") as f:
            f.write(browser.page_html(self.scene, lib, page_title))


class DisplayMenu:
    """
    Several event displays in one view, chosen from a drop-down menu.

    Use it to page through events without one WebGL view per event: only the
    display on show is drawn, so a page can hold many menus.

    Parameters
    ----------
    displays : list of EventDisplay
        The displays, in menu order. The first is shown at the start.
    labels : list of str, optional
        Menu entry for each display. Default is each display's title.

    Attributes
    ----------
    displays : list of EventDisplay
        The displays, in menu order.
    labels : list of str
        The menu entries.

    Raises
    ------
    ValueError
        If `displays` is empty or `labels` has the wrong length.
    """

    def __init__(self, displays: list[EventDisplay], labels: list[str] | None = None):
        if not displays:
            raise ValueError("DisplayMenu needs at least one display")
        if labels is None:
            labels = [d.scene.title or f"display {i}" for i, d in enumerate(displays)]
        if len(labels) != len(displays):
            raise ValueError("labels needs one entry per display")
        self.displays = list(displays)
        self.labels = [str(label) for label in labels]

    def to_html(self, lib: str = "cdn") -> str:
        """
        Return an HTML fragment with the menu and the view.

        Parameters
        ----------
        lib : {"cdn", "inline"}, optional
            How three.js is included (see `EventDisplay.to_html`). Default
            is "cdn".

        Returns
        -------
        str
            The HTML fragment.

        Raises
        ------
        ValueError
            If `lib` is not "cdn" or "inline".
        """
        return browser.menu_html([d.scene for d in self.displays], self.labels, lib)

    def _repr_html_(self) -> str:
        """Render in Jupyter and Quarto (three.js from the CDN)."""
        return self.to_html()

    def write_html(self, path: str, lib: str = "inline", page_title: str | None = None) -> None:
        """
        Write the menu to a standalone HTML page.

        Parameters
        ----------
        path : str
            Output file path.
        lib : {"inline", "cdn"}, optional
            How three.js is included. Default is "inline", so the page
            works offline.
        page_title : str, optional
            Browser tab title. Default is the first menu entry.

        Raises
        ------
        ValueError
            If `lib` is not "cdn" or "inline".
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(browser.page(self.to_html(lib), page_title or self.labels[0]))
