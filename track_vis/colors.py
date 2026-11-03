"""
colors.py

Built-in colour scales and helpers to sample them.

The scales are evenly spaced hex stops of the matplotlib/Plotly definitions
of plasma, viridis, inferno, magma, cividis and turbo, so the package needs
no plotting library.
"""

from __future__ import annotations

import numpy as np

COLORSCALES: dict[str, list[str]] = {
    "plasma": [
        "#0d0887",
        "#46039f",
        "#7201a8",
        "#9c179e",
        "#bd3786",
        "#d8576b",
        "#ed7953",
        "#fb9f3a",
        "#fdca26",
        "#f0f921",
    ],
    "viridis": [
        "#440154",
        "#482878",
        "#3e4989",
        "#31688e",
        "#26828e",
        "#1f9e89",
        "#35b779",
        "#6ece58",
        "#b5de2b",
        "#fde725",
    ],
    "inferno": [
        "#000004",
        "#1b0c41",
        "#4a0c6b",
        "#781c6d",
        "#a52c60",
        "#cf4446",
        "#ed6925",
        "#fb9b06",
        "#f7d13d",
        "#fcffa4",
    ],
    "magma": [
        "#000004",
        "#180f3d",
        "#440f76",
        "#721f81",
        "#9e2f7f",
        "#cd4071",
        "#f1605d",
        "#fd9668",
        "#feca8d",
        "#fcfdbf",
    ],
    "cividis": [
        "#00224e",
        "#123570",
        "#3b496c",
        "#575d6d",
        "#707173",
        "#8a8678",
        "#a59c74",
        "#c3b369",
        "#e1cc55",
        "#fee838",
    ],
    "turbo": [
        "#30123b",
        "#4145ab",
        "#4675ed",
        "#39a2fc",
        "#1bcfd4",
        "#24eca6",
        "#61fc6c",
        "#a4fc3b",
        "#d1e834",
        "#f3c63a",
        "#fe9b2d",
        "#f36315",
        "#d93806",
        "#b11901",
        "#7a0402",
    ],
}


def colorscale_stops(colorscale: str | list) -> list[list]:
    """
    Return the stops of a colour scale as ``[[pos, "#rrggbb"], ...]``.

    Parameters
    ----------
    colorscale : str or list
        A name from `COLORSCALES` (case-insensitive), a list of hex colours
        (spaced evenly over [0, 1]), or a list of ``[pos, "#rrggbb"]`` pairs.

    Returns
    -------
    list of list
        ``[position, hex colour]`` pairs with positions in [0, 1].

    Raises
    ------
    ValueError
        If `colorscale` is a name not in `COLORSCALES`.
    """
    if isinstance(colorscale, str):
        try:
            colors = COLORSCALES[colorscale.lower()]
        except KeyError:
            raise ValueError(f"unknown colorscale {colorscale!r}; choose from {sorted(COLORSCALES)}") from None
        n = len(colors) - 1
        return [[i / n, c] for i, c in enumerate(colors)]
    scale = list(colorscale)
    if scale and isinstance(scale[0], str):
        n = len(scale) - 1
        return [[i / n, c] for i, c in enumerate(scale)]
    return [[float(p), str(c)] for p, c in scale]


def hex_to_rgb(color: str) -> np.ndarray:
    """
    Convert a ``"#rrggbb"`` colour to RGB values in [0, 1].

    Parameters
    ----------
    color : str
        Hex colour string with a leading ``#``.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Red, green and blue in [0, 1].
    """
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255.0


def sample_colorscale(colorscale: str | list, u: np.ndarray) -> np.ndarray:
    """
    Map values in [0, 1] to colours by linear interpolation between stops.

    Parameters
    ----------
    colorscale : str or list
        Colour scale name or stops (see `colorscale_stops`).
    u : array_like, shape (N,)
        Positions on the scale; values outside [0, 1] are clipped.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        RGB colours in [0, 1].

    Raises
    ------
    ValueError
        If `colorscale` is an unknown name.
    """
    stops = colorscale_stops(colorscale)
    pos = np.array([p for p, _ in stops])
    rgb = np.array([hex_to_rgb(c) for _, c in stops])
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return np.stack([np.interp(u, pos, rgb[:, k]) for k in range(3)], axis=-1)
