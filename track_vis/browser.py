"""
browser.py

Turn a `Scene` into HTML. This is the only Python module that deals with
JavaScript.

The output is a fragment of three parts: the renderer script
(``js/event_display.js``), an empty ``<div>`` for the view, and one call
``TrackVis.mount(div_id, scene_json, script_urls)`` that draws the scene into
the div (or ``TrackVis.mountMenu(div_id, scenes, labels, script_urls)`` for a
drop-down of scenes). The renderer needs three.js r147, either loaded from pinned jsDelivr
URLs (``lib="cdn"``) or embedded from the copy in ``js/vendor``
(``lib="inline"``, ~660 kB, works offline).
"""

from __future__ import annotations

import html
import json
import uuid
from functools import cache
from importlib import resources

from track_vis.scene import Scene

THREE_VERSION = "0.147.0"
# (file in js/vendor, path inside the three npm package), in load order.
THREE_FILES = [
    ("three.min.js", "build/three.min.js"),
    ("OrbitControls.js", "examples/js/controls/OrbitControls.js"),
    ("LineSegmentsGeometry.js", "examples/js/lines/LineSegmentsGeometry.js"),
    ("LineMaterial.js", "examples/js/lines/LineMaterial.js"),
    ("LineSegments2.js", "examples/js/lines/LineSegments2.js"),
]
CDN_URLS = [f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/{path}" for _, path in THREE_FILES]


@cache
def _read_js(name: str) -> str:
    """Return the text of a file under ``track_vis/js``."""
    return resources.files("track_vis").joinpath("js", name).read_text(encoding="utf-8")


def _script(code: str, once_key: str | None = None) -> str:
    """Wrap JavaScript in a <script> tag; with `once_key` it runs once per page."""
    if once_key is not None:
        flag = f"window[{json.dumps('__trackvis_' + once_key)}]"
        code = f"if (!{flag}) {{ {flag} = true;\n{code}\n}}"
    return f"<script>{code}</script>"


def _hide_amd(code: str) -> str:
    """
    Run vendored three.js code with no AMD ``define`` in scope.

    three.min.js is a UMD build: on pages with an AMD loader (require.js, which
    Jupyter and Quarto pages load) it would register as a module instead of
    setting ``window.THREE``, which the renderer reads.
    """
    return f"(function (define) {{\n{code}\n}})();"


def _to_script_json(value) -> str:
    """Serialise to JSON that is safe inside <script> (``</`` cannot end the tag)."""
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def scene_html(scene: Scene, lib: str = "cdn") -> str:
    """
    Return an HTML fragment that draws `scene`.

    The view initialises when it scrolls near the viewport, since browsers
    allow only about 16 live WebGL views per page.

    Parameters
    ----------
    scene : Scene
        The scene to draw.
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
    return _fragment("mount", [scene.to_json()], scene.height, lib)


def menu_html(scenes: list[Scene], labels: list[str], lib: str = "cdn") -> str:
    """
    Return an HTML fragment that draws one of `scenes`, picked from a menu.

    Parameters
    ----------
    scenes : list of Scene
        The scenes, in menu order.
    labels : list of str
        Menu entry for each scene.
    lib : {"cdn", "inline"}, optional
        How three.js is included (see `scene_html`). Default is "cdn".

    Returns
    -------
    str
        The HTML fragment.

    Raises
    ------
    ValueError
        If `lib` is not "cdn" or "inline".
    """
    height = max(s.height for s in scenes) + 30  # room for the menu
    return _fragment("mountMenu", [[s.to_json() for s in scenes], list(labels)], height, lib)


def _fragment(mount_function: str, mount_args: list, height: int, lib: str) -> str:
    """Scripts, a view div and ``TrackVis.<mount_function>(div_id, *mount_args, urls)``."""
    if lib == "inline":
        parts = [_script(_hide_amd(_read_js(f"vendor/{name}")), once_key=name) for name, _ in THREE_FILES]
        urls = None
    elif lib == "cdn":
        parts = []
        urls = CDN_URLS
    else:
        raise ValueError("lib must be 'cdn' or 'inline'")
    div_id = "track-vis-" + uuid.uuid4().hex[:12]
    args = ", ".join(_to_script_json(a) for a in (div_id, *mount_args, urls))
    parts += [
        _script(_read_js("event_display.js")),
        f'<div id="{div_id}" class="track-vis" style="min-height:{int(height)}px"></div>',
        _script(f"TrackVis.{mount_function}({args});"),
    ]
    return "\n".join(parts)


def page_html(scene: Scene, lib: str = "inline", title: str | None = None) -> str:
    """
    Return a complete standalone HTML page that draws `scene`.

    Parameters
    ----------
    scene : Scene
        The scene to draw.
    lib : {"inline", "cdn"}, optional
        How three.js is included (see `scene_html`). Default is "inline",
        so the page works offline.
    title : str, optional
        Browser tab title. Default is the scene title.

    Returns
    -------
    str
        The HTML page.

    Raises
    ------
    ValueError
        If `lib` is not "cdn" or "inline".
    """
    return page(scene_html(scene, lib), title or scene.title)


def page(fragment: str, title: str | None = None) -> str:
    """
    Wrap an HTML fragment from `scene_html` or `menu_html` in a page.

    Parameters
    ----------
    fragment : str
        The body.
    title : str, optional
        Browser tab title.

    Returns
    -------
    str
        The HTML page.
    """
    title = html.escape(title or "track_vis event display")
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
        '<meta name="viewport" content="width=device-width, initial-scale=1"></head>'
        f'<body style="margin:16px;background:#fff">{fragment}</body></html>'
    )
