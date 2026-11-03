"""
Smoke test of the JavaScript renderer in headless Chromium.

pytest's other tests cover the Python side only. This one writes a standalone
page with every kind of layer, opens it in a real browser with software WebGL,
and checks that the renderer runs without errors and draws something.

It is skipped when Playwright or its browser is not installed, unless the
environment variable ``TRACK_VIS_REQUIRE_BROWSER=1`` is set (as in CI), in which
case a missing browser is a failure. Set ``TRACK_VIS_CHROMIUM`` to use a
specific Chromium executable.
"""

import os

import numpy as np
import pytest

from track_vis import EventDisplay

REQUIRE = os.environ.get("TRACK_VIS_REQUIRE_BROWSER") == "1"
SOFTWARE_GL = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]


@pytest.fixture(scope="module")
def browser():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        if REQUIRE:
            raise
        pytest.skip("playwright not installed")
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(args=SOFTWARE_GL, executable_path=os.environ.get("TRACK_VIS_CHROMIUM"))
        except Exception as e:  # any launch failure means no usable browser
            if REQUIRE:
                raise
            pytest.skip(f"no Chromium for Playwright: {e}")
        yield b
        b.close()


def _display(tracks, vertices):
    grid = np.zeros((6, 6, 6))
    grid[2:4, 2:4, 1:5] = 1.0
    return (
        EventDisplay(title="smoke test")
        .add_volume(grid, origin=(-30.0, -30.0, -30.0), voxel_size=10.0)
        .add_tracks(**tracks)
        .add_vertices(**vertices)
        .add_line([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]], "dashed axis", dashed=True)
    )


def test_page_renders_without_errors(browser, tmp_path, tracks, vertices):
    path = tmp_path / "page.html"
    _display(tracks, vertices).write_html(str(path))

    page = browser.new_page(viewport={"width": 960, "height": 700})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(path.as_uri())
    page.wait_for_function("document.querySelector('.track-vis')?._trackvis !== undefined", timeout=15000)

    assert errors == []
    # One checkbox per layer: volume, two track species, two vertex species, the line.
    labels = page.locator(".track-vis label").all_inner_texts()
    assert sum("smooth" in t or "max-intensity" in t for t in labels) == 2
    assert len(labels) == 3 + 2 + 6  # sliders, render-mode checkboxes, layers
    # Something other than the white background was drawn.
    coloured = page.evaluate(
        """() => {
            const gl = document.querySelector('.track-vis canvas').getContext('webgl2');
            const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
            document.querySelector('.track-vis')._trackvis.draw();  // fresh frame to read back
            const px = new Uint8Array(4 * w * h);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
            let n = 0;
            for (let i = 0; i < px.length; i += 4) if (px[i] + px[i + 1] + px[i + 2] < 600) n++;
            return n;
        }"""
    )
    assert coloured > 1000
    page.close()
