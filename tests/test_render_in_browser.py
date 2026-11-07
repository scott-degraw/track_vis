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
from pathlib import Path

import numpy as np
import pytest

from track_vis import DisplayMenu, EventDisplay
from track_vis.browser import page

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
    # The controls start hidden behind the "options" button, which shows and hides them.
    options = page.get_by_role("button", name="options")
    assert options.get_attribute("aria-expanded") == "false"
    assert not page.get_by_text("reset view").is_visible()
    options.click()
    assert page.get_by_text("reset view").is_visible()
    options.click()
    assert not page.get_by_text("reset view").is_visible()
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


def test_menu_switches_scenes(browser, tmp_path, tracks, vertices):
    path = tmp_path / "menu.html"
    first = _display(tracks, vertices)
    second = EventDisplay(title="second", bounds=[[-100, -100, -100], [100, 100, 100]]).add_vertices(**vertices)
    DisplayMenu([first, second], ["one", "two"]).write_html(str(path))

    page = browser.new_page(viewport={"width": 960, "height": 760})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(path.as_uri())
    page.wait_for_function("document.querySelector('.track-vis div')?._trackvis !== undefined", timeout=15000)
    assert page.locator(".track-vis canvas").count() == 1
    page.get_by_role("button", name="options").click()  # open the controls
    page.select_option(".track-vis select", "1")
    page.wait_for_function("document.querySelector('.track-vis div')._trackvis?.spec.title === 'second'", timeout=15000)
    assert page.locator(".track-vis canvas").count() == 1  # the first view was disposed
    assert page.get_by_text("reset view").is_visible()  # the next event keeps them open
    assert errors == []
    page.close()


# What an AMD loader such as require.js (which Quarto and Jupyter pages load) leaves on
# the page: three.js's UMD build would register with it instead of setting window.THREE.
FAKE_AMD_LOADER = "<script>window.define = function () {}; window.define.amd = {};</script>"
VENDOR = Path(__file__).resolve().parent.parent / "track_vis" / "js" / "vendor"


@pytest.mark.parametrize("lib", ["inline", "cdn"])
def test_renders_next_to_an_amd_loader(browser, tmp_path, tracks, lib):
    path = tmp_path / "amd.html"
    path.write_text(page(FAKE_AMD_LOADER + EventDisplay().add_tracks(**tracks).to_html(lib=lib)))

    page_ = browser.new_page()
    # Serve the pinned CDN URLs from the vendored copies, so the test needs no network.
    page_.route(
        "https://cdn.jsdelivr.net/**",
        lambda route: route.fulfill(
            content_type="text/javascript", body=(VENDOR / route.request.url.rsplit("/", 1)[-1]).read_text()
        ),
    )
    errors = []
    page_.on("pageerror", lambda e: errors.append(str(e)))
    page_.goto(path.as_uri())
    page_.wait_for_function(
        "document.querySelector('.track-vis')?._trackvis !== undefined"
        " || document.querySelector('.track-vis')?.textContent.startsWith('track_vis:')",
        timeout=15000,
    )
    assert page_.locator(".track-vis").inner_text().startswith("track_vis:") is False
    assert errors == []
    page_.close()
