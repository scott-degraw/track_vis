"""HTML output: the only place the Python and JavaScript halves meet."""

import json
import re

import pytest

from track_vis import EventDisplay
from track_vis.browser import CDN_URLS


def _mounted_scene(html):
    """The scene JSON passed to TrackVis.mount in an HTML fragment."""
    args = re.search(r"TrackVis\.mount\((.*)\);</script>", html).group(1)
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(args)  # element id
    scene, _ = decoder.raw_decode(args, end + len(", "))
    return scene


def test_cdn_fragment(tracks):
    out = EventDisplay(title="t").add_tracks(**tracks).to_html()
    assert 'class="track-vis"' in out
    assert all(url in out for url in CDN_URLS)
    scene = _mounted_scene(out)
    assert scene["title"] == "t" and len(scene["lines"]) == 2


def test_inline_fragment_embeds_three_once_per_page(tracks):
    out = EventDisplay().add_tracks(**tracks).to_html(lib="inline")
    assert "cdn.jsdelivr" not in out
    assert out.count("__trackvis_three.min.js") == 2  # the guard reads and sets one flag


def test_title_cannot_close_the_script_tag():
    out = EventDisplay(title="</script><b>x</b>").to_html()
    assert "</script><b>" not in out
    assert _mounted_scene(out)["title"] == "</script><b>x</b>"


def test_bad_lib_and_standalone_page(tmp_path):
    d = EventDisplay(title="A & B")
    with pytest.raises(ValueError, match="lib"):
        d.to_html(lib="nope")
    path = tmp_path / "page.html"
    d.write_html(str(path))
    text = path.read_text()
    assert text.startswith("<!doctype html>") and "<title>A &amp; B</title>" in text
