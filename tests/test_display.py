import base64
import json
import re

import awkward as ak
import numpy as np
import pytest

from track_vis import EventDisplay, bin_deposits, colorscale_stops, sample_colorscale, voxelize


def _mounted_spec(html):
    """The spec passed to TrackVis.mount in an HTML fragment."""
    args = re.search(r"TrackVis\.mount\((.*)\);</script>", html).group(1)
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(args)  # element id
    spec, _ = decoder.raw_decode(args, end + len(", "))
    return spec


def _decode_f32(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


@pytest.fixture
def tracks():
    return {
        "pdg": np.array([11, 11, 22]),
        "pos_x": ak.Array([[0.0, 10.0, 20.0], [0.0, -5.0], [0.0, 0.0, 0.0, 0.0]]),
        "pos_y": ak.Array([[0.0, 0.0, 0.0], [0.0, 5.0], [0.0, 10.0, 20.0, 30.0]]),
        "pos_z": ak.Array([[0.0, 1.0, 2.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        "time": ak.Array([[0.0, 0.1, 0.2], [0.0, 0.05], [0.0, 1.0, 2.0, 3.0]]),
        "deposited_energy": ak.Array([[0.1, 0.2, 0.3], [0.05, 0.05], [0.0, 0.0, 0.01, 0.0]]),
    }


def test_voxelize_roundtrip():
    centers = np.array([[50.0, 50.0, 50.0], [150.0, 50.0, 50.0], [50.0, 250.0, 50.0]])
    grid, origin = voxelize(centers, [1.0, 2.0, 3.0], 100.0)
    assert grid.shape == (2, 3, 1)
    np.testing.assert_allclose(origin, [0.0, 0.0, 0.0])
    assert grid[1, 0, 0] == 2.0 and grid[0, 2, 0] == 3.0
    assert grid.sum() == pytest.approx(6.0)


def test_bin_deposits_conserves_energy_and_aligns():
    pts = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [99.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    grid, origin = bin_deposits(pts, [1.0, 2.0, 4.0, 0.0], 100.0)
    assert grid.sum() == pytest.approx(7.0)
    np.testing.assert_allclose(origin, [-100.0, 0.0, 0.0])
    assert grid.shape == (2, 1, 1)
    assert grid[0, 0, 0] == 1.0 and grid[1, 0, 0] == 6.0


def test_colorscales():
    stops = colorscale_stops("Plasma")
    assert stops[0] == [0.0, "#0d0887"] and stops[-1][0] == 1.0
    rgb = sample_colorscale("plasma", [0.0, 1.0, 2.0])
    np.testing.assert_allclose(rgb[0], np.array([0x0D, 0x08, 0x87]) / 255)
    np.testing.assert_allclose(rgb[1], rgb[2])  # clipped
    assert colorscale_stops(["#000000", "#ffffff"]) == [[0.0, "#000000"], [1.0, "#ffffff"]]
    with pytest.raises(ValueError):
        colorscale_stops("nope")


def test_volume_texture_is_x_fastest():
    grid = np.zeros((3, 2, 2), dtype=np.float32)
    grid[2, 0, 0] = 1.0  # x = 2 -> linear index 2 in x-fastest order
    d = EventDisplay().add_volume(grid, (0, 0, 0), 10.0, pad=0)
    v = d.volumes[0]
    raw = np.frombuffer(base64.b64decode(v["data"]), dtype=np.uint8)
    assert v["shape"] == [3, 2, 2]
    assert raw[2] == 255 and raw.sum() == 255
    d2 = EventDisplay().add_volume(grid, (0, 0, 0), 10.0)  # default pad=1
    assert d2.volumes[0]["shape"] == [5, 4, 4]
    assert d2.volumes[0]["origin"] == [-10.0, -10.0, -10.0]


def test_add_tracks_groups_by_species(tracks):
    d = EventDisplay().add_tracks(**tracks)
    assert len(d.lines) == 2  # e- and gamma layers
    electron = d.lines[0]
    seg = _decode_f32(electron["segments"]).reshape(-1, 3)
    assert len(seg) == 2 * (2 + 1)  # 2 + 1 segments, two endpoints each
    assert len(electron["hover"]["text"]) == 5
    assert "2 tracks" in electron["name"]
    assert [b["title"] for b in d.colorbars] == ["Time (ns)"]
    assert d.colorbars[0]["max"] == pytest.approx(3.0)


def test_vertices_and_deposits(tracks):
    vertices = {
        "pdg": np.array([11, 11, 2212]),
        "x": np.zeros(3),
        "y": np.zeros(3),
        "z": np.zeros(3),
        "t": np.array([0.0, 0.5, 1.0]),
        "energy": np.array([1.0, 2.0, 3.0]),
    }
    d = EventDisplay().add_vertices(**vertices, t_min=0, t_max=3)
    d.add_deposits(tracks["pos_x"], tracks["pos_y"], tracks["pos_z"], tracks["deposited_energy"], voxel_size=10)
    assert len(d.points) == 2 and len(d.points[0]["hover"]) == 2
    assert len(d.volumes) == 1
    assert len(d.colorbars) == 1  # a second time bar replaces the first


def test_html_fragment(tracks):
    d = EventDisplay(title="t").add_tracks(**tracks).add_line([[0, 0, 0], [1, 1, 1]], "axis", dashed=True)
    out = d.to_html()
    assert 'class="track-vis"' in out and "TrackVis.mount(" in out
    assert "cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js" in out
    spec = _mounted_spec(out)
    assert spec["title"] == "t" and len(spec["lines"]) == 3
    inline = d.to_html(lib="inline")
    assert "cdn.jsdelivr" not in inline and inline.count("__trackvis_three.min.js") == 2
    assert d._repr_html_().startswith("<script>")
    with pytest.raises(ValueError):
        d.to_html(lib="nope")


def test_line_colours_follow_values_across_polylines():
    paths = [np.zeros((2, 3)), np.ones((3, 3))]
    d = EventDisplay().add_line(paths, values=[[0.0, 0.0], [1.0, 1.0, 1.0]], colorscale=["#000000", "#ffffff"])
    col = _decode_f32(d.lines[0]["colors"]).reshape(-1, 3)
    assert len(col) == 2 * (1 + 2)
    np.testing.assert_allclose(col[:2], 0.0)  # first polyline at vmin
    np.testing.assert_allclose(col[2:], 1.0)  # second at vmax
    single = EventDisplay().add_line(np.zeros((4, 3)), values=np.arange(4.0))
    assert len(_decode_f32(single.lines[0]["segments"])) == 3 * 2 * 3


def test_input_errors():
    d = EventDisplay()
    with pytest.raises(ValueError, match="voxel_size"):
        d.add_volume(np.zeros((2, 2, 2)), (0, 0, 0))
    with pytest.raises(ValueError, match="origin"):
        d.add_volume(np.zeros((2, 2, 2)), voxel_size=1.0)
    with pytest.raises(ValueError, match="3D"):
        d.add_volume(np.zeros((2, 2)), (0, 0, 0), 1.0)
    with pytest.raises(ValueError, match="hover_text"):
        d.add_line(np.zeros((3, 3)), hover_text=["a"])
    with pytest.raises(ValueError, match="hover_text"):
        d.add_points(np.zeros((2, 3)), hover_text=["a"])
    with pytest.raises(ValueError, match="positive"):
        bin_deposits(np.zeros((2, 3)), [0.0, 0.0], 10.0)


def test_title_cannot_close_the_script_tag():
    out = EventDisplay(title="</script><b>x</b>").to_html()
    assert "</script><b>" not in out
    assert _mounted_spec(out)["title"] == "</script><b>x</b>"
