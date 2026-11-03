"""Scene.to_json encodes arrays the way js/event_display.js decodes them."""

import base64

import numpy as np

from track_vis.scene import LineLayer, Scene, VolumeLayer, encode_float32, encode_volume


def _decode_f32(s):
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def test_volume_is_uint8_and_x_fastest():
    grid = np.zeros((3, 2, 2))
    grid[2, 0, 0] = 1.0  # x = 2 is linear index 2 when x varies fastest
    grid[0, 1, 0] = 0.5  # y = 1 is linear index 3
    raw = np.frombuffer(base64.b64decode(encode_volume(grid, e_max=1.0)), dtype=np.uint8)
    assert raw[2] == 255 and raw[3] == 128 and raw.sum() == 255 + 128


def test_float_arrays_round_trip():
    a = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(_decode_f32(encode_float32(a)), a.ravel())


def test_scene_json_fields():
    scene = Scene(title="t")
    scene.volumes.append(VolumeLayer("v", np.ones((2, 2, 2)), np.zeros(3), 10.0, 1.0, [[0, "#000000"]], "E"))
    scene.lines.append(LineLayer("l", np.zeros((2, 3)), np.zeros((2, 3))))
    out = scene.to_json()
    assert out["title"] == "t" and out["volume_style"]["threshold"] == 0.05
    assert out["volumes"][0]["shape"] == [2, 2, 2]
    assert out["lines"][0]["hover"] is None
