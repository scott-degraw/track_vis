import numpy as np
import pytest

from track_vis import colorscale_stops, sample_colorscale


def test_named_and_explicit_scales():
    stops = colorscale_stops("Plasma")
    assert stops[0] == [0.0, "#0d0887"] and stops[-1][0] == 1.0
    assert colorscale_stops(["#000000", "#ffffff"]) == [[0.0, "#000000"], [1.0, "#ffffff"]]
    with pytest.raises(ValueError):
        colorscale_stops("nope")


def test_sampling_interpolates_and_clips():
    rgb = sample_colorscale(["#000000", "#ffffff"], [0.0, 0.5, 2.0])
    np.testing.assert_allclose(rgb[0], 0.0)
    np.testing.assert_allclose(rgb[1], 0.5)
    np.testing.assert_allclose(rgb[2], 1.0)  # clipped
