import numpy as np
import pytest

from track_vis import bin_deposits, voxelize


def test_voxelize_places_and_sums_values():
    centers = np.array([[50.0, 50.0, 50.0], [150.0, 50.0, 50.0], [50.0, 250.0, 50.0], [50.0, 250.0, 50.0]])
    grid, origin = voxelize(centers, [1.0, 2.0, 3.0, 1.0], 100.0)
    assert grid.shape == (2, 3, 1)
    np.testing.assert_allclose(origin, [0.0, 0.0, 0.0])
    assert grid[1, 0, 0] == 2.0
    assert grid[0, 2, 0] == 4.0  # repeated centre summed


def test_bin_deposits_conserves_energy_and_aligns_to_grid():
    pts = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [99.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    grid, origin = bin_deposits(pts, [1.0, 2.0, 4.0, 0.0], 100.0)
    assert grid.sum() == pytest.approx(7.0)
    np.testing.assert_allclose(origin, [-100.0, 0.0, 0.0])
    assert grid.shape == (2, 1, 1)
    assert grid[0, 0, 0] == 1.0 and grid[1, 0, 0] == 6.0


def test_bin_deposits_needs_positive_energy():
    with pytest.raises(ValueError, match="positive"):
        bin_deposits(np.zeros((2, 3)), [0.0, 0.0], 10.0)
