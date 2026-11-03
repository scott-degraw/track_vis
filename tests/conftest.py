import awkward as ak
import numpy as np
import pytest


@pytest.fixture
def tracks():
    """Three tracks (two electrons, one gamma) in the ntuple's columnar layout."""
    return {
        "pdg": np.array([11, 11, 22]),
        "pos_x": ak.Array([[0.0, 10.0, 20.0], [0.0, -5.0], [0.0, 0.0, 0.0, 0.0]]),
        "pos_y": ak.Array([[0.0, 0.0, 0.0], [0.0, 5.0], [0.0, 10.0, 20.0, 30.0]]),
        "pos_z": ak.Array([[0.0, 1.0, 2.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        "time": ak.Array([[0.0, 0.1, 0.2], [0.0, 0.05], [0.0, 1.0, 2.0, 3.0]]),
        "deposited_energy": ak.Array([[0.1, 0.2, 0.3], [0.05, 0.05], [0.0, 0.0, 0.01, 0.0]]),
    }


@pytest.fixture
def vertices():
    """Three vertices, two of them electrons."""
    return {
        "pdg": np.array([11, 11, 2212]),
        "x": np.zeros(3),
        "y": np.zeros(3),
        "z": np.zeros(3),
        "t": np.array([0.0, 0.5, 1.0]),
        "energy": np.array([1.0, 2.0, 3.0]),
    }
