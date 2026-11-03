"""
voxels.py

Build dense voxel grids from sparse voxels or point energy deposits.
"""

from __future__ import annotations

import numpy as np


def voxelize(centers: np.ndarray, values: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Scatter per-voxel values at regular-grid centres into a dense array.

    Values of repeated centres are summed.

    Parameters
    ----------
    centers : array_like, shape (N, 3)
        Voxel centre positions (mm), all on one grid of pitch `voxel_size`.
    values : array_like, shape (N,)
        Value of each voxel, e.g. reconstructed energy (MeV).
    voxel_size : float
        Grid pitch (mm).

    Returns
    -------
    grid : numpy.ndarray, shape (nx, ny, nz)
        Dense ``float32`` array indexed by x, y, z voxel index.
    origin : numpy.ndarray, shape (3,)
        Min corner (mm) of voxel ``[0, 0, 0]``.
    """
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    lo = centers.min(axis=0)
    idx = np.rint((centers - lo) / voxel_size).astype(np.int64)
    grid = np.zeros(idx.max(axis=0) + 1, dtype=np.float32)
    np.add.at(grid, tuple(idx.T), np.asarray(values, dtype=np.float32))
    return grid, lo - voxel_size / 2


def bin_deposits(points: np.ndarray, energies: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum point energy deposits into cubes aligned to multiples of `voxel_size`.

    Deposits with zero or negative energy are ignored.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Deposit positions (mm).
    energies : array_like, shape (N,)
        Deposited energy at each point (MeV).
    voxel_size : float
        Cube edge length (mm).

    Returns
    -------
    grid : numpy.ndarray, shape (nx, ny, nz)
        Summed energy per cube, as ``float32``.
    origin : numpy.ndarray, shape (3,)
        Min corner (mm) of cube ``[0, 0, 0]``.

    Raises
    ------
    ValueError
        If no deposit has positive energy.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    energies = np.asarray(energies, dtype=float)
    keep = energies > 0
    if not keep.any():
        raise ValueError("no positive deposits to bin")
    cube = np.floor(points[keep] / voxel_size).astype(np.int64)
    first = cube.min(axis=0)
    idx = cube - first
    grid = np.zeros(idx.max(axis=0) + 1, dtype=np.float32)
    np.add.at(grid, tuple(idx.T), energies[keep].astype(np.float32))
    return grid, first * float(voxel_size)
