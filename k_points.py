"""Uniform Brillouin-zone sampling for the graphene reciprocal lattice."""

import numpy as np

from constants import b1, b2
from validation import integer


__all__ = ["monkhorst_pack_mesh"]


def monkhorst_pack_mesh(grid_size):
    """Return an equal-weight, Gamma-centered grid containing K and K'."""
    grid_size = integer("grid_size", grid_size, minimum=1)
    if grid_size % 3 != 0:
        raise ValueError("grid_size must be divisible by 3 to include K and K'.")
    coordinates = (np.arange(grid_size, dtype=float) - grid_size // 2) / grid_size
    r1, r2 = np.meshgrid(coordinates, coordinates, indexing="ij")
    points = (
        r1.ravel()[:, None] * np.asarray(b1[:2], dtype=float)
        + r2.ravel()[:, None] * np.asarray(b2[:2], dtype=float)
    )
    weights = np.full(grid_size**2, 1.0 / grid_size**2)
    return points, weights
