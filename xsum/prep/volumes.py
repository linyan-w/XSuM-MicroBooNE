"""Detector volumes (cm)."""
import numpy as np

from ..constants import ACTIVE_X, ACTIVE_Y, ACTIVE_Z


def in_active_volume(xyz: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """True where the point lies inside the TPC active volume shrunk by `margin` on every side."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return ((x > ACTIVE_X[0] + margin) & (x < ACTIVE_X[1] - margin) &
            (y > ACTIVE_Y[0] + margin) & (y < ACTIVE_Y[1] - margin) &
            (z > ACTIVE_Z[0] + margin) & (z < ACTIVE_Z[1] - margin))
