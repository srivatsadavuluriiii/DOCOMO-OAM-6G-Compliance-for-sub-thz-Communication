#!/usr/bin/env python3
"""
Beam width utility for Gaussian beams.

Provides `get_beam_width` as referenced by tests. Uses standard Gaussian beam
propagation: w(z) = w0 * sqrt(1 + (z/z_R)^2), z_R = pi w0^2 / lambda.
"""

import numpy as np


def get_beam_width(z: float, w0: float = 0.01135, wavelength: float = 1.07e-2) -> float:
    """Return Gaussian beam radius w(z) at propagation distance z (in meters).

    Args:
        z: Propagation distance in meters
        w0: Beam waist radius at z=0 (meters)
        wavelength: Wavelength (meters)

    Returns:
        Beam radius w(z) in meters
    """
    if w0 <= 0 or wavelength <= 0:
        raise ValueError("w0 and wavelength must be positive")
    z_r = np.pi * (w0 ** 2) / wavelength
    return float(w0 * np.sqrt(1.0 + (z / z_r) ** 2))


