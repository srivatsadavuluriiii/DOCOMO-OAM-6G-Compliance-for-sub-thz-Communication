#!/usr/bin/env python3
"""
Enhanced utilities for Gaussian and Laguerre-Gaussian (LG) beams used in tests.

Implements:
 - get_beam_width(z, w0, wavelength)
 - get_lg_beam_intensity(l, r_array, w, p)

The LG intensity follows the standard form (up to a constant factor):
I_l,p(r) ∝ (2 r^2 / w^2)^{|l|} [L_p^{|l|}(2 r^2 / w^2)]^2 exp(-2 r^2 / w^2)

For test normalization, absolute scaling is not required; relative form suffices.
"""

from typing import Iterable
import numpy as np
from scipy.special import genlaguerre


def get_beam_width(z: float, w0: float = 0.05, wavelength: float = 1.07e-2) -> float:
    """Gaussian beam radius w(z) with Rayleigh range z_R = pi w0^2 / lambda."""
    if w0 <= 0 or wavelength <= 0:
        raise ValueError("w0 and wavelength must be positive")
    z_r = np.pi * (w0 ** 2) / wavelength
    return float(w0 * np.sqrt(1.0 + (z / z_r) ** 2))


def get_lg_beam_intensity(l: int, r_array: Iterable[float], w: float, p: int = 0) -> np.ndarray:
    """Compute relative LG mode intensity profile at given radial coordinates.

    Args:
        l: Azimuthal index (can be negative; only |l| matters for intensity)
        r_array: Radial coordinates (meters) as iterable
        w: Beam radius parameter (meters)
        p: Radial index (non-negative integer)

    Returns:
        NumPy array of relative intensities with same length as r_array
    """
    if w <= 0:
        raise ValueError("w must be positive")
    if p < 0:
        raise ValueError("p must be non-negative")

    r = np.asarray(r_array, dtype=float)
    abs_l = abs(int(l))
    p = int(p)

    # Normalized radius squared
    x = 2.0 * (r ** 2) / (w ** 2)

    # Associated generalized Laguerre polynomial L_p^{|l|}(x)
    L = genlaguerre(p, abs_l)(x)

    # Relative intensity (omit absolute normalization constant)
    intensity = (x ** abs_l) * (L ** 2) * np.exp(-x)

    # Ensure finite values
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    return intensity.astype(float)


