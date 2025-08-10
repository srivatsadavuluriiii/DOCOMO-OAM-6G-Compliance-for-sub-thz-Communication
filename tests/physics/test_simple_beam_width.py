#!/usr/bin/env python3
"""
Simple test to verify beam width calculation with realistic parameters.
"""

import numpy as np

def get_beam_width(z: float, w0: float = 0.05, wavelength: float = 1.07e-2) -> float:
    """Calculate Gaussian beam width w(z) at distance z."""
    z_r = np.pi * w0**2 / wavelength  # Rayleigh range
    return w0 * np.sqrt(1 + (z / z_r)**2)

# Test parameters
wavelength = 1.07e-2  # 1.07 cm for 28 GHz
w0 = 0.05  # 5 cm beam waist radius

# Calculate Rayleigh range
z_r = np.pi * w0**2 / wavelength
print(f"Rayleigh range: {z_r:.3f} meters")

# Calculate beam divergence
divergence_rad = wavelength / (np.pi * w0)
divergence_mrad = divergence_rad * 1000
print(f"Beam divergence: {divergence_mrad:.1f} mrad")

# Test beam widths at different distances
distances = [0, 10, 50, 100, 500, 1000]  # meters

print(f"\nBeam widths at different distances:")
for z in distances:
    w_z = get_beam_width(z, w0, wavelength)
    print(f"  • {z}m: {w_z*100:.1f} cm")

# Verify the calculation is reasonable
print(f"\nVerification:")
print(f"  • At z=0: w(0) = {get_beam_width(0, w0, wavelength)*100:.1f} cm (should be {w0*100:.1f} cm)")
print(f"  • At z=z_r: w(z_r) = {get_beam_width(z_r, w0, wavelength)*100:.1f} cm (should be {w0*100*np.sqrt(2):.1f} cm)")
print(f"  • At z>>z_r: w(z) ≈ {w0*100*np.sqrt(2):.1f} * z/z_r cm") 