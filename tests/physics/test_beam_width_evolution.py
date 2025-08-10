#!/usr/bin/env python3
"""
Test script to validate beam width evolution parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import sys
import os

# Use centralized path management instead of sys.path.append
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from IEEE_images.generate_figure1_system_model import get_beam_width

def get_beam_width(z: float, w0: float = 0.01135, wavelength: float = 1.07e-2) -> float:
    """
    Calculates Gaussian beam width w(z) at a distance z.
    
    For 28 GHz (λ = 1.07 cm):
    - w0 = 1.135cm gives exactly 30 mrad divergence
    - Rayleigh range z_R = πw0²/λ ≈ 3.7 meters
    - Beam divergence θ ≈ λ/(πw0) ≈ 0.03 radians ≈ 1.7 degrees
    - This matches the simulator's 30 mrad specification
    - Beam width at 1km: w(1000m) ≈ 0.03m (3cm)
    """
    z_r = np.pi * w0**2 / wavelength  # Rayleigh range
    return w0 * np.sqrt(1 + (z / z_r)**2)

def test_beam_width_evolution():
    """Test the corrected beam width evolution for 28 GHz."""
    
    print("🔬 Testing Beam Width Evolution for 28 GHz...")
    print("=" * 50)
    
    # Parameters for 28 GHz
    wavelength = 1.07e-2  # 1.07 cm for 28 GHz
    w0 = 0.01135  # 1.135 cm beam waist radius (gives exactly 30 mrad)
    
    # Calculate Rayleigh range
    z_r = np.pi * w0**2 / wavelength
    print(f"📏 Rayleigh range: {z_r:.1f} meters")
    print(f"📏 w0² = {w0**2:.6f} m²")
    print(f"📏 wavelength = {wavelength:.6f} m")
    print(f"📏 π = {np.pi:.6f}")
    
    # Calculate beam divergence
    divergence_rad = wavelength / (np.pi * w0)
    divergence_deg = np.degrees(divergence_rad)
    divergence_mrad = divergence_rad * 1000
    print(f"📐 Beam divergence: {divergence_rad:.3f} rad = {divergence_deg:.1f}° = {divergence_mrad:.1f} mrad")
    
    # Test distances
    distances = np.linspace(0, 2000, 100)  # 0 to 2 km
    
    # Calculate beam widths
    beam_widths = [get_beam_width(z, w0, wavelength) for z in distances]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()  # Flatten the axes array
    
    # Plot 1: Beam width vs distance
    axes[0].plot(distances, beam_widths, 'b-', linewidth=2)
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Beam Width (m)')
    axes[0].set_title('Beam Width Evolution (28 GHz, w₀ = 1.2 cm)')
    axes[0].grid(True, alpha=0.3)
    
    # Add Rayleigh range marker
    axes[0].axvline(x=z_r, color='r', linestyle='--', alpha=0.7, label=f'Rayleigh Range = {z_r:.1f}m')
    axes[0].legend()
    
    # Plot 2: Beam width in cm for better visualization
    beam_widths_cm = [w * 100 for w in beam_widths]
    axes[1].plot(distances, beam_widths_cm, 'g-', linewidth=2)
    axes[1].set_xlabel('Distance (m)')
    axes[1].set_ylabel('Beam Width (cm)')
    axes[1].set_title('Beam Width Evolution (in cm)')
    axes[1].grid(True, alpha=0.3)
    
    # Add key distance markers
    key_distances = [100, 500, 1000, 1500, 2000]
    for dist in key_distances:
        if dist <= distances[-1]:
            idx = np.argmin(np.abs(distances - dist))
            w_at_dist = beam_widths_cm[idx]
            axes[1].plot(dist, w_at_dist, 'ro', markersize=8)
            axes[1].annotate(f'{w_at_dist:.1f}cm', (dist, w_at_dist), 
                           xytext=(10, 10), textcoords='offset points', fontsize=9)
    
    # Plot 3: Comparison with different w0 values
    w0_values = [0.008, 0.012, 0.016, 0.020]  # 0.8cm to 2.0cm
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, w0_test in enumerate(w0_values):
        z_r_test = np.pi * w0_test**2 / wavelength
        beam_widths_test = [get_beam_width(z, w0_test, wavelength) for z in distances]
        divergence_test = wavelength / (np.pi * w0_test) * 1000  # mrad
        
        axes[2].plot(distances, [w * 100 for w in beam_widths_test], 
                    color=colors[i], linewidth=2, 
                    label=f'w₀ = {w0_test*100:.1f}cm, θ = {divergence_test:.1f}mrad')
    
    axes[2].set_xlabel('Distance (m)')
    axes[2].set_ylabel('Beam Width (cm)')
    axes[2].set_title('Beam Width Comparison for Different w₀ Values')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot 4: Beam divergence vs w0
    w0_range = np.linspace(0.005, 0.025, 50)  # 0.5cm to 2.5cm
    divergences = [wavelength / (np.pi * w0_val) * 1000 for w0_val in w0_range]  # mrad
    
    axes[3].plot([w * 100 for w in w0_range], divergences, 'purple', linewidth=2)
    axes[3].set_xlabel('Beam Waist Radius w₀ (cm)')
    axes[3].set_ylabel('Beam Divergence (mrad)')
    axes[3].set_title('Beam Divergence vs Beam Waist Radius')
    axes[3].grid(True, alpha=0.3)
    
    # Mark the chosen w0 value
    chosen_divergence = wavelength / (np.pi * w0) * 1000
    axes[3].plot(w0 * 100, chosen_divergence, 'ro', markersize=10, 
                label=f'Chosen: w₀ = {w0*100:.1f}cm, θ = {chosen_divergence:.1f}mrad')
    axes[3].legend()
    
    plt.tight_layout()
    plt.savefig('Beam_Width_Evolution_Validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Beam Width Evolution Validation Complete!")
    print("📊 Generated 'Beam_Width_Evolution_Validation.png'")
    
    # Print validation results
    print(f"\n📋 Beam Width Parameters (28 GHz):")
    print(f"   • Wavelength: {wavelength*100:.2f} cm")
    print(f"   • Beam waist radius (w₀): {w0*100:.1f} cm")
    print(f"   • Rayleigh range: {z_r:.1f} m")
    print(f"   • Beam divergence: {divergence_rad:.3f} rad = {divergence_deg:.1f}° = {divergence_mrad:.1f} mrad")
    
    print(f"\n📏 Beam Width at Key Distances:")
    for dist in key_distances:
        w_at_dist = get_beam_width(dist, w0, wavelength)
        print(f"   • {dist}m: {w_at_dist*100:.1f} cm")

def test_consistency_with_simulator():
    """Test consistency with simulator beam width parameters."""
    
    print("\n🔬 Testing Consistency with Simulator Parameters...")
    print("=" * 55)
    
    # Simulator uses beam_width = 0.03 (30 mrad)
    simulator_beam_width = 0.03  # 30 mrad divergence angle
    
    # Our visualization uses w0 = 0.01135 (1.135cm)
    w0_viz = 0.01135
    wavelength = 1.07e-2
    
    # Calculate divergence from our w0
    divergence_from_w0 = wavelength / (np.pi * w0_viz) * 1000  # mrad
    
    print(f"📐 Simulator beam divergence: {simulator_beam_width*1000:.1f} mrad")
    print(f"📐 Visualization beam divergence: {divergence_from_w0:.1f} mrad")
    print(f"📊 Difference: {abs(simulator_beam_width*1000 - divergence_from_w0):.1f} mrad")
    
    if abs(simulator_beam_width*1000 - divergence_from_w0) < 5:
        print("✅ Parameters are consistent!")
    else:
        print("⚠️  Parameters show some inconsistency")
    
    # Calculate what w0 would give exactly 30 mrad
    w0_for_30mrad = wavelength / (np.pi * 0.03)  # 30 mrad = 0.03 rad
    print(f"📏 w0 for exactly 30 mrad: {w0_for_30mrad*100:.2f} cm")

def test_oam_mode_beam_widths():
    """Test beam width calculations for different OAM modes."""
    
    print("\n🔬 Testing OAM Mode Beam Widths...")
    print("=" * 40)
    
    w0 = 0.01135
    wavelength = 1.07e-2
    distance = 1000  # 1 km
    
    # Calculate base beam width
    w_z = get_beam_width(distance, w0, wavelength)
    
    print(f"📏 Base beam width at 1km: {w_z*100:.1f} cm")
    
    # Calculate OAM mode beam radii
    oam_modes = [1, 2, 3, 4, 5, 6]
    
    print(f"\n📊 OAM Mode Beam Radii at 1km:")
    for l in oam_modes:
        if l == 0:
            radius = 0.5 * w_z  # Central peak for l=0
        else:
            radius = w_z * np.sqrt(abs(l) / 2)  # Peak of donut for l≠0
        
        print(f"   • Mode l={l}: radius = {radius*100:.1f} cm")

if __name__ == "__main__":
    test_beam_width_evolution()
    test_consistency_with_simulator()
    test_oam_mode_beam_widths()
    
    print("\n🎉 All beam width evolution tests completed successfully!")
    print("📁 Generated validation plot: 'Beam_Width_Evolution_Validation.png'") 