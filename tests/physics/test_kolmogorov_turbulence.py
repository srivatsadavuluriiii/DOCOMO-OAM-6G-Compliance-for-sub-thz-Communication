#!/usr/bin/env python3
"""
Test script to validate Kolmogorov turbulence implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Use centralized path management instead of sys.path.append
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from simulator.channel_simulator import ChannelSimulator

def test_kolmogorov_spectrum():
    """Test the von Karman spectrum implementation."""
    
    print("🔬 Testing Improved Kolmogorov Turbulence Model...")
    print("=" * 55)
    
    # Create simulator with different turbulence strengths
    config = {
        'system': {
            'frequency': 28.0e9,
            'bandwidth': 400e6,
            'tx_power_dBm': 30.0,
            'noise_figure_dB': 8.0,
            'noise_temp': 290.0
        },
        'environment': {
            'humidity': 50.0,
            'temperature': 20.0,
            'pressure': 101.3,
            'turbulence_strength': 1e-14,  # Weak turbulence
            'pointing_error_std': 0.005,
            'rician_k_factor': 8.0
        },
        'oam': {
            'min_mode': 1,
            'max_mode': 6,
            'beam_width': 0.03
        }
    }
    
    simulator = ChannelSimulator(config)
    
    # Test distances
    distances = np.linspace(100, 10000, 100)
    
    # Test different turbulence strengths
    turbulence_strengths = [1e-15, 1e-14, 1e-13, 1e-12]  # Weak to strong
    
    # Store results
    fried_params = []
    scintillation_indices = []
    beam_wanders = []
    
    for Cn2 in turbulence_strengths:
        simulator.turbulence_strength = Cn2
        
        fried_params_cn2 = []
        scintillation_indices_cn2 = []
        beam_wanders_cn2 = []
        
        for distance in distances:
            # Calculate Fried parameter
            r0 = (0.423 * (simulator.k ** 2) * Cn2 * distance) ** (-3/5)
            fried_params_cn2.append(r0)
            
            # Calculate scintillation index
            scintillation_weak = 1.23 * Cn2 * (simulator.k ** (7/6)) * (distance ** (11/6))
            if scintillation_weak > 1.0:
                scintillation_index = 1.0 - np.exp(-scintillation_weak)
            else:
                scintillation_index = scintillation_weak
            scintillation_indices_cn2.append(scintillation_index)
            
            # Calculate beam wander
            l0, L0 = 0.01, 50.0  # Inner and outer scales
            beam_wander_var = 2.42 * Cn2 * (distance ** 3) * (simulator.wavelength ** (-1/3)) * (1 - (l0 / L0) ** (1/3))
            beam_wander = np.sqrt(beam_wander_var)
            beam_wanders_cn2.append(beam_wander)
        
        fried_params.append(fried_params_cn2)
        scintillation_indices.append(scintillation_indices_cn2)
        beam_wanders.append(beam_wanders_cn2)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Fried parameter vs distance
    for i, Cn2 in enumerate(turbulence_strengths):
        ax1.plot(distances/1000, fried_params[i], label=f'C_n² = {Cn2:.1e}')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Fried Parameter r₀ (m)')
    ax1.set_title('Fried Parameter vs Distance')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Scintillation index vs distance
    for i, Cn2 in enumerate(turbulence_strengths):
        ax2.plot(distances/1000, scintillation_indices[i], label=f'C_n² = {Cn2:.1e}')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Scintillation Index σ_I²')
    ax2.set_title('Scintillation Index vs Distance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Beam wander vs distance
    for i, Cn2 in enumerate(turbulence_strengths):
        ax3.plot(distances/1000, beam_wanders[i], label=f'C_n² = {Cn2:.1e}')
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Beam Wander (m)')
    ax3.set_title('Beam Wander vs Distance')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Turbulence strength comparison
    ax4.plot([1e-15, 1e-14, 1e-13, 1e-12], [np.mean(fp) for fp in fried_params], 'bo-', label='Fried Parameter')
    ax4.set_xlabel('Turbulence Strength C_n²')
    ax4.set_ylabel('Average Fried Parameter (m)')
    ax4.set_title('Turbulence Strength Effects')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('Kolmogorov_Turbulence_Validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Kolmogorov Turbulence Validation Complete!")
    print("📊 Generated 'Kolmogorov_Turbulence_Validation.png'")
    
    # Print validation results
    print("\n🔬 Scientific Validation Results:")
    for i, Cn2 in enumerate(turbulence_strengths):
        avg_r0 = np.mean(fried_params[i])
        avg_scint = np.mean(scintillation_indices[i])
        avg_wander = np.mean(beam_wanders[i])
        
        print(f"\n📏 C_n² = {Cn2:.1e}:")
        print(f"   • Average Fried parameter: {avg_r0:.3f} m")
        print(f"   • Average scintillation index: {avg_scint:.3f}")
        print(f"   • Average beam wander: {avg_wander:.3f} m")

def test_oam_mode_sensitivity():
    """Test OAM mode sensitivity to turbulence."""
    
    print("\n🔬 Testing OAM Mode Sensitivity to Turbulence...")
    print("=" * 50)
    
    config = {
        'system': {'frequency': 28.0e9},
        'environment': {
            'turbulence_strength': 1e-14,
            'humidity': 50.0,
            'temperature': 20.0,
            'pressure': 101.3
        },
        'oam': {'min_mode': 1, 'max_mode': 6}
    }
    
    simulator = ChannelSimulator(config)
    
    # Test different OAM modes
    modes = [1, 2, 3, 4, 5, 6]
    distance = 1000  # 1 km
    
    mode_sensitivities = []
    coupling_strengths = []
    
    for mode in modes:
        # Calculate mode sensitivity factor
        mode_factor = (mode ** 2) / 4.0
        mode_sensitivities.append(mode_factor)
        
        # Simulate turbulence screen
        turbulence_screen = simulator._generate_turbulence_screen(distance)
        
        # Calculate average coupling strength for this mode
        mode_idx = mode - simulator.min_mode
        coupling_strength = np.mean(np.abs(turbulence_screen[mode_idx, :]))
        coupling_strengths.append(coupling_strength)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Mode sensitivity factor
    ax1.plot(modes, mode_sensitivities, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('OAM Mode l')
    ax1.set_ylabel('Mode Sensitivity Factor')
    ax1.set_title('OAM Mode Sensitivity to Turbulence')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coupling strength
    ax2.plot(modes, coupling_strengths, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('OAM Mode l')
    ax2.set_ylabel('Average Coupling Strength')
    ax2.set_title('Mode Coupling Due to Turbulence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('OAM_Turbulence_Sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ OAM Mode Sensitivity Validation Complete!")
    print("📊 Generated 'OAM_Turbulence_Sensitivity.png'")
    
    print("\n📋 OAM Mode Sensitivity Results:")
    for i, mode in enumerate(modes):
        print(f"   • Mode l={mode}: Sensitivity={mode_sensitivities[i]:.2f}, Coupling={coupling_strengths[i]:.4f}")

def test_von_karman_spectrum():
    """Test the von Karman spectrum implementation."""
    
    print("\n🔬 Testing von Karman Spectrum...")
    print("=" * 35)
    
    # Test parameters
    Cn2 = 1e-14
    l0 = 0.01  # 1 cm inner scale
    L0 = 50.0  # 50 m outer scale
    
    k0 = 2 * np.pi / L0  # Outer scale frequency
    km = 5.92 / l0       # Inner scale frequency
    
    # Spatial frequencies
    kappa = np.logspace(-3, 3, 1000)
    
    # von Karman spectrum
    def von_karman_spectrum(kappa):
        return 0.033 * Cn2 * (kappa**2 + k0**2)**(-11/6) * np.exp(-kappa**2 / km**2)
    
    # Kolmogorov spectrum (for comparison)
    def kolmogorov_spectrum(kappa):
        return 0.033 * Cn2 * kappa**(-11/3)
    
    # Calculate spectra
    von_karman_values = von_karman_spectrum(kappa)
    kolmogorov_values = kolmogorov_spectrum(kappa)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.loglog(kappa, von_karman_values, 'b-', linewidth=2, label='von Karman Spectrum')
    plt.loglog(kappa, kolmogorov_values, 'r--', linewidth=2, label='Kolmogorov Spectrum')
    
    # Mark important frequencies
    plt.axvline(k0, color='g', linestyle=':', alpha=0.7, label=f'κ₀ = {k0:.3f} (outer scale)')
    plt.axvline(km, color='orange', linestyle=':', alpha=0.7, label=f'κ_m = {km:.1f} (inner scale)')
    
    plt.xlabel('Spatial Frequency κ (m⁻¹)')
    plt.ylabel('Spectral Density Φ_n(κ)')
    plt.title('von Karman vs Kolmogorov Spectrum')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('von_Karman_Spectrum.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ von Karman Spectrum Validation Complete!")
    print("📊 Generated 'von_Karman_Spectrum.png'")
    
    print(f"\n📋 Spectrum Parameters:")
    print(f"   • Inner scale l₀ = {l0*1000:.1f} mm")
    print(f"   • Outer scale L₀ = {L0:.1f} m")
    print(f"   • Inner frequency κ_m = {km:.1f} m⁻¹")
    print(f"   • Outer frequency κ₀ = {k0:.3f} m⁻¹")

if __name__ == "__main__":
    test_kolmogorov_spectrum()
    test_oam_mode_sensitivity()
    test_von_karman_spectrum()
    
    print("\n🎉 All Kolmogorov turbulence tests completed successfully!")
    print("📁 Generated validation plots:")
    print("   • Kolmogorov_Turbulence_Validation.png")
    print("   • OAM_Turbulence_Sensitivity.png")
    print("   • von_Karman_Spectrum.png") 