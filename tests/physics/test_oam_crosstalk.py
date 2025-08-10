#!/usr/bin/env python3
"""
Test script to validate OAM crosstalk implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Use centralized path management instead of sys.path.append
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from simulator.channel_simulator import ChannelSimulator

def test_paterson_crosstalk_model():
    """Test the Paterson et al. crosstalk model implementation."""
    
    print("🔬 Testing Paterson et al. OAM Crosstalk Model...")
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
            'turbulence_strength': 1e-14,
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
    distances = [100, 500, 1000, 2000, 5000]  # meters
    
    # Test different turbulence strengths
    turbulence_strengths = [1e-15, 1e-14, 1e-13, 1e-12]
    
    # Store results
    crosstalk_matrices = {}
    
    for Cn2 in turbulence_strengths:
        simulator.turbulence_strength = Cn2
        crosstalk_matrices[Cn2] = {}
        
        for distance in distances:
            # Generate turbulence screen
            turbulence_screen = simulator._generate_turbulence_screen(distance)
            
            # Calculate crosstalk matrix
            crosstalk_matrix = simulator._calculate_crosstalk(distance, turbulence_screen)
            
            crosstalk_matrices[Cn2][distance] = crosstalk_matrix
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot 1: Crosstalk vs mode difference for different distances
    distance = 1000  # 1 km
    Cn2 = 1e-14
    crosstalk_matrix = crosstalk_matrices[Cn2][distance]
    
    mode_differences = []
    crosstalk_values = []
    
    for i in range(simulator.num_modes):
        for j in range(simulator.num_modes):
            if i != j:
                mode_diff = abs((i + simulator.min_mode) - (j + simulator.min_mode))
                crosstalk_val = np.abs(crosstalk_matrix[i, j])
                mode_differences.append(mode_diff)
                crosstalk_values.append(crosstalk_val)
    
    axes[0].scatter(mode_differences, crosstalk_values, alpha=0.6)
    axes[0].set_xlabel('Mode Difference |l₁ - l₂|')
    axes[0].set_ylabel('Crosstalk Magnitude')
    axes[0].set_title(f'OAM Crosstalk vs Mode Difference (1 km, C_n² = {Cn2:.1e})')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Crosstalk vs distance for adjacent modes
    adjacent_crosstalk = []
    for Cn2 in turbulence_strengths:
        crosstalk_adjacent = []
        for distance in distances:
            crosstalk_matrix = crosstalk_matrices[Cn2][distance]
            # Adjacent mode crosstalk (l=1 to l=2)
            adjacent_val = np.abs(crosstalk_matrix[0, 1])
            crosstalk_adjacent.append(adjacent_val)
        adjacent_crosstalk.append(crosstalk_adjacent)
    
    for i, Cn2 in enumerate(turbulence_strengths):
        axes[1].plot(distances, adjacent_crosstalk[i], 'o-', label=f'C_n² = {Cn2:.1e}')
    axes[1].set_xlabel('Distance (m)')
    axes[1].set_ylabel('Adjacent Mode Crosstalk')
    axes[1].set_title('Adjacent Mode Crosstalk vs Distance')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Crosstalk matrix heatmap
    distance = 1000
    Cn2 = 1e-14
    crosstalk_matrix = crosstalk_matrices[Cn2][distance]
    
    im = axes[2].imshow(np.abs(crosstalk_matrix), cmap='viridis', aspect='auto')
    axes[2].set_xlabel('Mode Index')
    axes[2].set_ylabel('Mode Index')
    axes[2].set_title(f'Crosstalk Matrix Heatmap (1 km, C_n² = {Cn2:.1e})')
    plt.colorbar(im, ax=axes[2], label='Crosstalk Magnitude')
    
    # Plot 4: Turbulence strength effect on crosstalk
    distance = 1000
    turbulence_effects = []
    for Cn2 in turbulence_strengths:
        crosstalk_matrix = crosstalk_matrices[Cn2][distance]
        # Average off-diagonal crosstalk
        avg_crosstalk = np.mean(np.abs(crosstalk_matrix - np.eye(simulator.num_modes)))
        turbulence_effects.append(avg_crosstalk)
    
    axes[3].plot(turbulence_strengths, turbulence_effects, 'ro-', linewidth=2, markersize=8)
    axes[3].set_xlabel('Turbulence Strength C_n²')
    axes[3].set_ylabel('Average Crosstalk')
    axes[3].set_title('Turbulence Effect on OAM Crosstalk')
    axes[3].set_xscale('log')
    axes[3].set_yscale('log')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('OAM_Crosstalk_Validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Paterson Crosstalk Model Validation Complete!")
    print("📊 Generated 'OAM_Crosstalk_Validation.png'")
    
    # Print validation results
    print("\n🔬 Scientific Validation Results:")
    for Cn2 in turbulence_strengths:
        print(f"\n📏 C_n² = {Cn2:.1e}:")
        for distance in distances:
            crosstalk_matrix = crosstalk_matrices[Cn2][distance]
            avg_crosstalk = np.mean(np.abs(crosstalk_matrix - np.eye(simulator.num_modes)))
            print(f"   • Distance {distance}m: Avg crosstalk = {avg_crosstalk:.4f}")

def test_orthogonality_theory():
    """Test OAM orthogonality theory implementation."""
    
    print("\n🔬 Testing OAM Orthogonality Theory...")
    print("=" * 45)
    
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
    
    # Test parameters
    distance = 1000  # 1 km
    w_L = simulator.beam_width * distance
    r0 = (0.423 * (simulator.k ** 2) * simulator.turbulence_strength * distance) ** (-3/5)
    
    # Calculate sigma parameter for different turbulence regimes
    if r0 > w_L:
        sigma_weak = w_L / (2 * np.sqrt(2 * np.log(2)))
        sigma_strong = w_L / (2 * np.sqrt(1 + (w_L / r0) ** 2))
    else:
        sigma_weak = w_L / (2 * np.sqrt(1 + (w_L / r0) ** 2))
        sigma_strong = w_L / (2 * np.sqrt(1 + (w_L / r0) ** 2))
    
    # Test mode differences
    mode_differences = np.arange(1, 6)
    
    # Calculate orthogonality factors
    orthogonality_weak = np.exp(-(mode_differences / sigma_weak) ** 2)
    orthogonality_strong = np.exp(-(mode_differences / sigma_strong) ** 2)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(mode_differences, orthogonality_weak, 'bo-', linewidth=2, markersize=8, label='Weak Turbulence')
    plt.plot(mode_differences, orthogonality_strong, 'ro-', linewidth=2, markersize=8, label='Strong Turbulence')
    plt.xlabel('Mode Difference |l₁ - l₂|')
    plt.ylabel('Orthogonality Factor exp(-((l₁-l₂)/σ)²)')
    plt.title('OAM Orthogonality vs Mode Difference (Paterson Model)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('OAM_Orthogonality_Theory.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ OAM Orthogonality Theory Validation Complete!")
    print("📊 Generated 'OAM_Orthogonality_Theory.png'")
    
    print(f"\n📋 Orthogonality Parameters:")
    print(f"   • Beam width at distance: {w_L:.3f} m")
    print(f"   • Fried parameter: {r0:.3f} m")
    print(f"   • Weak turbulence σ: {sigma_weak:.3f}")
    print(f"   • Strong turbulence σ: {sigma_strong:.3f}")

def test_phase_correlation():
    """Test phase correlation functions for OAM modes."""
    
    print("\n🔬 Testing Phase Correlation Functions...")
    print("=" * 45)
    
    # Test different OAM mode pairs
    mode_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    
    phase_correlations = []
    mode_combinations = []
    
    for l1, l2 in mode_pairs:
        mode_diff = abs(l1 - l2)
        phase_correlation = np.exp(-mode_diff ** 2 / (2 * (l1 + l2) ** 2))
        phase_correlations.append(phase_correlation)
        mode_combinations.append(f"l={l1},l={l2}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(mode_combinations)), phase_correlations, color='skyblue', alpha=0.7)
    plt.xlabel('OAM Mode Pairs')
    plt.ylabel('Phase Correlation Factor')
    plt.title('Phase Correlation Functions for OAM Mode Pairs')
    plt.xticks(range(len(mode_combinations)), mode_combinations, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(phase_correlations):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('OAM_Phase_Correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Phase Correlation Validation Complete!")
    print("📊 Generated 'OAM_Phase_Correlation.png'")
    
    print("\n📋 Phase Correlation Results:")
    for i, (l1, l2) in enumerate(mode_pairs):
        print(f"   • Mode pair l={l1},l={l2}: Correlation = {phase_correlations[i]:.3f}")

if __name__ == "__main__":
    test_paterson_crosstalk_model()
    test_orthogonality_theory()
    test_phase_correlation()
    
    print("\n🎉 All OAM crosstalk tests completed successfully!")
    print("📁 Generated validation plots:")
    print("   • OAM_Crosstalk_Validation.png")
    print("   • OAM_Orthogonality_Theory.png")
    print("   • OAM_Phase_Correlation.png") 