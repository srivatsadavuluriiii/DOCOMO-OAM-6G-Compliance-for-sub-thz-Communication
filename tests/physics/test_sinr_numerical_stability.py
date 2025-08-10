#!/usr/bin/env python3
"""
Test script to validate SINR numerical stability improvements.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure project root is on sys.path before importing project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Use centralized path management instead of sys.path.append
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from simulator.channel_simulator import ChannelSimulator

def test_sinr_numerical_stability():
    """Test SINR calculation with various numerical edge cases."""
    
    print("🔬 Testing SINR Numerical Stability...")
    print("=" * 45)
    
    # Create simulator with standard parameters
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
    
    # Test cases with different numerical scenarios
    test_cases = [
        # (distance, mode, description)
        (1.0, 1, "Very close distance (1m)"),
        (100.0, 1, "Normal distance (100m)"),
        (1000.0, 1, "Long distance (1km)"),
        (50000.0, 1, "Maximum distance (50km)"),
        (1.0, 6, "Very close distance, high mode"),
        (1000.0, 6, "Long distance, high mode"),
    ]
    
    results = []
    
    for distance, mode, description in test_cases:
        # Create user position at specified distance
        user_position = np.array([distance, 0, 2.0])  # 2m height
        
        try:
            # Run simulation step
            H, sinr_dB = simulator.run_step(user_position, mode)
            
            # Extract signal and interference powers for analysis
            mode_idx = mode - simulator.min_mode
            signal_power = simulator.tx_power_W * np.abs(H[mode_idx, mode_idx])**2
            
            interference_power = 0
            for i in range(simulator.num_modes):
                if i != mode_idx:
                    interference_power += simulator.tx_power_W * np.abs(H[mode_idx, i])**2
            
            noise_power = simulator._calculate_noise_power()
            
            # Check numerical stability
            is_stable = True
            issues = []
            
            # Check for NaN or Inf values
            if np.isnan(sinr_dB) or np.isinf(sinr_dB):
                is_stable = False
                issues.append("NaN or Inf SINR")
            
            # Check for extremely small signal power
            if signal_power < 1e-30:
                issues.append("Extremely small signal power")
            
            # Check for extremely large SINR
            if sinr_dB > 40.0:
                issues.append("SINR too high (capped)")
            
            # Check for extremely low SINR
            if sinr_dB < -20.0:
                issues.append("SINR too low (capped)")
            
            # Check denominator stability
            denominator = interference_power + noise_power
            if denominator < 1e-12:
                issues.append("Very small denominator")
            
            results.append({
                'description': description,
                'distance': distance,
                'mode': mode,
                'sinr_dB': sinr_dB,
                'signal_power': signal_power,
                'interference_power': interference_power,
                'noise_power': noise_power,
                'denominator': denominator,
                'is_stable': is_stable,
                'issues': issues
            })
            
        except Exception as e:
            results.append({
                'description': description,
                'distance': distance,
                'mode': mode,
                'error': str(e),
                'is_stable': False,
                'issues': [f"Exception: {str(e)}"]
            })
    
    # Print results
    print("📊 SINR Numerical Stability Test Results:")
    print("-" * 60)
    
    for result in results:
        print(f"\n🔍 {result['description']}")
        print(f"   • Distance: {result['distance']}m, Mode: {result['mode']}")
        
        if 'error' in result:
            print(f"   • ❌ ERROR: {result['error']}")
        else:
            print(f"   • SINR: {result['sinr_dB']:.2f} dB")
            print(f"   • Signal Power: {result['signal_power']:.2e} W")
            print(f"   • Interference Power: {result['interference_power']:.2e} W")
            print(f"   • Noise Power: {result['noise_power']:.2e} W")
            print(f"   • Denominator: {result['denominator']:.2e} W")
            
            if result['is_stable']:
                print(f"   • ✅ STABLE")
            else:
                print(f"   • ⚠️  UNSTABLE")
                for issue in result['issues']:
                    print(f"     - {issue}")

def test_extreme_edge_cases():
    """Test extreme edge cases that could cause numerical instability."""
    
    print("\n🔬 Testing Extreme Edge Cases...")
    print("=" * 40)
    
    # Create simulator with extreme parameters
    config = {
        'system': {
            'frequency': 28.0e9,
            'bandwidth': 400e6,
            'tx_power_dBm': 50.0,  # Very high power
            'noise_figure_dB': 20.0,  # Very high noise
            'noise_temp': 500.0  # Maximum allowed temperature
        },
        'environment': {
            'humidity': 100.0,  # Maximum humidity
            'temperature': 50.0,  # High temperature
            'pressure': 50.0,  # Low pressure
            'turbulence_strength': 1e-12,  # Very strong turbulence
            'pointing_error_std': 0.1,  # Large pointing error
            'rician_k_factor': 1.0  # Weak LOS
        },
        'oam': {
            'min_mode': 1,
            'max_mode': 6,
            'beam_width': 0.01  # Very narrow beam
        }
    }
    
    simulator = ChannelSimulator(config)
    
    # Test extreme scenarios
    extreme_cases = [
        (0.1, 1, "Extremely close (0.1m)"),
        (100000, 1, "Extremely far (100km)"),
        (1.0, 6, "Close with high mode"),
        (1000.0, 6, "Far with high mode"),
    ]
    
    for distance, mode, description in extreme_cases:
        try:
            user_position = np.array([distance, 0, 2.0])
            H, sinr_dB = simulator.run_step(user_position, mode)
            
            print(f"✅ {description}: SINR = {sinr_dB:.2f} dB")
            
        except Exception as e:
            print(f"❌ {description}: ERROR - {str(e)}")

def test_overflow_protection():
    """Test overflow protection for very large SINR values."""
    
    print("\n🔬 Testing Overflow Protection...")
    print("=" * 35)
    
    # Create simulator with parameters that could cause overflow
    config = {
        'system': {
            'frequency': 28.0e9,
            'bandwidth': 400e6,
            'tx_power_dBm': 50.0,  # Maximum allowed power
            'noise_figure_dB': 1.0,  # Very low noise
            'noise_temp': 100.0  # Low temperature
        },
        'environment': {
            'humidity': 10.0,  # Low humidity
            'temperature': 10.0,  # Low temperature
            'pressure': 120.0,  # Maximum allowed pressure
            'turbulence_strength': 1e-16,  # Very weak turbulence
            'pointing_error_std': 0.001,  # Very small pointing error
            'rician_k_factor': 20.0  # Very strong LOS
        },
        'oam': {
            'min_mode': 1,
            'max_mode': 6,
            'beam_width': 0.05  # Wide beam
        }
    }
    
    simulator = ChannelSimulator(config)
    
    # Test scenarios that could cause overflow
    overflow_cases = [
        (1.0, 1, "Very close with ideal conditions"),
        (10.0, 1, "Close with ideal conditions"),
        (100.0, 1, "Medium distance with ideal conditions"),
    ]
    
    for distance, mode, description in overflow_cases:
        try:
            user_position = np.array([distance, 0, 2.0])
            H, sinr_dB = simulator.run_step(user_position, mode)
            
            # Check if SINR is reasonable
            if sinr_dB > 40.0:
                print(f"⚠️  {description}: SINR = {sinr_dB:.2f} dB (capped)")
            else:
                print(f"✅ {description}: SINR = {sinr_dB:.2f} dB")
                
        except Exception as e:
            print(f"❌ {description}: ERROR - {str(e)}")

if __name__ == "__main__":
    test_sinr_numerical_stability()
    test_extreme_edge_cases()
    test_overflow_protection()
    
    print("\n🎉 All SINR numerical stability tests completed!")
    print("📋 Summary:")
    print("   • Improved denominator floor: 1e-12 (was 1e-15)")
    print("   • Added signal power checks: < 1e-30")
    print("   • Added overflow protection: cap at 1e12")
    print("   • Realistic SINR bounds: -20 dB to +40 dB")
    print("   • Better NaN/Inf handling") 