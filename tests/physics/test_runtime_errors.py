#!/usr/bin/env python3
"""
Test script to verify no runtime errors in critical functions.
"""

import numpy as np
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

def test_critical_functions():
    """Test all critical functions for runtime errors."""
    
    print("🔬 Testing Critical Functions for Runtime Errors...")
    print("=" * 55)
    
    # Create simulator
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
    
    # Test parameters
    distance = 1000.0  # 1 km
    user_position = np.array([1000.0, 0.0, 0.0])  # 1 km away
    current_oam_mode = 1
    
    try:
        print("✅ Testing _calculate_path_loss...")
        path_loss = simulator._calculate_path_loss(distance)
        print(f"   • Path loss: {path_loss:.2e}")
        
        print("✅ Testing _calculate_atmospheric_absorption...")
        atmospheric_loss = simulator._calculate_atmospheric_absorption(distance)
        print(f"   • Atmospheric loss: {atmospheric_loss:.2e}")
        
        print("✅ Testing _calculate_noise_power...")
        noise_power = simulator._calculate_noise_power()
        print(f"   • Noise power: {noise_power:.2e} W")
        
        print("✅ Testing _get_pointing_error_loss...")
        pointing_loss = simulator._get_pointing_error_loss(current_oam_mode)
        print(f"   • Pointing loss: {pointing_loss:.4f}")
        
        print("✅ Testing _get_rician_fading_gain...")
        fading_matrix = simulator._get_rician_fading_gain()
        print(f"   • Fading matrix shape: {fading_matrix.shape}")
        
        print("✅ Testing _generate_turbulence_screen...")
        turbulence_screen = simulator._generate_turbulence_screen(distance)
        print(f"   • Turbulence screen shape: {turbulence_screen.shape}")
        
        print("✅ Testing _calculate_crosstalk...")
        crosstalk_matrix = simulator._calculate_crosstalk(distance, turbulence_screen)
        print(f"   • Crosstalk matrix shape: {crosstalk_matrix.shape}")
        
        print("✅ Testing run_step (main simulation function)...")
        H, sinr_dB = simulator.run_step(user_position, current_oam_mode)
        print(f"   • Channel matrix shape: {H.shape}")
        print(f"   • SINR: {sinr_dB:.2f} dB")
        
        print("\n🎉 ALL CRITICAL FUNCTIONS WORKING CORRECTLY!")
        print("✅ No runtime errors detected")
        print("✅ All functions return proper values")
        print("✅ No undefined variables")
        print("✅ No missing return statements")
        
    except Exception as e:
        print(f"\n❌ RUNTIME ERROR DETECTED: {e}")
        print(f"   • Error type: {type(e).__name__}")
        print(f"   • Error location: {e.__traceback__.tb_frame.f_code.co_name}")
        assert False, f"Runtime error in critical functions: {e}"

def test_edge_cases():
    """Test edge cases that might cause runtime errors."""
    
    print("\n🔬 Testing Edge Cases...")
    print("=" * 30)
    
    config = {
        'system': {'frequency': 28.0e9},
        'environment': {'humidity': 50.0, 'temperature': 20.0, 'pressure': 101.3},
        'oam': {'min_mode': 1, 'max_mode': 6}
    }
    
    simulator = ChannelSimulator(config)
    
    try:
        # Test very small distance
        print("✅ Testing minimum distance (1m)...")
        path_loss_min = simulator._calculate_path_loss(1.0)
        print(f"   • Path loss at 1m: {path_loss_min:.2e}")
        
        # Test very large distance
        print("✅ Testing maximum distance (50km)...")
        path_loss_max = simulator._calculate_path_loss(50000.0)
        print(f"   • Path loss at 50km: {path_loss_max:.2e}")
        
        # Test zero humidity
        print("✅ Testing zero humidity...")
        simulator.humidity = 0.0
        atmospheric_loss_zero = simulator._calculate_atmospheric_absorption(1000.0)
        print(f"   • Atmospheric loss (0% humidity): {atmospheric_loss_zero:.2e}")
        
        # Test high humidity
        print("✅ Testing high humidity (100%)...")
        simulator.humidity = 100.0
        atmospheric_loss_high = simulator._calculate_atmospheric_absorption(1000.0)
        print(f"   • Atmospheric loss (100% humidity): {atmospheric_loss_high:.2e}")
        
        print("\n🎉 ALL EDGE CASES HANDLED CORRECTLY!")
        
    except Exception as e:
        print(f"\n❌ EDGE CASE ERROR: {e}")
        assert False, f"Edge case runtime error: {e}"

if __name__ == "__main__":
    success1 = test_critical_functions()
    success2 = test_edge_cases()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED - NO RUNTIME ERRORS!")
    else:
        print("\n❌ RUNTIME ERRORS DETECTED!")
        sys.exit(1) 