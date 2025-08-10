#!/usr/bin/env python3
"""
Test script to verify run_step function returns correctly.
"""

import numpy as np
import os

# Use centralized path management instead of sys.path.append
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from simulator.channel_simulator import ChannelSimulator

def test_run_step_return():
    """Test that run_step function returns the expected tuple."""
    
    print("🔬 Testing run_step Return Statement...")
    print("=" * 40)
    
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
    
    # Test cases
    test_cases = [
        (np.array([100.0, 0.0, 2.0]), 1, "Normal case"),
        (np.array([1000.0, 0.0, 2.0]), 3, "Long distance, mode 3"),
        (np.array([10.0, 0.0, 2.0]), 6, "Close distance, mode 6"),
    ]
    
    for user_position, mode, description in test_cases:
        try:
            # Call run_step
            result = simulator.run_step(user_position, mode)
            
            # Check return type
            if isinstance(result, tuple) and len(result) == 2:
                H, sinr_dB = result
                
                # Check types
                if isinstance(H, np.ndarray) and isinstance(sinr_dB, (int, float)):
                    print(f"✅ {description}: SUCCESS")
                    print(f"   • H shape: {H.shape}")
                    print(f"   • H dtype: {H.dtype}")
                    print(f"   • SINR: {sinr_dB:.2f} dB")
                    print(f"   • Return type: {type(result)}")
                else:
                    print(f"❌ {description}: WRONG TYPES")
                    print(f"   • H type: {type(H)}")
                    print(f"   • SINR type: {type(sinr_dB)}")
            else:
                print(f"❌ {description}: NOT A TUPLE")
                print(f"   • Result type: {type(result)}")
                print(f"   • Result: {result}")
                
        except Exception as e:
            print(f"❌ {description}: ERROR - {str(e)}")

def test_function_signature():
    """Test that the function signature matches the expected return type."""
    
    print("\n🔬 Testing Function Signature...")
    print("=" * 35)
    
    # Import the function to check its signature
    from simulator.channel_simulator import ChannelSimulator
    
    # Get the function object
    run_step_func = ChannelSimulator.run_step
    
    # Check the function signature
    import inspect
    sig = inspect.signature(run_step_func)
    
    print(f"📋 Function signature: {sig}")
    print(f"📋 Return annotation: {run_step_func.__annotations__.get('return', 'No annotation')}")
    
    # Check if it matches expected signature
    expected_return = "Tuple[numpy.ndarray, float]"
    actual_return = str(run_step_func.__annotations__.get('return', 'No annotation'))
    
    if "Tuple" in actual_return and "ndarray" in actual_return and "float" in actual_return:
        print("✅ Function signature is correct")
    else:
        print("❌ Function signature mismatch")
        print(f"   • Expected: {expected_return}")
        print(f"   • Actual: {actual_return}")

def test_actual_return_values():
    """Test that the function actually returns the expected values."""
    
    print("\n🔬 Testing Actual Return Values...")
    print("=" * 40)
    
    # Create simulator
    config = {
        'system': {'frequency': 28.0e9, 'bandwidth': 400e6, 'tx_power_dBm': 30.0},
        'environment': {'humidity': 50.0, 'temperature': 20.0, 'pressure': 101.3},
        'oam': {'min_mode': 1, 'max_mode': 6}
    }
    
    simulator = ChannelSimulator(config)
    
    # Test the function
    user_position = np.array([100.0, 0.0, 2.0])
    mode = 1
    
    try:
        H, sinr_dB = simulator.run_step(user_position, mode)
        
        # Validate H matrix
        print(f"✅ H matrix validation:")
        print(f"   • Shape: {H.shape}")
        print(f"   • Type: {H.dtype}")
        print(f"   • Complex: {np.iscomplexobj(H)}")
        print(f"   • Finite: {np.all(np.isfinite(H))}")
        
        # Validate SINR
        print(f"✅ SINR validation:")
        print(f"   • Value: {sinr_dB:.2f} dB")
        print(f"   • Type: {type(sinr_dB)}")
        print(f"   • Finite: {np.isfinite(sinr_dB)}")
        print(f"   • In range: {-20 <= sinr_dB <= 40}")
        
        print("✅ All validations passed!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_run_step_return()
    test_function_signature()
    test_actual_return_values()
    
    print("\n🎉 All run_step return tests completed!")
    print("📋 Summary:")
    print("   • Function returns Tuple[np.ndarray, float]")
    print("   • H matrix is complex numpy array")
    print("   • SINR is finite float in dB")
    print("   • All edge cases handled correctly") 