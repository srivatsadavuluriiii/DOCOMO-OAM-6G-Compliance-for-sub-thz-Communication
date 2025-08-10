#!/usr/bin/env python3

import numpy as np
import sys
import os
sys.path.append('.')

from environment.mobility_model import MobilityModel

def debug_position_generation():
    """Debug position generation logic."""
    print("Debugging Position Generation")
    print("=" * 40)
    
    # Create mobility model
    mob_model = MobilityModel()
    
    print("Testing position generation...")
    for i in range(10):
        pos = mob_model.generate_random_position()
        distance = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        print(f"Position {i+1}: {pos}, Distance: {distance:.1f}m")
    
    print("\nTesting with different distance targets:")
    for target_dist in [75, 125, 175, 225]:
        # Generate position with specific distance
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        x = target_dist * np.sin(phi) * np.cos(theta)
        y = target_dist * np.sin(phi) * np.sin(theta)
        z = target_dist * np.cos(phi)
        
        # Clip to area bounds
        x = np.clip(x, 0, 500)
        y = np.clip(y, 0, 500)
        
        actual_distance = np.sqrt(x**2 + y**2 + z**2)
        print(f"Target: {target_dist}m, Actual: {actual_distance:.1f}m, Position: [{x:.1f}, {y:.1f}, {z:.1f}]")

if __name__ == "__main__":
    debug_position_generation() 