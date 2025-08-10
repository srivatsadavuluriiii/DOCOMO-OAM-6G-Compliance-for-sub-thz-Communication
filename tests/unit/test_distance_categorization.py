#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from environment.distance_optimizer import DistanceOptimizer, DistanceOptimizationConfig

def test_distance_categorization():
    """Test distance categorization logic."""
    print("Testing Distance Categorization")
    print("=" * 40)
    
    # Create optimizer with default config
    config = DistanceOptimizationConfig()
    optimizer = DistanceOptimizer(config)
    
    print("Initial thresholds:")
    print(f"  Near: {optimizer.adaptive_near_threshold}m")
    print(f"  Medium: {optimizer.adaptive_medium_threshold}m")
    print(f"  Far: {optimizer.adaptive_far_threshold}m")
    print()
    
    print("Testing categorization:")
    test_distances = [25, 75, 125, 175, 250, 108]  # 108 is the average from training
    for dist in test_distances:
        category = optimizer.get_distance_category(dist)
        print(f"  {dist}m -> {category}")
    
    print()
    print("Testing with adaptive thresholds:")
    # Simulate some optimization calls to see if thresholds change
    for i in range(10):
        optimizer.optimize_mode_selection(108.0, 1e8, 4, [1, 2, 3, 4, 5, 6, 7, 8])
    
    print(f"After optimization calls:")
    print(f"  Near: {optimizer.adaptive_near_threshold}m")
    print(f"  Medium: {optimizer.adaptive_medium_threshold}m")
    print(f"  Far: {optimizer.adaptive_far_threshold}m")
    
    # Test categorization again
    print()
    print("Categorization after optimization:")
    for dist in test_distances:
        category = optimizer.get_distance_category(dist)
        print(f"  {dist}m -> {category}")

if __name__ == "__main__":
    test_distance_categorization() 