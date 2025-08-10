#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from environment.distance_optimized_env import DistanceOptimizedEnv
from utils.config_utils import load_config
import numpy as np

def test_distance_performance():
    """Test distance category performance during training."""
    print("Testing Distance Category Performance")
    print("=" * 50)
    
    # Load configuration
    config = load_config('config/base_config_new.yaml')
    
    # Create environment
    env = DistanceOptimizedEnv(config)
    
    print("Running simulation for 1000 steps with frequent resets...")
    
    # Run simulation
    state, info = env.reset()
    distances = []
    categories = []
    
    for step in range(1000):
        action = 0  # STAY action
        next_state, reward, done, truncated, info = env.step(action)
        
        distance = np.linalg.norm(info.get('position', [0, 0, 0]))
        category = env.distance_optimizer.get_distance_category(distance)
        
        distances.append(distance)
        categories.append(category)
        
        # Force reset every 50 steps to get more variety
        if step % 50 == 0 and step > 0:
            state, info = env.reset()
        elif done or truncated:
            state, info = env.reset()
        else:
            state = next_state
        
        # Print some debug info every 200 steps
        if step % 200 == 0:
            print(f"Step {step}: distance={distance:.1f}m, category={category}")
    
    print(f"\nDistance statistics:")
    print(f"  Min: {min(distances):.1f}m")
    print(f"  Max: {max(distances):.1f}m")
    print(f"  Mean: {np.mean(distances):.1f}m")
    print(f"  Std: {np.std(distances):.1f}m")
    
    print(f"\nCategory distribution:")
    from collections import Counter
    category_counts = Counter(categories)
    for category, count in category_counts.items():
        percentage = (count / len(categories)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print(f"\nDistance ranges by category:")
    for category in ['near', 'medium', 'far']:
        category_distances = [d for d, c in zip(distances, categories) if c == category]
        if category_distances:
            print(f"  {category}: {min(category_distances):.1f}m - {max(category_distances):.1f}m")
    
    # Get performance stats
    performance = env.get_distance_category_performance()
    print(f"\nPerformance by category:")
    for category, stats in performance.items():
        print(f"  {category}:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg Throughput: {stats['avg_throughput']/1e6:.1f} Mbps")
        print(f"    Avg Distance: {stats['avg_distance']:.1f}m")
        print(f"    Avg Optimization Score: {stats['avg_optimization_score']:.3f}")

if __name__ == "__main__":
    test_distance_performance() 