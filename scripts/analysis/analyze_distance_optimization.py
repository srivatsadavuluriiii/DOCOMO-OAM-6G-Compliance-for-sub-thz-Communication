#!/usr/bin/env python3
"""
Distance Optimization Analysis Script

This script analyzes the performance of distance optimization strategies
and compares them against baseline approaches.
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from environment.oam_env import OAM_Env
from environment.distance_optimized_env import DistanceOptimizedEnv
from environment.distance_optimizer import DistanceOptimizer, DistanceOptimizationConfig
from utils.config_utils import load_config, merge_configs


def run_baseline_analysis(config: Dict[str, Any], num_episodes: int = 100) -> Dict[str, Any]:
    """
    Run baseline analysis without distance optimization.
    
    Args:
        config: Configuration dictionary
        num_episodes: Number of episodes to run
        
    Returns:
        Baseline performance metrics
    """
    print("🔍 Running Baseline Analysis...")
    
    env = OAM_Env(config)
    
    baseline_results = {
        'episode_rewards': [],
        'episode_throughputs': [],
        'episode_distances': [],
        'episode_handovers': [],
        'distance_performance': {'near': [], 'medium': [], 'far': []}
    }
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_throughput = 0.0
        episode_distance = 0.0
        episode_handovers = 0
        steps = 0
        
        while True:
            # Use random actions for baseline
            action = np.random.randint(0, 3)
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_throughput += info.get('throughput', 0.0)
            episode_distance += np.linalg.norm(info.get('position', [0, 0, 0]))
            episode_handovers += info.get('handovers', 0)
            
            # Categorize by distance
            distance = np.linalg.norm(info.get('position', [0, 0, 0]))
            if distance <= 50:
                category = 'near'
            elif distance <= 150:
                category = 'medium'
            else:
                category = 'far'
            
            baseline_results['distance_performance'][category].append({
                'throughput': info.get('throughput', 0.0),
                'distance': distance,
                'mode': info.get('mode', 1)
            })
            
            state = next_state
            steps += 1
            
            if done or truncated:
                break
        
        # Store episode metrics
        baseline_results['episode_rewards'].append(episode_reward)
        baseline_results['episode_throughputs'].append(episode_throughput / max(steps, 1))
        baseline_results['episode_distances'].append(episode_distance / max(steps, 1))
        baseline_results['episode_handovers'].append(episode_handovers)
    
    return baseline_results


def run_distance_optimization_analysis(config: Dict[str, Any], num_episodes: int = 100) -> Dict[str, Any]:
    """
    Run analysis with distance optimization.
    
    Args:
        config: Configuration dictionary
        num_episodes: Number of episodes to run
        
    Returns:
        Distance optimization performance metrics
    """
    print("🎯 Running Distance Optimization Analysis...")
    
    env = DistanceOptimizedEnv(config)
    
    optimization_results = {
        'episode_rewards': [],
        'episode_throughputs': [],
        'episode_distances': [],
        'episode_handovers': [],
        'episode_optimization_scores': [],
        'distance_performance': {'near': [], 'medium': [], 'far': []},
        'optimization_stats': []
    }
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_throughput = 0.0
        episode_distance = 0.0
        episode_handovers = 0
        episode_optimization_score = 0.0
        steps = 0
        
        while True:
            # Use random actions but with distance optimization
            action = np.random.randint(0, 3)
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_throughput += info.get('throughput', 0.0)
            episode_distance += np.linalg.norm(info.get('position', [0, 0, 0]))
            episode_handovers += info.get('handovers', 0)
            
            # Track optimization scores
            if 'optimization_scores' in info:
                episode_optimization_score += info['optimization_scores'].get('optimization_score', 0.0)
            
            # Categorize by distance
            distance = np.linalg.norm(info.get('position', [0, 0, 0]))
            category = env.distance_optimizer.get_distance_category(distance)
            
            optimization_results['distance_performance'][category].append({
                'throughput': info.get('throughput', 0.0),
                'distance': distance,
                'mode': info.get('mode', 1),
                'optimization_score': info.get('optimization_scores', {}).get('optimization_score', 0.0)
            })
            
            state = next_state
            steps += 1
            
            if done or truncated:
                break
        
        # Store episode metrics
        optimization_results['episode_rewards'].append(episode_reward)
        optimization_results['episode_throughputs'].append(episode_throughput / max(steps, 1))
        optimization_results['episode_distances'].append(episode_distance / max(steps, 1))
        optimization_results['episode_handovers'].append(episode_handovers)
        optimization_results['episode_optimization_scores'].append(episode_optimization_score / max(steps, 1))
        
        # Get optimization stats
        if episode % 10 == 0:
            stats = env.get_distance_optimization_stats()
            optimization_results['optimization_stats'].append({
                'episode': episode,
                **stats
            })
    
    return optimization_results


def compare_performance(baseline_results: Dict[str, Any], 
                       optimization_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare baseline vs distance optimization performance.
    
    Args:
        baseline_results: Baseline performance metrics
        optimization_results: Distance optimization performance metrics
        
    Returns:
        Comparison metrics
    """
    print("📊 Comparing Performance...")
    
    comparison = {}
    
    # Overall performance comparison
    baseline_avg_reward = np.mean(baseline_results['episode_rewards'])
    optimization_avg_reward = np.mean(optimization_results['episode_rewards'])
    reward_improvement = ((optimization_avg_reward - baseline_avg_reward) / baseline_avg_reward) * 100
    
    baseline_avg_throughput = np.mean(baseline_results['episode_throughputs'])
    optimization_avg_throughput = np.mean(optimization_results['episode_throughputs'])
    throughput_improvement = ((optimization_avg_throughput - baseline_avg_throughput) / baseline_avg_throughput) * 100
    
    baseline_avg_handovers = np.mean(baseline_results['episode_handovers'])
    optimization_avg_handovers = np.mean(optimization_results['episode_handovers'])
    handover_change = ((optimization_avg_handovers - baseline_avg_handovers) / baseline_avg_handovers) * 100
    
    comparison['overall'] = {
        'baseline_avg_reward': baseline_avg_reward,
        'optimization_avg_reward': optimization_avg_reward,
        'reward_improvement_percent': reward_improvement,
        'baseline_avg_throughput': baseline_avg_throughput,
        'optimization_avg_throughput': optimization_avg_throughput,
        'throughput_improvement_percent': throughput_improvement,
        'baseline_avg_handovers': baseline_avg_handovers,
        'optimization_avg_handovers': optimization_avg_handovers,
        'handover_change_percent': handover_change
    }
    
    # Distance category comparison
    comparison['distance_categories'] = {}
    
    for category in ['near', 'medium', 'far']:
        baseline_data = baseline_results['distance_performance'][category]
        optimization_data = optimization_results['distance_performance'][category]
        
        if baseline_data and optimization_data:
            baseline_throughput = np.mean([d['throughput'] for d in baseline_data])
            optimization_throughput = np.mean([d['throughput'] for d in optimization_data])
            throughput_improvement = ((optimization_throughput - baseline_throughput) / baseline_throughput) * 100
            
            baseline_distance = np.mean([d['distance'] for d in baseline_data])
            optimization_distance = np.mean([d['distance'] for d in optimization_data])
            
            comparison['distance_categories'][category] = {
                'baseline_throughput': baseline_throughput,
                'optimization_throughput': optimization_throughput,
                'throughput_improvement_percent': throughput_improvement,
                'baseline_avg_distance': baseline_distance,
                'optimization_avg_distance': optimization_distance,
                'baseline_samples': len(baseline_data),
                'optimization_samples': len(optimization_data)
            }
    
    return comparison


def plot_comparison_results(baseline_results: Dict[str, Any], 
                          optimization_results: Dict[str, Any],
                          comparison: Dict[str, Any],
                          results_dir: str):
    """Plot comparison results."""
    print("📈 Generating comparison plots...")
    
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Overall performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards comparison
    axes[0, 0].plot(baseline_results['episode_rewards'], alpha=0.7, label='Baseline')
    axes[0, 0].plot(optimization_results['episode_rewards'], alpha=0.7, label='Distance Optimization')
    axes[0, 0].set_title('Episode Rewards Comparison')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Throughput comparison
    baseline_throughputs = np.array(baseline_results['episode_throughputs']) / 1e6
    optimization_throughputs = np.array(optimization_results['episode_throughputs']) / 1e6
    axes[0, 1].plot(baseline_throughputs, alpha=0.7, label='Baseline')
    axes[0, 1].plot(optimization_throughputs, alpha=0.7, label='Distance Optimization')
    axes[0, 1].set_title('Episode Throughput Comparison')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Throughput (Mbps)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Handovers comparison
    axes[1, 0].plot(baseline_results['episode_handovers'], alpha=0.7, label='Baseline')
    axes[1, 0].plot(optimization_results['episode_handovers'], alpha=0.7, label='Distance Optimization')
    axes[1, 0].set_title('Episode Handovers Comparison')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Handovers')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Optimization scores
    if 'episode_optimization_scores' in optimization_results:
        axes[1, 1].plot(optimization_results['episode_optimization_scores'])
        axes[1, 1].set_title('Distance Optimization Scores')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Optimization Score')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Distance category performance
    categories = ['near', 'medium', 'far']
    baseline_throughputs = []
    optimization_throughputs = []
    
    for category in categories:
        baseline_data = baseline_results['distance_performance'][category]
        optimization_data = optimization_results['distance_performance'][category]
        
        baseline_throughputs.append(np.mean([d['throughput'] for d in baseline_data]) / 1e6 if baseline_data else 0)
        optimization_throughputs.append(np.mean([d['throughput'] for d in optimization_data]) / 1e6 if optimization_data else 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Throughput by category
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, baseline_throughputs, width, label='Baseline', alpha=0.7)
    axes[0].bar(x + width/2, optimization_throughputs, width, label='Distance Optimization', alpha=0.7)
    axes[0].set_title('Throughput by Distance Category')
    axes[0].set_xlabel('Distance Category')
    axes[0].set_ylabel('Throughput (Mbps)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(True)
    
    # Improvement percentages
    improvements = []
    for i, category in enumerate(categories):
        if baseline_throughputs[i] > 0:
            improvement = ((optimization_throughputs[i] - baseline_throughputs[i]) / baseline_throughputs[i]) * 100
            improvements.append(improvement)
        else:
            improvements.append(0)
    
    axes[1].bar(categories, improvements, color='green' if np.mean(improvements) > 0 else 'red')
    axes[1].set_title('Throughput Improvement by Category')
    axes[1].set_xlabel('Distance Category')
    axes[1].set_ylabel('Improvement (%)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distance_category_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to {plots_dir}")


def save_analysis_results(baseline_results: Dict[str, Any], 
                         optimization_results: Dict[str, Any],
                         comparison: Dict[str, Any],
                         results_dir: str):
    """Save analysis results."""
    print("💾 Saving analysis results...")
    
    # Save comparison metrics
    import json
    comparison_path = os.path.join(results_dir, 'comparison_metrics.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    # Save detailed results
    detailed_results = {
        'baseline_results': baseline_results,
        'optimization_results': optimization_results,
        'comparison': comparison
    }
    
    detailed_path = os.path.join(results_dir, 'detailed_results.json')
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"💾 Results saved to {results_dir}")


def print_analysis_summary(comparison: Dict[str, Any]):
    """Print analysis summary."""
    print("\n📊 Distance Optimization Analysis Summary")
    print("=" * 50)
    
    overall = comparison['overall']
    
    print(f"🎯 Overall Performance:")
    print(f"   Reward Improvement: {overall['reward_improvement_percent']:.2f}%")
    print(f"   Throughput Improvement: {overall['throughput_improvement_percent']:.2f}%")
    print(f"   Handover Change: {overall['handover_change_percent']:.2f}%")
    
    print(f"\n📏 Distance Category Performance:")
    for category, stats in comparison['distance_categories'].items():
        print(f"   {category.capitalize()}:")
        print(f"     Throughput Improvement: {stats['throughput_improvement_percent']:.2f}%")
        print(f"     Baseline Throughput: {stats['baseline_throughput']/1e6:.1f} Mbps")
        print(f"     Optimization Throughput: {stats['optimization_throughput']/1e6:.1f} Mbps")
    
    # Determine if distance optimization is beneficial
    avg_improvement = np.mean([stats['throughput_improvement_percent'] 
                              for stats in comparison['distance_categories'].values()])
    
    print(f"\n💡 Analysis Conclusion:")
    if avg_improvement > 5:
        print(f"   ✅ Distance optimization provides significant improvement ({avg_improvement:.1f}% average)")
    elif avg_improvement > 0:
        print(f"   ⚠️  Distance optimization provides modest improvement ({avg_improvement:.1f}% average)")
    else:
        print(f"   ❌ Distance optimization does not provide improvement ({avg_improvement:.1f}% average)")


def main():
    """Main analysis function."""
    print("🔍 Distance Optimization Analysis")
    print("=" * 50)
    
    # Load configuration (cwd-agnostic)
    base_config = load_config(os.path.join(PROJECT_ROOT, 'config', 'base_config_new.yaml'))
    distance_config = load_config(os.path.join(PROJECT_ROOT, 'config', 'distance_optimization_config.yaml'))
    config = merge_configs(base_config, distance_config)
    
    # Create results directory under project root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(PROJECT_ROOT, f"results/distance_optimization_analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"📁 Results will be saved to: {results_dir}")
    
    # Run analyses
    baseline_results = run_baseline_analysis(config, num_episodes=100)
    optimization_results = run_distance_optimization_analysis(config, num_episodes=100)
    
    # Compare performance
    comparison = compare_performance(baseline_results, optimization_results)
    
    # Generate plots
    plot_comparison_results(baseline_results, optimization_results, comparison, results_dir)
    
    # Save results
    save_analysis_results(baseline_results, optimization_results, comparison, results_dir)
    
    # Print summary
    print_analysis_summary(comparison)
    
    print(f"\n📁 All results saved to: {results_dir}")


if __name__ == "__main__":
    main() 