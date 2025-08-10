#!/usr/bin/env python3
"""
Distance Optimization Training Script

This script trains an RL agent using the distance-optimized environment
to improve performance through intelligent distance-aware mode selection.
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional

# Ensure project root is on sys.path before importing project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from models.agent import Agent
from environment.distance_optimized_env import DistanceOptimizedEnv
from utils.config_utils import load_config, merge_configs
from utils.path_utils import create_results_dir


def load_distance_optimization_config() -> Dict[str, Any]:
    """Load distance optimization configuration."""
    base_config = load_config(os.path.join(PROJECT_ROOT, 'config', 'base_config_new.yaml'))
    distance_config = load_config(os.path.join(PROJECT_ROOT, 'config', 'distance_optimization_config.yaml'))
    
    # Merge configurations
    config = merge_configs(base_config, distance_config)
    
    return config


def train_distance_optimization_agent(config: Dict[str, Any], 
                                    results_dir: str,
                                    num_episodes: int = 1000) -> Dict[str, Any]:
    """
    Train an RL agent with distance optimization.
    
    Args:
        config: Configuration dictionary
        results_dir: Directory to save results
        num_episodes: Number of training episodes
        
    Returns:
        Training metrics and statistics
    """
    print("🚀 Starting Distance Optimization Training")
    print("=" * 50)
    
    # Initialize environment
    env = DistanceOptimizedEnv(config)
    
    # Initialize agent
    agent = Agent(
        state_dim=8,  # OAM environment state dimension
        action_dim=3,  # STAY, UP, DOWN
        hidden_layers=config.get('rl_base', {}).get('network', {}).get('hidden_layers', [128, 128]),
        learning_rate=config.get('training', {}).get('learning_rate', 1e-4),
        gamma=config.get('training', {}).get('gamma', 0.99),
        buffer_capacity=config.get('rl_base', {}).get('replay_buffer', {}).get('capacity', 50000),
        batch_size=config.get('training', {}).get('batch_size', 128),
        target_update_freq=config.get('training', {}).get('target_update_freq', 10)
    )
    
    # Training parameters
    exploration_config = config.get('rl_base', {}).get('exploration', {})
    epsilon_start = exploration_config.get('epsilon_start', 1.0)
    epsilon_end = exploration_config.get('epsilon_end', 0.01)
    epsilon_decay = exploration_config.get('epsilon_decay', 0.99)
    
    # Training tracking
    episode_rewards = []
    episode_throughputs = []
    episode_distances = []
    episode_optimization_scores = []
    distance_optimization_stats = []
    
    print(f"📊 Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset environment
        state, info = env.reset()
        
        episode_reward = 0.0
        episode_throughput = 0.0
        episode_distance = 0.0
        episode_optimization_score = 0.0
        steps = 0
        
        # Calculate epsilon for this episode
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        
        while True:
            # Choose action
            action = agent.choose_action(state, epsilon)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Learn
            if agent.replay_buffer.is_ready(agent.batch_size):
                agent.learn()
            
            # Update state and tracking
            state = next_state
            episode_reward += reward
            episode_throughput += info.get('throughput', 0.0)
            episode_distance += np.linalg.norm(info.get('position', [0, 0, 0]))
            
            # Track optimization scores
            if 'optimization_scores' in info:
                episode_optimization_score += info['optimization_scores'].get('optimization_score', 0.0)
            
            steps += 1
            
            if done or truncated:
                break
        
        # End episode
        agent.end_episode()
        
        # Calculate averages
        avg_throughput = episode_throughput / max(steps, 1)
        avg_distance = episode_distance / max(steps, 1)
        avg_optimization_score = episode_optimization_score / max(steps, 1)
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_throughputs.append(avg_throughput)
        episode_distances.append(avg_distance)
        episode_optimization_scores.append(avg_optimization_score)
        
        # Get distance optimization stats
        if episode % 10 == 0:
            stats = env.get_distance_optimization_stats()
            distance_optimization_stats.append({
                'episode': episode,
                **stats
            })
        
        # Print progress
        if episode % 100 == 0:
            throughput_mbps = avg_throughput/1e6
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Throughput: {throughput_mbps:.1f} Mbps")
            print(f"  Distance: {avg_distance:.1f}m")
            print(f"  Opt Score: {avg_optimization_score:.3f}")
            print()
    
    # Save models
    models_dir = os.path.join(results_dir, 'models')
    agent.save_models(models_dir)
    
    # Prepare training results
    training_results = {
        'episode_rewards': episode_rewards,
        'episode_throughputs': episode_throughputs,
        'episode_distances': episode_distances,
        'episode_optimization_scores': episode_optimization_scores,
        'distance_optimization_stats': distance_optimization_stats,
        'final_distance_optimization_stats': env.get_distance_optimization_stats(),
        'distance_category_performance': env.get_distance_category_performance()
    }
    
    return training_results


def plot_distance_optimization_results(results: Dict[str, Any], results_dir: str):
    """Plot distance optimization training results."""
    print("📈 Generating distance optimization plots...")
    
    # Create plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward curve
    axes[0, 0].plot(results['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Throughput curve
    axes[0, 1].plot(np.array(results['episode_throughputs']) / 1e6)
    axes[0, 1].set_title('Average Throughput')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Throughput (Mbps)')
    axes[0, 1].grid(True)
    
    # Distance curve
    axes[1, 0].plot(results['episode_distances'])
    axes[1, 0].set_title('Average Distance')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Distance (m)')
    axes[1, 0].grid(True)
    
    # Optimization score curve
    axes[1, 1].plot(results['episode_optimization_scores'])
    axes[1, 1].set_title('Average Optimization Score')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Optimization Score')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distance_optimization_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Distance optimization statistics
    if results['distance_optimization_stats']:
        stats = results['distance_optimization_stats']
        episodes = [s['episode'] for s in stats]
        success_rates = [s['success_rate'] for s in stats]
        optimization_scores = [s['average_optimization_score'] for s in stats]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Success rate
        axes[0].plot(episodes, success_rates)
        axes[0].set_title('Distance Optimization Success Rate')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Success Rate')
        axes[0].grid(True)
        
        # Average optimization score
        axes[1].plot(episodes, optimization_scores)
        axes[1].set_title('Average Optimization Score')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Optimization Score')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'distance_optimization_stats.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Distance category performance
    category_performance = results['distance_category_performance']
    if category_performance:
        categories = list(category_performance.keys())
        throughputs = [category_performance[cat]['avg_throughput'] / 1e6 for cat in categories]
        distances = [category_performance[cat]['avg_distance'] for cat in categories]
        optimization_scores = [category_performance[cat]['avg_optimization_score'] for cat in categories]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Throughput by category
        axes[0].bar(categories, throughputs)
        axes[0].set_title('Average Throughput by Distance Category')
        axes[0].set_ylabel('Throughput (Mbps)')
        axes[0].grid(True)
        
        # Distance by category
        axes[1].bar(categories, distances)
        axes[1].set_title('Average Distance by Category')
        axes[1].set_ylabel('Distance (m)')
        axes[1].grid(True)
        
        # Optimization score by category
        axes[2].bar(categories, optimization_scores)
        axes[2].set_title('Average Optimization Score by Category')
        axes[2].set_ylabel('Optimization Score')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'distance_category_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Plots saved to {plots_dir}")


def save_training_results(results: Dict[str, Any], results_dir: str, config: Dict[str, Any]):
    """Save training results and configuration."""
    print("💾 Saving training results...")
    
    # Save configuration
    config_path = os.path.join(results_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save training metrics
    metrics = {
        'final_stats': results['final_distance_optimization_stats'],
        'distance_category_performance': results['distance_category_performance'],
        'training_summary': {
            'total_episodes': len(results['episode_rewards']),
            'final_avg_reward': np.mean(results['episode_rewards'][-100:]),
            'final_avg_throughput': np.mean(results['episode_throughputs'][-100:]) / 1e6,
            'final_avg_distance': np.mean(results['episode_distances'][-100:]),
            'final_avg_optimization_score': np.mean(results['episode_optimization_scores'][-100:])
        }
    }
    
    metrics_path = os.path.join(results_dir, 'training_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"💾 Results saved to {results_dir}")


def main():
    """Main training function."""
    print("🎯 Distance Optimization Training")
    print("=" * 50)
    
    # Load configuration
    config = load_distance_optimization_config()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = create_results_dir(f"distance_optimization_{timestamp}")
    
    # Save configuration
    config_path = os.path.join(results_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📁 Results will be saved to: {results_dir}")
    print(f"⚙️  Configuration saved to: {config_path}")
    
    # Train agent
    training_results = train_distance_optimization_agent(config, results_dir)
    
    # Generate plots
    plot_distance_optimization_results(training_results, results_dir)
    
    # Save results
    save_training_results(training_results, results_dir, config)
    
    # Print final statistics
    final_stats = training_results['final_distance_optimization_stats']
    print("\n🎉 Training Complete!")
    print("=" * 50)
    print(f"📊 Final Distance Optimization Statistics:")
    print(f"   Total Optimizations: {final_stats['total_optimizations']}")
    print(f"   Success Rate: {final_stats['success_rate']:.3f}")
    print(f"   Average Optimization Score: {final_stats['average_optimization_score']:.3f}")
    
    category_performance = training_results['distance_category_performance']
    if category_performance:
        print(f"\n📏 Distance Category Performance:")
        for category, stats in category_performance.items():
            throughput_mbps = stats['avg_throughput']/1e6
            distance_m = stats['avg_distance']
            score = stats['avg_optimization_score']
            print(f"   {category.capitalize()}:")
            print(f"     Throughput: {throughput_mbps:.1f} Mbps")
            print(f"     Distance: {distance_m:.1f}m")
            print(f"     Score: {score:.3f}")
    
    print(f"\n📁 All results saved to: {results_dir}")


if __name__ == "__main__":
    main() 