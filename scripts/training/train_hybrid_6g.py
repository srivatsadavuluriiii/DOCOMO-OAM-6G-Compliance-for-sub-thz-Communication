#!/usr/bin/env python3
"""
Training script for Hybrid 6G OAM System

This script trains a reinforcement learning agent on the hybrid OAM environment
that supports multiple frequency bands (mmWave, 140 GHz, 300 GHz) with intelligent
band selection and switching.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
from typing import Dict, Any, List

# Ensure project root is on sys.path before importing project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from environment.hybrid_oam_env import HybridOAM_Env
from models.agent import Agent
from utils.config_utils import load_config, merge_configs
from utils.path_utils import create_results_dir
from utils.visualization_unified import plot_training_curves


def load_hybrid_config() -> Dict[str, Any]:
    """Load hybrid 6G configuration."""
    base_config = load_config(os.path.join(PROJECT_ROOT, 'config', 'base_config_new.yaml'))
    hybrid_config = load_config(os.path.join(PROJECT_ROOT, 'config', 'hybrid_6g_config.yaml'))
    config = merge_configs(base_config, hybrid_config)
    return config


def train_hybrid_agent(config: Dict[str, Any], results_dir: str, 
                      num_episodes: int = 2000) -> Dict[str, Any]:
    """
    Train a hybrid 6G agent.
    
    Args:
        config: Configuration dictionary
        results_dir: Directory to save results
        num_episodes: Number of training episodes
        
    Returns:
        Training metrics and statistics
    """
    print("🚀 Starting Hybrid 6G Training")
    print("=" * 50)
    
    # Initialize hybrid environment
    env = HybridOAM_Env(config)
    
    # Initialize agent with updated action space (5 actions)
    agent = Agent(
        state_dim=10,  # Updated state dimension for hybrid environment
        action_dim=5,  # [STAY, UP, DOWN, SWITCH_BAND_UP, SWITCH_BAND_DOWN]
        hidden_layers=config.get('rl_base', {}).get('network', {}).get('hidden_layers', [128, 128]),
        learning_rate=config.get('training', {}).get('learning_rate', 1e-4),
        gamma=config.get('training', {}).get('gamma', 0.99),
        buffer_capacity=config.get('rl_base', {}).get('replay_buffer', {}).get('capacity', 50000),
        batch_size=config.get('training', {}).get('batch_size', 256),
        target_update_freq=config.get('training', {}).get('target_update_freq', 5)
    )
    
    # Training parameters
    exploration_config = config.get('training', {}).get('exploration', {})
    epsilon_start = exploration_config.get('epsilon_start', 1.0)
    epsilon_end = exploration_config.get('epsilon_end', 0.01)
    epsilon_decay = exploration_config.get('epsilon_decay', 0.995)
    
    # Band exploration parameters
    band_exploration_config = config.get('training', {}).get('band_exploration', {})
    initial_band_switch_prob = band_exploration_config.get('initial_band_switch_prob', 0.3)
    final_band_switch_prob = band_exploration_config.get('final_band_switch_prob', 0.05)
    band_switch_decay = band_exploration_config.get('band_switch_decay', 0.999)
    
    # Training tracking
    episode_rewards = []
    episode_throughputs = []
    episode_distances = []
    episode_band_switches = []
    band_performance = {
        'mmwave': {'rewards': [], 'throughputs': [], 'usage': 0},
        'sub_thz_low': {'rewards': [], 'throughputs': [], 'usage': 0},
        'sub_thz_high': {'rewards': [], 'throughputs': [], 'usage': 0}
    }
    
    print(f"📊 Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset environment
        state, info = env.reset()
        
        episode_reward = 0.0
        episode_throughput = 0.0
        episode_distance = 0.0
        episode_band_switch_count = 0
        steps = 0
        
        # Calculate epsilon for this episode
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        
        # Calculate band switch probability for this episode
        band_switch_prob = max(final_band_switch_prob, 
                              initial_band_switch_prob * (band_switch_decay ** episode))
        
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
            
            # Track band switches
            if info.get('band_switched', False):
                episode_band_switch_count += 1
            
            # Track band performance
            current_band = info.get('band', 'mmwave')
            band_performance[current_band]['rewards'].append(reward)
            band_performance[current_band]['throughputs'].append(info.get('throughput', 0.0))
            band_performance[current_band]['usage'] += 1
            
            steps += 1
            
            if done or truncated:
                break
        
        # End episode
        agent.end_episode()
        
        # Calculate averages
        avg_throughput = episode_throughput / max(steps, 1)
        avg_distance = episode_distance / max(steps, 1)
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_throughputs.append(avg_throughput)
        episode_distances.append(avg_distance)
        episode_band_switches.append(episode_band_switch_count)
        
        # Print progress
        if episode % 100 == 0:
            current_band = env.current_band
            frequency_ghz = env.frequency_bands[current_band] / 1e9
            throughput_gbps = avg_throughput / 1e9
            
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Throughput: {throughput_gbps:.1f} Gbps")
            print(f"  Distance: {avg_distance:.1f}m")
            print(f"  Band: {current_band} ({frequency_ghz:.0f} GHz)")
            print(f"  Band Switches: {episode_band_switch_count}")
            print()
    
    # Save models
    models_dir = os.path.join(results_dir, 'models')
    agent.save_models(models_dir)
    
    # Prepare training results
    training_results = {
        'episode_rewards': episode_rewards,
        'episode_throughputs': episode_throughputs,
        'episode_distances': episode_distances,
        'episode_band_switches': episode_band_switches,
        'band_performance': band_performance,
        'hybrid_stats': env.get_hybrid_stats()
    }
    
    return training_results


def plot_hybrid_results(results: Dict[str, Any], results_dir: str):
    """Plot hybrid training results."""
    print("📈 Generating hybrid training plots...")
    
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
    axes[0, 1].plot(np.array(results['episode_throughputs']) / 1e9)
    axes[0, 1].set_title('Average Throughput')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Throughput (Gbps)')
    axes[0, 1].grid(True)
    
    # Distance curve
    axes[1, 0].plot(results['episode_distances'])
    axes[1, 0].set_title('Average Distance')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Distance (m)')
    axes[1, 0].grid(True)
    
    # Band switches curve
    axes[1, 1].plot(results['episode_band_switches'])
    axes[1, 1].set_title('Band Switches per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Band Switches')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'hybrid_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Band performance comparison
    band_performance = results['band_performance']
    bands = list(band_performance.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Band usage
    usage_counts = [band_performance[band]['usage'] for band in bands]
    axes[0].bar(bands, usage_counts)
    axes[0].set_title('Band Usage')
    axes[0].set_ylabel('Usage Count')
    axes[0].grid(True)
    
    # Average rewards by band
    avg_rewards = [np.mean(band_performance[band]['rewards']) if band_performance[band]['rewards'] else 0 
                  for band in bands]
    axes[1].bar(bands, avg_rewards)
    axes[1].set_title('Average Rewards by Band')
    axes[1].set_ylabel('Average Reward')
    axes[1].grid(True)
    
    # Average throughput by band
    avg_throughputs = [np.mean(band_performance[band]['throughputs']) / 1e9 if band_performance[band]['throughputs'] else 0 
                      for band in bands]
    axes[2].bar(bands, avg_throughputs)
    axes[2].set_title('Average Throughput by Band')
    axes[2].set_ylabel('Throughput (Gbps)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'band_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to {plots_dir}")


def save_hybrid_results(results: Dict[str, Any], results_dir: str, config: Dict[str, Any]):
    """Save hybrid training results."""
    print("💾 Saving hybrid training results...")
    
    # Save configuration
    config_path = os.path.join(results_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save training metrics
    metrics = {
        'final_stats': results['hybrid_stats'],
        'band_performance': results['band_performance'],
        'training_summary': {
            'total_episodes': len(results['episode_rewards']),
            'final_avg_reward': np.mean(results['episode_rewards'][-100:]),
            'final_avg_throughput': np.mean(results['episode_throughputs'][-100:]) / 1e9,
            'final_avg_distance': np.mean(results['episode_distances'][-100:]),
            'total_band_switches': sum(results['episode_band_switches'])
        }
    }
    
    metrics_path = os.path.join(results_dir, 'training_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"💾 Results saved to {results_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Hybrid 6G OAM Agent')
    parser.add_argument('--num-episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🎯 Hybrid 6G OAM Training")
    print("=" * 50)
    
    # Load configuration
    config = load_hybrid_config()
    
    # Create results directory
    if args.output_dir:
        results_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = create_results_dir(f"hybrid_6g_{timestamp}")
    
    # Save configuration
    config_path = os.path.join(results_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📁 Results will be saved to: {results_dir}")
    print(f"⚙️  Configuration saved to: {config_path}")
    
    # Train agent
    training_results = train_hybrid_agent(config, results_dir, args.num_episodes)
    
    # Generate plots
    plot_hybrid_results(training_results, results_dir)
    
    # Save results
    save_hybrid_results(training_results, results_dir, config)
    
    # Print final statistics
    hybrid_stats = training_results['hybrid_stats']
    print("\n🎉 Hybrid Training Complete!")
    print("=" * 50)
    print(f"📊 Final Hybrid Statistics:")
    print(f"   Total Band Switches: {hybrid_stats['band_switch_count']}")
    print(f"   Current Band: {hybrid_stats['current_band']}")
    
    band_performance = training_results['band_performance']
    print(f"\n📡 Band Performance Summary:")
    for band, stats in band_performance.items():
        if stats['usage'] > 0:
            avg_reward = np.mean(stats['rewards']) if stats['rewards'] else 0
            avg_throughput = np.mean(stats['throughputs']) / 1e9 if stats['throughputs'] else 0
            print(f"   {band}: Usage={stats['usage']}, Avg Reward={avg_reward:.2f}, "
                  f"Avg Throughput={avg_throughput:.1f} Gbps")
    
    print(f"\n📁 All results saved to: {results_dir}")


if __name__ == "__main__":
    main() 