#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# Ensure project root is on sys.path before importing project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from models.agent import Agent
from environment.oam_env import OAM_Env
from utils.visualization_unified import plot_training_curves
from utils.config_utils import load_config
from utils.hierarchical_config import load_hierarchical_config, merge_configs
from utils.visualization_unified import MetricsLogger
from utils.state_dimension_validator import validate_state_dimensions, get_state_dimension_report
import time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a DQN agent for OAM handover decisions")
    
    parser.add_argument("--config", type=str, default="rl_config_new",
                        help="Path to RL configuration file")
    parser.add_argument("--sim-config", type=str, default="config/simulation_params.yaml",
                        help="Path to simulation configuration file")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Episodes between logging")
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Episodes between evaluations")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Override number of episodes from config")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max steps per episode from config")
    
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_agent(
    env: OAM_Env,
    agent: Agent,
    n_episodes: int = 10,
    epsilon: float = 0.05
) -> Dict[str, float]:
    """
    Evaluate the agent's performance.
    
    Args:
        env: Environment to evaluate in
        agent: Agent to evaluate
        n_episodes: Number of evaluation episodes
        epsilon: Exploration probability (usually 0 for evaluation)
        
    Returns:
        Dictionary of evaluation metrics
    """
    total_rewards = []
    total_avg_throughputs = []
    total_handovers = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        sum_throughput = 0.0
        while not (done or truncated):
            # Choose action
            action = agent.choose_action(state, epsilon)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update tracking
            episode_reward += reward
            state = next_state
            steps += 1
            sum_throughput += float(info.get('throughput', 0.0))
        
        # Store episode metrics
        total_rewards.append(episode_reward)
        avg_throughput = sum_throughput / max(steps, 1)
        total_avg_throughputs.append(avg_throughput)
        total_handovers.append(info['handovers'])
        episode_lengths.append(steps)
    
    # Calculate average metrics
    avg_reward = np.mean(total_rewards)
    avg_throughput = np.mean(total_avg_throughputs)
    avg_handovers = np.mean(total_handovers)
    avg_episode_length = np.mean(episode_lengths)
    
    return {
        'reward': avg_reward,
        'throughput': avg_throughput,
        'handovers': avg_handovers,
        'episode_length': avg_episode_length
    }


def train(args: argparse.Namespace) -> None:
    """
    Train the DQN agent.
    
    Args:
        args: Command line arguments
    """
    # Set random seed
    set_random_seed(args.seed)
    
    # Load configurations
    if args.config.endswith('.yaml'):
        # Legacy config file
        rl_config = load_config(args.config)
        sim_config = load_config(args.sim_config)
        
        # Merge configurations
        config = merge_configs(sim_config, rl_config)
    else:
        # Hierarchical config
        config = load_hierarchical_config(args.config)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"train_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Set device
    device = torch.device("cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")
    
    # Create environment
    env = OAM_Env(config)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    network_cfg = config.get('network') or config.get('rl_base', {}).get('network', {})
    replay_cfg = config.get('replay_buffer') or config.get('rl_base', {}).get('replay_buffer', {})

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=network_cfg.get('hidden_layers', [128, 128]),
        learning_rate=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        buffer_capacity=replay_cfg.get('capacity', 50000),
        batch_size=config['training']['batch_size'],
        target_update_freq=config['training']['target_update_freq'],
        device=device
    )
    
    # Validate state dimensions
    print("🔍 Validating state dimensions...")
    if not validate_state_dimensions(env, config, agent):
        print("❌ State dimension validation failed!")
        print(get_state_dimension_report(env, config, agent))
        raise ValueError("State dimension mismatch detected. Check the validation report above.")
    else:
        print("✅ State dimension validation passed!")
        print(get_state_dimension_report(env, config, agent))
    
    # Create metrics logger
    log_dir = os.path.join(output_dir, "logs")
    metrics_logger = MetricsLogger(log_dir)
    
    # Training parameters
    num_episodes = args.num_episodes if args.num_episodes is not None else config['training']['num_episodes']
    max_steps = args.max_steps if args.max_steps is not None else config['training']['max_steps_per_episode']
    exploration_cfg = config.get('exploration') or config.get('rl_base', {}).get('exploration', {})
    epsilon_start = exploration_cfg.get('epsilon_start', 1.0)
    epsilon_end = exploration_cfg.get('epsilon_end', 0.01)
    epsilon_decay = exploration_cfg.get('epsilon_decay', 0.99)
    
    # Training loop
    epsilon = epsilon_start
    total_steps = 0
    episode_rewards = []
    episode_throughputs = []  # per-episode average throughput (bps)
    episode_throughput_sums = []  # per-episode raw sum throughput (bps-step aggregation)
    episode_handovers = []
    
    print(f"Starting training for {num_episodes} episodes with max {max_steps} steps per episode...")
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state, info = env.reset()
        episode_reward = 0
        episode_loss = 0
        done = False
        truncated = False
        
        for step in range(max_steps):
            # Choose action
            action = agent.choose_action(state, epsilon)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done or truncated)
            
            # Update state and tracking
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Learn from experience
            if agent.replay_buffer.is_ready(agent.batch_size):
                loss = agent.learn()
                episode_loss += loss
            
            # Check if episode is done
            if done or truncated:
                break
        
        # End of episode updates
        agent.end_episode()
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        steps_taken = getattr(env, 'steps', (step + 1))
        episode_avg_throughput = float(env.episode_throughput) / max(int(steps_taken), 1)
        episode_throughputs.append(episode_avg_throughput)
        episode_throughput_sums.append(float(env.episode_throughput))
        episode_handovers.append(env.episode_handovers)
        
        # Log metrics
        metrics_logger.log_scalar("reward", episode_reward, episode)
        # Log episode-average throughput as the primary throughput metric
        metrics_logger.log_scalar("throughput", episode_avg_throughput, episode)
        # Also log raw sum for analysis/back-compat
        metrics_logger.log_scalar("throughput_sum", float(env.episode_throughput), episode)
        metrics_logger.log_scalar("handovers", env.episode_handovers, episode)
        metrics_logger.log_scalar("epsilon", epsilon, episode)
        
        if episode_loss > 0:
            metrics_logger.log_scalar("loss", episode_loss / step, episode)
        
        # Print progress
        if episode % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            avg_throughput = np.mean(episode_throughputs[-args.log_interval:])
            avg_handovers = np.mean(episode_handovers[-args.log_interval:])
            
            print(f"Episode {episode}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                   f"Avg Throughput: {avg_throughput:.2e} | "
                  f"Avg Handovers: {avg_handovers:.2f} | "
                  f"Epsilon: {epsilon:.4f}")
        
        # Evaluate agent
        if episode % args.eval_interval == 0:
            eval_metrics = evaluate_agent(env, agent, n_episodes=5)
            
            print(f"\nEvaluation at episode {episode}:")
            print(f"  Avg Reward: {eval_metrics['reward']:.2f}")
            print(f"  Avg Throughput: {eval_metrics['throughput']:.2e}")
            print(f"  Avg Handovers: {eval_metrics['handovers']:.2f}")
            print(f"  Avg Episode Length: {eval_metrics['episode_length']:.2f}")
            
            # Log evaluation metrics
            for key, value in eval_metrics.items():
                metrics_logger.log_scalar(f"eval_{key}", value, episode)
            
            # Save model weights
            save_dir = os.path.join(output_dir, "models", f"episode_{episode}")
            agent.save_models(save_dir)
    
    # Training complete
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save final model
    final_save_dir = os.path.join(output_dir, "models", "final")
    agent.save_models(final_save_dir)
    
    # Plot training curves
    plot_path = os.path.join(output_dir, "training_curves.png")
    plot_training_curves(metrics_logger, plot_path)
    
    # Save training metrics
    metrics = {
        "rewards": episode_rewards,
        # Store per-episode average throughput to align with evaluation
        "throughputs": episode_throughputs,
        # Provide raw sums for analysis
        "throughputs_sum": episode_throughput_sums,
        "handovers": episode_handovers,
        "training_time": training_time
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    
    # Close logger
    metrics_logger.close()


if __name__ == "__main__":
    args = parse_args()
    train(args) 

