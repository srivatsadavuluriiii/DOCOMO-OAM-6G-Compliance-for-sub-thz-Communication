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

                                                                     
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from models.agent import Agent
from environment.docomo_6g_env import DOCOMO_6G_Environment
from utils.visualization_unified import create_evaluation_dashboard, visualize_q_values
                                                                                   
import matplotlib.pyplot as plt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent for OAM handover decisions")
    
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing the trained model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file (defaults to config.yaml in model directory)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save evaluation results (defaults to model directory)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information during evaluation")
    
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def collect_q_values(
    agent: Agent,
    states: np.ndarray
) -> np.ndarray:
    """
    Collect Q-values for a batch of states.
    
    Args:
        agent: Trained agent
        states: Batch of states
        
    Returns:
        Array of Q-values for each state-action pair
    """
    with torch.no_grad():
        states_tensor = torch.FloatTensor(states).to(agent.device)
        q_values = agent.policy_net(states_tensor).cpu().numpy()
    
    return q_values


def evaluate(args: argparse.Namespace) -> None:
    """
    Evaluate a trained DQN agent.
    
    Args:
        args: Command line arguments
    """
                     
    set_random_seed(args.seed)
    
                                      
    model_dir = args.model_dir
    if args.output_dir is None:
        output_dir = os.path.join(model_dir, "evaluation")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
                            
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
                                        
    if args.config is None:
        config_path = os.path.join(model_dir, "config.yaml")
    else:
        config_path = args.config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}
    
                
    device = torch.device("cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")
    
                                                                          
    candidate_final = os.path.join(model_dir, "models", "final")
    candidate_root = os.path.join(model_dir, "models")
    model_path = candidate_final if os.path.isdir(candidate_final) else candidate_root
    policy_path = os.path.join(model_path, "policy_net.pth")
    expected_state_dim = None
    expected_action_dim = None
    try:
        sd = torch.load(policy_path, map_location="cpu")
        w0 = sd.get("model.0.weight", None)
        w_out = sd.get("model.4.weight", None)
        if w0 is not None and hasattr(w0, "size"):
            expected_state_dim = int(w0.size(1))
        if w_out is not None and hasattr(w_out, "size"):
            expected_action_dim = int(w_out.size(0))
    except Exception:
        pass

                               
    if args.config is not None:
        env = DOCOMO_6G_Environment(config_path=args.config)
    else:
        env = DOCOMO_6G_Environment(config=config)
    
                                     
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
                                                                 
    network_cfg = config.get('network', {}) if isinstance(config, dict) else {}
    hidden_layers = network_cfg.get('hidden_layers', [128, 128])
    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=hidden_layers,
        device=device
    )
    
                                                            
    best_model_path = os.path.join(model_dir, "best_model.pth")
    if os.path.isfile(best_model_path):
        try:
            ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
            if 'policy_net_state_dict' in ckpt:
                agent.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
            if 'target_net_state_dict' in ckpt:
                agent.target_net.load_state_dict(ckpt['target_net_state_dict'])
            print(f"Loaded best model from {best_model_path}")
        except Exception as e:
            print(f"Warning: Could not load best_model.pth: {e}")
    else:
                                                 
        candidate_final = os.path.join(model_dir, "models", "final")
        candidate_root = os.path.join(model_dir, "models")
        model_path = candidate_final if os.path.isdir(candidate_final) else candidate_root
        try:
            agent.load_models(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
    
                     
    episode_rewards = []
    episode_throughputs = []
    episode_handovers = []
    episode_sinrs = []
    episode_modes = []
    episode_distances = []
    collected_states = []
    
    print(f"Starting evaluation for {args.episodes} episodes...")
    
    for episode in range(1, args.episodes + 1):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_sinr_values = []
        episode_mode_values = []
        episode_distance_values = []
        
        while not (done or truncated):
                                              
            collected_states.append(state)
            
                                     
            episode_sinr_values.append(state[0])        
            episode_distance_values.append(state[1])            
            episode_mode_values.append(state[5])                
            
                                                              
            action = agent.choose_action(state, epsilon=0.0)
            
                         
            next_state, reward, done, truncated, info = env.step(action)
            
                             
            episode_reward += reward
            state = next_state
            
                                 
            if args.render:
                env.render()
        
                               
        episode_rewards.append(episode_reward)
        try:
                                        
            kpis = env.kpi_tracker.get_current_kpis()
            episode_throughputs.append(kpis.get('current_throughput_gbps', 0.0) * 1e9)
        except Exception:
            episode_throughputs.append(0.0)
        try:
            ep_stats = getattr(env, 'episode_stats', {})
            episode_handovers.append(int(ep_stats.get('handover_count', 0)))
        except Exception:
            episode_handovers.append(0)
        episode_sinrs.append(episode_sinr_values)
        episode_modes.append(episode_mode_values)
        episode_distances.append(episode_distance_values)
        
                        
        if args.verbose or episode % 10 == 0:
            print(f"Episode {episode}/{args.episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Throughput: {env.episode_throughput:.2e} | "
                  f"Handovers: {env.episode_handovers}")
    
                               
    avg_reward = np.mean(episode_rewards)
    avg_throughput = np.mean(episode_throughputs)
    avg_handovers = np.mean(episode_handovers)
    
    print("\nEvaluation Results:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Throughput: {avg_throughput:.2e} bps")
    print(f"  Average Handovers: {avg_handovers:.2f}")
    
                                   
    if len(collected_states) > 1000:
                                                          
        collected_states = np.array(collected_states)
        indices = np.random.choice(len(collected_states), 1000, replace=False)
        sampled_states = collected_states[indices]
    else:
        sampled_states = np.array(collected_states)
    
    q_values = collect_q_values(agent, sampled_states)
    
                           
    
                      
    if action_dim == 3:
        action_names = ["STAY", "UP", "DOWN"]
    elif action_dim == 8:
        action_names = ["STAY", "OAM_UP", "OAM_DOWN", "BAND_UP", "BAND_DOWN", "BEAM_TRACK", "PREDICT", "HANDOVER"]
    else:
        action_names = [f"A{i}" for i in range(action_dim)]
    q_plot_path = os.path.join(plots_dir, "q_values.png")
    visualize_q_values(q_values, action_names, q_plot_path)
    
                                                              
    plt.figure(figsize=(10, 8))
    
                              
    all_sinrs = np.concatenate(episode_sinrs)
    all_distances = np.concatenate(episode_distances)
    all_modes = np.concatenate(episode_modes)
    
    scatter = plt.scatter(all_distances, all_sinrs, c=all_modes, cmap='viridis', 
                          alpha=0.6, edgecolors='w', linewidth=0.5)
    
    plt.colorbar(scatter, label='OAM Mode')
    plt.xlabel('Distance (m)')
    plt.ylabel('SINR (dB)')
    plt.title('SINR vs. Distance with OAM Mode Selection')
    plt.grid(True, alpha=0.3)
    
                    
    sinr_plot_path = os.path.join(plots_dir, 'sinr_distance_mode.png')
    plt.savefig(sinr_plot_path)
    
                                     
    dashboard_path = os.path.join(output_dir, "dashboard.html")
    create_evaluation_dashboard(
        episode_rewards,
        episode_throughputs,
        episode_handovers,
        dashboard_path
    )
    
                             
    metrics = {
        "avg_reward": float(avg_reward),
        "avg_throughput": float(avg_throughput),
        "avg_handovers": float(avg_handovers),
        "rewards": [float(r) for r in episode_rewards],
        "throughputs": [float(t) for t in episode_throughputs],
        "handovers": [int(h) for h in episode_handovers]
    }
    
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nEvaluation results saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args) 
