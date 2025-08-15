#!/usr/bin/env python3
"""
DOCOMO 6G Professional Training Script
Train DQN agent for DOCOMO 6G OAM optimization using existing project infrastructure
Targets: 1 Tbps peak, 0.1ms latency, 500 km/h mobility, 99.99999% reliability
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

                                                                     
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

                                        
from models.agent import Agent
from environment.docomo_6g_env import DOCOMO_6G_Environment
from models.multi_objective_reward import MultiObjectiveReward
from utils.visualization_unified import plot_training_curves, MetricsLogger
from utils.config_utils import load_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN agent for DOCOMO 6G OAM optimization")
    
    parser.add_argument("--config", type=str, default="config/docomo_6g/docomo_6g_config.yaml",
                        help="Path to DOCOMO 6G configuration file")
    parser.add_argument("--output-dir", type=str, default="results/docomo_6g_training",
                        help="Directory to save results")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Episodes between logging")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Episodes between model saves")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum steps per episode")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Episodes for evaluation")
    parser.add_argument("--num-users", type=int, default=1,
                        help="Number of users (agents) in the simulation")
    
    return parser.parse_args()

class DOCOMOTrainingManager:
    """Professional DOCOMO 6G Training Manager using existing project infrastructure"""
    
    def __init__(self, config_path: str, output_dir: str, args: argparse.Namespace):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.args = args
        
                                 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
                          
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
                           
        if not args.no_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif not args.no_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"  Using device: {self.device}")
        
                            
        self.config = self._load_config()
        
                                
        print(" Initializing DOCOMO 6G Environment...")
        self.env = DOCOMO_6G_Environment(config_path=config_path, num_users=self.args.num_users)
        print(f"    State space: {self.env.observation_space}")
        print(f"    Action space: {self.env.action_space}")
        print(f"    Frequency bands: {len(self.env.band_names)}")
        
        print(" Initializing DQN Agents...")
        self.agents: Dict[str, Agent] = {}
        self.agent_ids = self.env.agent_ids # Get agent IDs from environment

        agent_config = self.config.get('agent', {})
        rl_config = self.config.get('docomo_6g_system', {}).get('reinforcement_learning', {})
        training_config = rl_config.get('training', {})

        for agent_id in self.agent_ids:
            # For MARL, each agent observes its own state + other agents' info.
            # The observation space for each agent is a Box within the overall Dict space.
            single_agent_obs_space = self.env.observation_space[agent_id]
            state_dim = single_agent_obs_space.shape[0]
            action_dim = self.env.action_space[agent_id].n # Each agent has its own discrete action space
            
            self.agents[agent_id] = Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_layers=agent_config.get('hidden_layers', [256, 256]),
                learning_rate=training_config.get('learning_rate', 1e-4),
                gamma=agent_config.get('gamma', 0.99),
                buffer_capacity=training_config.get('buffer_capacity', 100000),
                batch_size=training_config.get('batch_size', 64),
                target_update_freq=training_config.get('target_update_freq', 10),
                device=self.device,
                double_dqn=training_config.get('double_dqn', False),
                dueling_dqn=training_config.get('dueling_dqn', False)
            )
            self.agents[agent_id].policy_net.to(self.device)
            self.agents[agent_id].target_net.to(self.device)
            print(f"    Agent {agent_id}: Double DQN: {self.agents[agent_id].double_dqn}, Dueling DQN: {self.agents[agent_id].policy_net.dueling}")

        print(f"    Total model parameters: {sum(p.numel() for agent in self.agents.values() for p in agent.policy_net.parameters())}")
        
        self.epsilon_start = agent_config.get('epsilon_start', 1.0)
        self.epsilon_end = agent_config.get('epsilon_end', 0.01)
        self.epsilon_decay_steps = agent_config.get('epsilon_decay_steps', 10000)
        self.epsilon = self.epsilon_start
        self.step_count = 0
        
        self.metrics_logger = MetricsLogger(log_dir=str(self.output_dir))
        self.training_metrics = {
            'episodes': [],
            'rewards': [], # This will now store sum/avg of all agents' rewards
            'losses': [],  # This will now store sum/avg of all agents' losses
            'agents_metrics': {} # New dictionary to store per-agent metrics
            # 'kpis': [],  # Removed, moved to agents_metrics
            # 'avg_throughput_gbps': [],  # Removed, moved to agents_metrics
            # 'avg_latency_ms': [],       # Removed, moved to agents_metrics
            # 'band_performance': {band: [] for band in self.env.band_names} # Removed, moved to agents_metrics
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load DOCOMO 6G configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f" Loaded config from {self.config_path}")
            return config
        except Exception as e:
            print(f" Error loading config: {e}")
                                   
            return {
                'agent': {
                    'hidden_layers': [256, 256],
                    'learning_rate': 1e-4,
                    'gamma': 0.99,
                    'buffer_capacity': 100000,
                    'batch_size': 64,
                    'target_update_freq': 10,
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.2,  # Maintain higher exploration
                    'epsilon_decay_steps': 100000  # Much slower decay for consistency
            }
        }
    
    def _select_action(self, states: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Select actions for all agents using epsilon-greedy policy"""
        actions = {}
        for agent_id, state_tensor in states.items():
            if np.random.random() < self.epsilon:
                actions[agent_id] = self.env.action_space[agent_id].sample()
            else:
                actions[agent_id] = self.agents[agent_id].choose_action(state_tensor.cpu().numpy(), epsilon=0.0)
        return actions
    
    def _decay_epsilon(self):
        """Decay epsilon linearly"""
        if self.step_count < self.epsilon_decay_steps:
            decay_ratio = self.step_count / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay_ratio
        else:
            self.epsilon = self.epsilon_end
    
    def train(self, episodes: Optional[int] = None) -> Dict[str, Any]:
        """Train the DOCOMO 6G agent, potentially with a curriculum."""
        if episodes is None:
            episodes = self.args.episodes
            
        curriculum_config = self.config.get('curriculum', {})
        use_curriculum = curriculum_config.get('enabled', False) and 'stages' in curriculum_config

        if use_curriculum:
            self._train_curriculum(curriculum_config['stages'])
        else:
            self._train_single_stage(episodes)

        training_time = time.time() - self.training_start_time
        
        print("\n" + "="*60)
        print(" DOCOMO 6G Training Completed!")
        total_episodes = sum(s['episodes'] for s in curriculum_config['stages']) if use_curriculum else episodes
        if total_episodes > 0:
            print(f"     Total time: {training_time:.2f}s ({training_time/total_episodes:.2f}s/episode)")
        else:
            print(f"     Total time: {training_time:.2f}s (evaluation-only)")
        print(f"    Best reward: {self.best_reward:.4f}")
        print(f"    Best DOCOMO compliance: {self.best_compliance:.1%}")
        print(f"    Final epsilon: {self.epsilon:.4f}")
        # Print buffer sizes for all agents
        total_buffer_size = sum(len(agent.replay_buffer) for agent in self.agents.values())
        print(f"    Total buffer size: {total_buffer_size} (avg: {total_buffer_size/len(self.agents):.0f} per agent)")
        
        final_metrics = {}
        if int(self.args.eval_episodes) > 0:
            final_metrics = self._evaluate_agent(self.args.eval_episodes)
        
        self._save_final_results(training_time, self.best_reward, self.best_compliance, final_metrics)
        
        return final_metrics

    def _train_curriculum(self, stages: List[Dict[str, Any]]):
        """Train the agent using a curriculum of stages."""
        print("\n Starting DOCOMO 6G Training with Curriculum")
        print("="*60)
        self.training_start_time = time.time()
        self.best_reward = float('-inf')
        self.best_compliance = 0.0

        for i, stage in enumerate(stages):
            print(f"\n--- Stage {i+1}: {stage['name']} ---")
            
            # Merge config overrides for the current stage
            stage_config = self.config.copy()
            if 'config_overrides' in stage:
                # Deep merge would be better here, but for now, we'll do a simple update
                stage_config.update(stage['config_overrides'])

            self.env.reconfigure(stage_config)
            
            self._train_single_stage(stage['episodes'])

    def _train_single_stage(self, episodes: int):
        """Train the agent for a single stage or a full training run."""
        if not hasattr(self, 'training_start_time'):
            self.training_start_time = time.time()
            self.best_reward = float('-inf')
            self.best_compliance = 0.0

        for episode in range(episodes):
            episode_start = time.time()
            
            states, infos = self.env.reset()
            # Convert initial states to tensors for each agent
            states = {agent_id: torch.FloatTensor(s).to(self.device) for agent_id, s in states.items()}
            
            episode_rewards: Dict[str, float] = {agent_id: 0.0 for agent_id in self.agent_ids}
            episode_losses: Dict[str, float] = {agent_id: 0.0 for agent_id in self.agent_ids}
            steps = 0
            # Per-agent band usage and KPI accumulators
            band_usage: Dict[str, Dict[str, int]] = {agent_id: {band: 0 for band in self.env.band_names} for agent_id in self.agent_ids}
            step_tp_list: Dict[str, List[float]] = {agent_id: [] for agent_id in self.agent_ids}
            step_lat_list: Dict[str, List[float]] = {agent_id: [] for agent_id in self.agent_ids}
            band_tp_sums: Dict[str, Dict[str, float]] = {agent_id: {band: 0.0 for band in self.env.band_names} for agent_id in self.agent_ids}
            
            while steps < self.args.max_steps:
                actions = self._select_action(states) # Get actions for all agents
                self.step_count += 1
                
                self._decay_epsilon()
                
                next_states, rewards, dones, truncateds, infos = self.env.step(actions) # Step with all agents' actions
                
                # Process each agent's experience
                for agent_id in self.agent_ids:
                    state = states[agent_id]
                    action = actions[agent_id]
                    reward = rewards[agent_id]
                    next_state = torch.FloatTensor(next_states[agent_id]).to(self.device)
                    done = dones[agent_id]
                    truncated = truncateds[agent_id]
                    info = infos[agent_id]
                    
                    self.agents[agent_id].replay_buffer.push(
                        state.cpu().numpy(),
                        action,
                        reward,
                        next_state.cpu().numpy(),
                        done or truncated
                    )
                    
                    episode_rewards[agent_id] += reward
                    current_band = info.get('frequency_band', 'unknown')
                    if current_band in band_usage[agent_id]:
                        band_usage[agent_id][current_band] += 1
                    
                    try:
                        perf = info.get('performance_metrics', {})
                        tp_gbps = float(perf.get('throughput_gbps', 0.0))
                        lat_ms = float(perf.get('latency_ms', 0.0))
                        step_tp_list[agent_id].append(tp_gbps)
                        step_lat_list[agent_id].append(lat_ms)
                        if current_band in band_tp_sums[agent_id]:
                            band_tp_sums[agent_id][current_band] += tp_gbps
                    except Exception:
                        pass
                    
                    if self.agents[agent_id].replay_buffer.is_ready(self.agents[agent_id].batch_size):
                        loss_value = self.agents[agent_id].learn()
                        if loss_value is not None and isinstance(loss_value, (int, float)):
                            episode_losses[agent_id] += loss_value
                
                states = {agent_id: torch.FloatTensor(next_states[agent_id]).to(self.device) for agent_id in self.agent_ids}
                steps += 1
                
                # Check if all agents are done or truncated
                if all(dones.values()) or all(truncateds.values()):
                    break
            
            # Update target networks for all agents
            if (episode + 1) % self.agents[self.agent_ids[0]].target_update_freq == 0: # Assuming same update freq for all agents
                for agent_id in self.agent_ids:
                    self.agents[agent_id].update_target_network()
            
            # Aggregate training metrics per agent
            self.training_metrics['episodes'].append(episode)
            for agent_id in self.agent_ids:
                if agent_id not in self.training_metrics['agents_metrics']:
                    self.training_metrics['agents_metrics'][agent_id] = {
                        'rewards': [],
                        'losses': [],
                        'avg_throughput_gbps': [],
                        'avg_latency_ms': [],
                        'kpis': [],
                        'compliance': [],
                        'band_performance': {band: [] for band in self.env.band_names}
                    }
                
                self.training_metrics['agents_metrics'][agent_id]['rewards'].append(episode_rewards[agent_id])
                self.training_metrics['agents_metrics'][agent_id]['losses'].append(episode_losses[agent_id] / max(steps, 1))
                self.training_metrics['agents_metrics'][agent_id]['avg_throughput_gbps'].append(float(np.mean(step_tp_list[agent_id])) if step_tp_list[agent_id] else 0.0)
                self.training_metrics['agents_metrics'][agent_id]['avg_latency_ms'].append(float(np.mean(step_lat_list[agent_id])) if step_lat_list[agent_id] else 0.0)
                
                if hasattr(self.env, 'kpi_trackers') and self.env.kpi_trackers[agent_id].measurements:
                    kpis = self.env.kpi_trackers[agent_id].get_current_kpis()
                    self.training_metrics['agents_metrics'][agent_id]['kpis'].append(kpis)
                    
                    compliance = self._calculate_docomo_compliance(kpis, agent_id)
                    self.training_metrics['agents_metrics'][agent_id]['compliance'].append(compliance)
                    if compliance > self.best_compliance: # Still track global best compliance
                        self.best_compliance = compliance

                for band, count in band_usage[agent_id].items():
                    if count > 0:
                        avg_tp_band = band_tp_sums[agent_id].get(band, 0.0) / float(count)
                        self.training_metrics['agents_metrics'][agent_id]['band_performance'][band].append(avg_tp_band)
            
            # Global best reward (e.g., sum of all agents' rewards, or average)
            current_total_reward = sum(episode_rewards.values())
            if current_total_reward > self.best_reward:
                self.best_reward = current_total_reward
                self._save_best_model(episode, current_total_reward, self.best_compliance)
            
            if (episode + 1) % self.args.log_interval == 0:
                episode_time = time.time() - episode_start
                
                print(f"Episode {episode+1:4d}: "
                      f"Total Reward={current_total_reward:7.2f}, "
                      f"Avg Loss={np.mean(list(episode_losses.values())):6.4f}, "
                      f"Steps={steps:3d}, "
                      f"ε={self.epsilon:.3f}, "
                      f"Time={episode_time:.2f}s")
                
                for agent_id in self.agent_ids:
                    agent_metrics = self.training_metrics['agents_metrics'][agent_id]
                    
                    # Use episode averages instead of potentially misleading current KPI values
                    if agent_metrics['avg_throughput_gbps']:
                        throughput = agent_metrics['avg_throughput_gbps'][-1]  # Latest episode average
                        latency = agent_metrics['avg_latency_ms'][-1] if agent_metrics['avg_latency_ms'] else 0
                        
                        # Get reliability from KPI tracker if available
                        if agent_metrics['kpis']:
                            latest_kpis = agent_metrics['kpis'][-1]
                            reliability = latest_kpis.get('current_reliability', 0)
                        else:
                            reliability = 0
                            latest_kpis = {}
                    else:
                        throughput = latency = reliability = 0
                        latest_kpis = {}
                        
                    print(f"  Agent {agent_id} KPIs: T={throughput:.3f}Gbps, L={latency:.3f}ms, R={reliability:.4f}")
                        
                    # Most used band for this agent
                    total_band_uses = sum(band_usage[agent_id].values())
                    if total_band_uses > 0:
                        most_used_band_agent = max(band_usage[agent_id].items(), key=lambda x: x[1])
                        try:
                            freq_val = self.env.frequency_bands[most_used_band_agent[0]].get('frequency', 28.0e9)
                            freq_ghz = float(freq_val) / 1e9
                        except Exception:
                            freq_ghz = 28.0
                        # Get handover count from KPI tracker
                        handover_count = 0
                        if hasattr(self.env, 'kpi_trackers') and agent_id in self.env.kpi_trackers:
                            handover_count = self.env.kpi_trackers[agent_id].session_stats.get('handover_count', 0)
                        
                        print(f"  Agent {agent_id} Band: {most_used_band_agent[0]} ({freq_ghz:.0f} GHz) - {most_used_band_agent[1]} uses (H: {handover_count})")
                    else:
                        # No band switches occurred in this episode, show current band
                        current_band = self.env.current_bands[agent_id]
                        try:
                            freq_val = self.env.frequency_bands[current_band].get('frequency', 28.0e9)
                            freq_ghz = float(freq_val) / 1e9
                        except Exception:
                            freq_ghz = 28.0
                        # Get handover count from KPI tracker
                        handover_count = 0
                        if hasattr(self.env, 'kpi_trackers') and agent_id in self.env.kpi_trackers:
                            handover_count = self.env.kpi_trackers[agent_id].session_stats.get('handover_count', 0)
                        
                        print(f"  Agent {agent_id} Band: {current_band} ({freq_ghz:.0f} GHz) - no switches (H: {handover_count})")
                        
                    targets = getattr(self.env.kpi_trackers[agent_id], 'docomo_targets', None)
                    comp = self.env.kpi_trackers[agent_id].get_compliance_score() if hasattr(self.env, 'kpi_trackers') else {}
                    overall_compliance_agent = agent_metrics['compliance'][-1] if agent_metrics['compliance'] else 0.0
                    if targets:
                        print(
                            f"  Agent {agent_id} [DOCOMO] T: "
                            f"{throughput:.3f}/{targets.user_data_rate_gbps:.0f} Gbps | "
                            f"L: {latency:.3f}/{targets.latency_ms:.3f} ms | "
                            f"R: {reliability:.7f}/{targets.reliability:.7f} | "
                            f"M: {latest_kpis.get('current_mobility_kmh', 0):.1f}/{targets.mobility_kmh:.0f} km/h | "
                            f"Compliance: {overall_compliance_agent*100:.1f}%"
                        )
                
            if (episode + 1) % self.args.save_interval == 0:
                self._save_checkpoint(episode, current_total_reward)
    
    def _calculate_docomo_compliance(self, kpis: Dict[str, Any], agent_id: str) -> float:
        """Calculate DOCOMO 6G compliance score using KPI tracker (overall)."""
        if hasattr(self.env, 'kpi_trackers') and agent_id in self.env.kpi_trackers:
            comp = self.env.kpi_trackers[agent_id].get_compliance_score()
                                                                
            return float(comp.get('overall_current', comp.get('overall_avg', 0.0)))
        return 0.0
    
    def _evaluate_agent(self, num_episodes: int) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        print(f"\n Evaluating agent over {num_episodes} episodes...")

                                                                                
        try:
            best_path = self.output_dir / "best_model.pth"
            if best_path.exists():
                ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                if 'policy_net_state_dict' in ckpt: # Old single-agent save
                    self.agents[self.agent_ids[0]].policy_net.load_state_dict(ckpt['policy_net_state_dict'])
                    self.agents[self.agent_ids[0]].target_net.load_state_dict(ckpt['target_net_state_dict'])
                    print(f"    Loaded best single-agent model for evaluation: {best_path}")
                elif 'agents_policy_net_state_dicts' in ckpt: # New multi-agent save
                    for agent_id in self.agent_ids:
                        if agent_id in ckpt['agents_policy_net_state_dicts']:
                            self.agents[agent_id].policy_net.load_state_dict(ckpt['agents_policy_net_state_dicts'][agent_id])
                            self.agents[agent_id].target_net.load_state_dict(ckpt['agents_target_net_state_dicts'][agent_id])
                    print(f"    Loaded best multi-agent model for evaluation: {best_path}")
        except Exception as e:
            print(f"     Could not load best model for evaluation: {e}")
        
        eval_rewards: Dict[str, List[float]] = {agent_id: [] for agent_id in self.agent_ids}
        eval_kpis: Dict[str, List[Dict[str, Any]]] = {agent_id: [] for agent_id in self.agent_ids}
        eval_band_usage: Dict[str, Dict[str, int]] = {agent_id: {band: 0 for band in self.env.band_names} for agent_id in self.agent_ids}
        eval_compliances: Dict[str, List[float]] = {agent_id: [] for agent_id in self.agent_ids}
        eval_episode_handovers: Dict[str, List[int]] = {agent_id: [] for agent_id in self.agent_ids}
        
        old_epsilon = self.epsilon
        self.epsilon = 0.0 # No exploration during evaluation
        
        for episode in range(num_episodes):
            states, infos = self.env.reset()
            episode_rewards_current: Dict[str, float] = {agent_id: 0.0 for agent_id in self.agent_ids}
            steps = 0
            
            while steps < self.args.max_steps:
                actions = {} # Actions for all agents
                for agent_id in self.agent_ids:
                    state_tensor = torch.FloatTensor(states[agent_id]).to(self.device)
                    actions[agent_id] = self.agents[agent_id].choose_action(state_tensor.cpu().numpy(), epsilon=0.0)
                
                next_states, rewards, dones, truncateds, infos = self.env.step(actions)
                
                for agent_id in self.agent_ids:
                    episode_rewards_current[agent_id] += rewards[agent_id]
                    current_band = infos[agent_id].get('frequency_band', 'unknown')
                    if current_band in eval_band_usage[agent_id]:
                        eval_band_usage[agent_id][current_band] += 1
                
                states = next_states # Update states for next step
                steps += 1
                
                if all(dones.values()) or all(truncateds.values()):
                    break
            
            for agent_id in self.agent_ids:
                eval_rewards[agent_id].append(episode_rewards_current[agent_id])
                
                if hasattr(self.env, 'kpi_trackers') and agent_id in self.env.kpi_trackers:
                    comp = self.env.kpi_trackers[agent_id].get_compliance_score()
                    if 'overall_current' in comp:
                        eval_compliances[agent_id].append(float(comp['overall_current']))
                                                                           
                try:
                    if isinstance(getattr(self.env, 'episode_stats', None), dict) and agent_id in self.env.episode_stats:
                        eval_episode_handovers[agent_id].append(int(self.env.episode_stats[agent_id].get('handover_count', 0)))
                except Exception:
                    pass
                
                if hasattr(self.env, 'kpi_trackers') and self.env.kpi_trackers[agent_id].measurements:
                    kpis = self.env.kpi_trackers[agent_id].get_current_kpis()
                    eval_kpis[agent_id].append(kpis)
        
        self.epsilon = old_epsilon
        
        eval_metrics: Dict[str, Dict[str, Any]] = {}
        for agent_id in self.agent_ids:
            agent_eval_metrics = {
                'mean_reward': np.mean(eval_rewards[agent_id]),
                'std_reward': np.std(eval_rewards[agent_id]),
                'min_reward': np.min(eval_rewards[agent_id]),
                'max_reward': np.max(eval_rewards[agent_id]),
                'band_usage': eval_band_usage[agent_id]
            }
            
            if eval_kpis[agent_id]:
                avg_kpis = {}
                for key in eval_kpis[agent_id][0].keys():
                    if key != 'timestamp':
                        values = [kpi.get(key, 0) for kpi in eval_kpis[agent_id] if isinstance(kpi.get(key, 0), (int, float))]
                        if values:
                            avg_kpis[f'avg_{key}'] = np.mean(values)
                agent_eval_metrics.update(avg_kpis)
                
            if eval_compliances[agent_id]:
                agent_eval_metrics['docomo_compliance'] = float(np.mean(eval_compliances[agent_id]))
                                                      
            if eval_episode_handovers[agent_id]:
                agent_eval_metrics['avg_episode_handovers'] = float(np.mean(eval_episode_handovers[agent_id]))
            
            eval_metrics[agent_id] = agent_eval_metrics
            
            print(f"\n    Agent {agent_id} Evaluation Results:")
            print(f"        Mean reward: {eval_metrics[agent_id]['mean_reward']:.4f} ± {eval_metrics[agent_id]['std_reward']:.4f}")
            if 'docomo_compliance' in eval_metrics[agent_id]:
                print(f"        DOCOMO compliance: {eval_metrics[agent_id]['docomo_compliance']:.1%}")
        
        return eval_metrics
    
    def _save_best_model(self, episode: int, reward: float, compliance: float):
        """Save the best performing model"""
        save_path = self.output_dir / "best_model.pth"
        torch.save({
            'episode': episode,
            'reward': reward,
            'compliance': compliance,
            'agents_policy_net_state_dicts': {agent_id: agent.policy_net.state_dict() for agent_id, agent in self.agents.items()},
            'agents_target_net_state_dicts': {agent_id: agent.target_net.state_dict() for agent_id, agent in self.agents.items()},
            'agents_optimizer_state_dicts': {agent_id: agent.optimizer.state_dict() for agent_id, agent in self.agents.items()},
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'config': self.config
        }, save_path)
    
    def _save_checkpoint(self, episode: int, reward: float):
        """Save training checkpoint"""
        checkpoint_path = self.output_dir / f"checkpoint_episode_{episode+1}.pth"
        torch.save({
            'episode': episode,
            'reward': reward,
            'agents_policy_net_state_dicts': {agent_id: agent.policy_net.state_dict() for agent_id, agent in self.agents.items()},
            'agents_target_net_state_dicts': {agent_id: agent.target_net.state_dict() for agent_id, agent in self.agents.items()},
            'agents_optimizer_state_dicts': {agent_id: agent.optimizer.state_dict() for agent_id, agent in self.agents.items()},
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'training_metrics': self.training_metrics
        }, checkpoint_path)
    
    def _save_final_results(self, training_time: float, best_reward: float, 
                           best_compliance: float, final_metrics: Dict[str, Any]):
        """Save comprehensive training results"""
        results = {
            'training_info': {
                'episodes': self.args.episodes,
                'training_time': training_time,
                'best_reward': best_reward,
                'best_compliance': best_compliance,
                'device': str(self.device),
                'model_parameters': sum(p.numel() for p in self.agents[self.agent_ids[0]].policy_net.parameters())
            },
            'config': self.config,
            'training_metrics': self.training_metrics,
            'final_evaluation': final_metrics,
            'environment_info': {
                'state_dim': self.env.observation_space[self.agent_ids[0]].shape[0] if self.agent_ids else 0,
                'action_dim': self.env.action_space[self.agent_ids[0]].n if self.agent_ids else 0,
                'frequency_bands': list(self.env.band_names),
                'num_users': self.env.num_users,
                'current_slice': {agent_id: getattr(self.env, 'current_slice_names', {}).get(agent_id, 'N/A') for agent_id in self.agent_ids},
                'current_qos_targets': {agent_id: getattr(self.env, 'current_qos_targets', {}).get(agent_id, {}) for agent_id in self.agent_ids},
            }
        }
        
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
                        
        self._generate_plots()
        
        print(f"\n Results saved to: {self.output_dir}")
        print("    training_results.json - Complete training data")
        print("    best_model.pth - Best performing model") 
        print("    plots/ - Training visualizations")
    
    def _generate_plots(self):
        """Generate training visualization plots"""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
                         
        if self.training_metrics['rewards']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
                     
            episodes = self.training_metrics['episodes']
            rewards = self.training_metrics['rewards']
            axes[0, 0].plot(episodes, rewards, alpha=0.7)
            if len(rewards) >= 2:
                w = min(10, len(rewards))
                smooth = np.convolve(rewards, np.ones(w)/w, mode='same')
                                   
                if len(smooth) != len(episodes):
                    if len(smooth) > len(episodes):
                        smooth = smooth[:len(episodes)]
                    else:
                        smooth = np.pad(smooth, (0, len(episodes)-len(smooth)))
                axes[0, 0].plot(episodes, smooth, 'r-', linewidth=2)
            axes[0, 0].set_title('Training Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
            
                    
            if self.training_metrics['losses']:
                losses = self.training_metrics['losses']
                axes[0, 1].plot(episodes, losses, alpha=0.7)
                if len(losses) >= 2:
                    w = min(10, len(losses))
                    smooth_l = np.convolve(losses, np.ones(w)/w, mode='same')
                    if len(smooth_l) != len(episodes):
                        if len(smooth_l) > len(episodes):
                            smooth_l = smooth_l[:len(episodes)]
                        else:
                            smooth_l = np.pad(smooth_l, (0, len(episodes)-len(smooth_l)))
                    axes[0, 1].plot(episodes, smooth_l, 'r-', linewidth=2)
                axes[0, 1].set_title('Training Losses')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True)
            
                             
            # Prefer episode-average KPIs if available; fallback to last-step KPIs
            if self.training_metrics['avg_throughput_gbps']:
                throughputs = self.training_metrics['avg_throughput_gbps']
                latencies = self.training_metrics['avg_latency_ms'] if self.training_metrics['avg_latency_ms'] else []
            else:
                kpis = self.training_metrics['kpis'] if self.training_metrics['kpis'] else []
                throughputs = [k.get('current_throughput_gbps', 0.0) for k in kpis]
                latencies = [k.get('current_latency_ms', 0.0) for k in kpis]

            ax_kpi = axes[1, 0]
            if throughputs:
                ax_kpi.plot(episodes[:len(throughputs)], throughputs, 'g-', label='Avg Throughput (Gbps)')
            ax_kpi.axhline(y=100, color='g', linestyle='--', alpha=0.7, label='Target: 100 Gbps')
            ax_kpi.set_ylabel('Throughput (Gbps)', color='g')
            ax_kpi.tick_params(axis='y', labelcolor='g')

            ax_lat = ax_kpi.twinx()
            if latencies:
                ax_lat.plot(episodes[:len(latencies)], latencies, 'b-', label='Avg Latency (ms)')
            ax_lat.axhline(y=0.1, color='b', linestyle='--', alpha=0.7, label='Target: 0.1 ms')
            ax_lat.set_ylabel('Latency (ms)', color='b')
            ax_lat.tick_params(axis='y', labelcolor='b')

            ax_kpi.set_title('DOCOMO KPI Performance')
            ax_kpi.set_xlabel('Episode')
            ax_kpi.grid(True)
            
                              
            axes[1, 1].set_title('Frequency Band Performance')
            for band, rewards in self.training_metrics['band_performance'].items():
                if rewards:
                    try:
                        freq_val = self.env.frequency_bands[band].get('frequency', 28.0e9)
                        freq_ghz = float(freq_val) / 1e9
                    except Exception:
                        freq_ghz = 28.0
                    axes[1, 1].plot(rewards, label=f'{band} ({freq_ghz:.0f} GHz)', alpha=0.7)
            axes[1, 1].set_xlabel('Usage Instance')
            axes[1, 1].set_ylabel('Avg Throughput per Use (Gbps)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "docomo_training_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("    docomo_training_curves.png - Training progress visualization")

def main():
    """Main training function"""
    args = parse_args()
    
    print(" DOCOMO 6G Professional Training System")
    print("="*60)
    print(f"    Config: {args.config}")
    print(f"    Output: {args.output_dir}")
    print(f"    Episodes: {args.episodes}")
    print(f"    Seed: {args.seed}")
    
                                 
    trainer = DOCOMOTrainingManager(args.config, args.output_dir, args)
    
                    
    final_metrics = trainer.train()
    
    print("\n DOCOMO 6G Training Successfully Completed!")
    print("System ready for production deployment!")
    
    return final_metrics

if __name__ == "__main__":
    main()
