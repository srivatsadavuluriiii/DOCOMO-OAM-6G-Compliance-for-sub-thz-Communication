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
        
                           
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
        print(f"  Using device: {self.device}")
        
                            
        self.config = self._load_config()
        
                                
        print(" Initializing DOCOMO 6G Environment...")
        self.env = DOCOMO_6G_Environment(config_path=config_path)
        print(f"    State space: {self.env.observation_space.shape}")
        print(f"    Action space: {self.env.action_space.n}")
        print(f"    Frequency bands: {len(self.env.band_names)}")
        
                                                     
        print(" Initializing DQN Agent...")
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
                                                             
        agent_config = self.config.get('agent', {})
        self.agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=agent_config.get('hidden_layers', [256, 256]),
            learning_rate=agent_config.get('learning_rate', 1e-4),
            gamma=agent_config.get('gamma', 0.99),
            buffer_capacity=agent_config.get('buffer_capacity', 100000),
            batch_size=agent_config.get('batch_size', 64),
            target_update_freq=agent_config.get('target_update_freq', 10),
            device=self.device
        )
        
                                                         
        self.epsilon_start = agent_config.get('epsilon_start', 1.0)
        self.epsilon_end = agent_config.get('epsilon_end', 0.01)
        self.epsilon_decay_steps = agent_config.get('epsilon_decay_steps', 10000)
        self.epsilon = self.epsilon_start
        self.step_count = 0
        
                              
        self.agent.policy_net.to(self.device)
        self.agent.target_net.to(self.device)
        
        print(f"     Model parameters: {sum(p.numel() for p in self.agent.policy_net.parameters())}")
        
                                     
        self.metrics_logger = MetricsLogger(log_dir=str(self.output_dir))
        self.training_metrics = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'kpis': [],
            'band_performance': {band: [] for band in self.env.band_names}
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
                    'epsilon_end': 0.01,
                                    'epsilon_decay_steps': 10000
            }
        }
    
    def _select_action(self, state: torch.Tensor) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.agent.choose_action(state.cpu().numpy(), epsilon=0.0)
    
    def _decay_epsilon(self):
        """Decay epsilon linearly"""
        if self.step_count < self.epsilon_decay_steps:
            decay_ratio = self.step_count / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay_ratio
        else:
            self.epsilon = self.epsilon_end
    
    def train(self, episodes: Optional[int] = None) -> Dict[str, Any]:
        """Train the DOCOMO 6G agent"""
        if episodes is None:
            episodes = self.args.episodes
            
        print(f"\n Starting DOCOMO 6G Training: {episodes} episodes")
        print("="*60)
        
        training_start = time.time()
        best_reward = float('-inf')
        best_compliance = 0.0
        
        for episode in range(episodes):
            episode_start = time.time()
            
                               
            state, info = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            
            episode_reward = 0
            episode_loss = 0
            steps = 0
            band_usage = {band: 0 for band in self.env.band_names}
            
                          
            while steps < self.args.max_steps:
                                                    
                action = self._select_action(state)
                self.step_count += 1
                
                               
                self._decay_epsilon()
                
                                  
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = torch.FloatTensor(next_state).to(self.device)
                
                                  
                self.agent.replay_buffer.push(
                    state.cpu().numpy(),
                    action,
                    reward,
                    next_state.cpu().numpy(),
                    done or truncated
                )
                
                               
                episode_reward += reward
                if hasattr(self.env, 'current_band'):
                    band_usage[self.env.current_band] += 1
                
                                         
                if self.agent.replay_buffer.is_ready(self.agent.batch_size):
                    loss_value = self.agent.learn()
                    if loss_value is not None and isinstance(loss_value, (int, float)):
                        episode_loss += loss_value
                
                state = next_state
                steps += 1
                
                if done or truncated:
                    break
            
                                                
            if (episode + 1) % self.agent.target_update_freq == 0:
                self.agent.update_target_network()
            
                                     
            self.training_metrics['episodes'].append(episode)
            self.training_metrics['rewards'].append(episode_reward)
            self.training_metrics['losses'].append(episode_loss / max(steps, 1))
            
                                 
            if hasattr(self.env, 'kpi_tracker') and self.env.kpi_tracker.measurements:
                kpis = self.env.kpi_tracker.get_current_kpis()
                self.training_metrics['kpis'].append(kpis)
                
                                             
                compliance = self._calculate_docomo_compliance(kpis)
                if compliance > best_compliance:
                    best_compliance = compliance
            
                                    
            for band, count in band_usage.items():
                if count > 0:
                    self.training_metrics['band_performance'][band].append(episode_reward / count)
            
                                               
            if episode_reward > best_reward:
                best_reward = episode_reward
                self._save_best_model(episode, episode_reward, best_compliance)
            
                     
            if (episode + 1) % self.args.log_interval == 0:
                episode_time = time.time() - episode_start
                
                print(f"Episode {episode+1:4d}: "
                      f"Reward={episode_reward:7.2f}, "
                      f"Loss={episode_loss/max(steps,1):6.4f}, "
                      f"Steps={steps:3d}, "
                      f"ε={self.epsilon:.3f}, "
                      f"Time={episode_time:.2f}s")
                
                                   
                if self.training_metrics['kpis']:
                    latest_kpis = self.training_metrics['kpis'][-1]
                    throughput = latest_kpis.get('current_throughput_gbps', 0)
                    latency = latest_kpis.get('current_latency_ms', 0)
                    reliability = latest_kpis.get('current_reliability', 0)
                    
                    print(f"          KPIs: T={throughput:.1f}Gbps, L={latency:.3f}ms, R={reliability:.4f}")
                    
                                                                    
                    most_used = max(band_usage.items(), key=lambda x: x[1])
                    try:
                        freq_val = self.env.frequency_bands[most_used[0]].get('frequency', 28.0e9)
                        freq_ghz = float(freq_val) / 1e9
                    except Exception:
                        freq_ghz = 28.0
                    print(f"          Band: {most_used[0]} ({freq_ghz:.0f} GHz) - {most_used[1]} uses")

                                                    
                    targets = getattr(self.env.kpi_tracker, 'docomo_targets', None)
                    comp = self.env.kpi_tracker.get_compliance_score() if hasattr(self.env, 'kpi_tracker') else {}
                    overall = comp.get('overall_current', 0.0)
                    if targets:
                        print(
                            "          [DOCOMO] T: "
                            f"{throughput:.1f}/{targets.user_data_rate_gbps:.0f} Gbps | "
                            f"L: {latency:.3f}/{targets.latency_ms:.3f} ms | "
                            f"R: {reliability:.7f}/{targets.reliability:.7f} | "
                            f"M: {latest_kpis.get('current_mobility_kmh', 0):.1f}/{targets.mobility_kmh:.0f} km/h | "
                            f"Compliance: {overall*100:.1f}%"
                        )
            
                            
            if (episode + 1) % self.args.save_interval == 0:
                self._save_checkpoint(episode, episode_reward)
        
        training_time = time.time() - training_start
        
                          
        print("\n" + "="*60)
        print(" DOCOMO 6G Training Completed!")
        if episodes > 0:
            print(f"     Total time: {training_time:.2f}s ({training_time/episodes:.2f}s/episode)")
        else:
            print(f"     Total time: {training_time:.2f}s (evaluation-only)")
        print(f"    Best reward: {best_reward:.4f}")
        print(f"    Best DOCOMO compliance: {best_compliance:.1%}")
        print(f"    Final epsilon: {self.epsilon:.4f}")
        print(f"    Buffer size: {len(self.agent.replay_buffer)}")
        
                          
        final_metrics = self._evaluate_agent(self.args.eval_episodes)
        
                                                                               
        if self.training_metrics['kpis']:
            tps = [k.get('current_throughput_gbps', 0.0) for k in self.training_metrics['kpis']]
            if tps:
                final_metrics['avg_training_throughput_gbps'] = float(np.mean(tps))
                final_metrics['peak_training_throughput_gbps'] = float(np.max(tps))
        
                            
        self._save_final_results(training_time, best_reward, best_compliance, final_metrics)
        
        return final_metrics
    
    def _calculate_docomo_compliance(self, kpis: Dict[str, Any]) -> float:
        """Calculate DOCOMO 6G compliance score using KPI tracker (overall)."""
        if hasattr(self.env, 'kpi_tracker'):
            comp = self.env.kpi_tracker.get_compliance_score()
                                                                
            return float(comp.get('overall_current', comp.get('overall_avg', 0.0)))
        return 0.0
    
    def _evaluate_agent(self, num_episodes: int) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        print(f"\n Evaluating agent over {num_episodes} episodes...")

                                                                                
        try:
            best_path = self.output_dir / "best_model.pth"
            if best_path.exists():
                ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                if 'policy_net_state_dict' in ckpt:
                    self.agent.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
                if 'target_net_state_dict' in ckpt:
                    self.agent.target_net.load_state_dict(ckpt['target_net_state_dict'])
                print(f"    Loaded best model for evaluation: {best_path}")
        except Exception as e:
            print(f"     Could not load best model for evaluation: {e}")
        
        eval_rewards = []
        eval_kpis = []
        eval_band_usage = {band: 0 for band in self.env.band_names}
        eval_compliances = []
        eval_episode_handovers = []
        
                                      
        old_epsilon = self.epsilon
        self.epsilon = 0.0                  
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < self.args.max_steps:
                state_tensor = torch.FloatTensor(state).to(self.device)
                action = self.agent.choose_action(state_tensor.cpu().numpy(), epsilon=0.0)
                
                next_state, reward, done, truncated, info = self.env.step(action)
                
                episode_reward += reward
                if hasattr(self.env, 'current_band'):
                    eval_band_usage[self.env.current_band] += 1
                
                state = next_state
                steps += 1
                
                if done or truncated:
                    break
            
            eval_rewards.append(episode_reward)
                                                         
            if hasattr(self.env, 'kpi_tracker'):
                comp = self.env.kpi_tracker.get_compliance_score()
                if 'overall_current' in comp:
                    eval_compliances.append(float(comp['overall_current']))
                                                                         
            try:
                if isinstance(getattr(self.env, 'episode_stats', None), dict):
                    eval_episode_handovers.append(int(self.env.episode_stats.get('handover_count', 0)))
            except Exception:
                pass
            
                          
            if hasattr(self.env, 'kpi_tracker') and self.env.kpi_tracker.measurements:
                kpis = self.env.kpi_tracker.get_current_kpis()
                eval_kpis.append(kpis)
        
                                  
        self.epsilon = old_epsilon
        
                                      
        eval_metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'band_usage': eval_band_usage
        }
        
        if eval_kpis:
                                     
            avg_kpis = {}
            for key in eval_kpis[0].keys():
                if key != 'timestamp':
                    values = [kpi.get(key, 0) for kpi in eval_kpis if isinstance(kpi.get(key, 0), (int, float))]
                    if values:
                        avg_kpis[f'avg_{key}'] = np.mean(values)
            
            eval_metrics.update(avg_kpis)
            
                                                                         
            if eval_compliances:
                eval_metrics['docomo_compliance'] = float(np.mean(eval_compliances))
                                                      
        if eval_episode_handovers:
            eval_metrics['avg_episode_handovers'] = float(np.mean(eval_episode_handovers))
        
        print(f"    Mean reward: {eval_metrics['mean_reward']:.4f} ± {eval_metrics['std_reward']:.4f}")
        if 'docomo_compliance' in eval_metrics:
            print(f"    DOCOMO compliance: {eval_metrics['docomo_compliance']:.1%}")
        
        return eval_metrics
    
    def _save_best_model(self, episode: int, reward: float, compliance: float):
        """Save the best performing model"""
        save_path = self.output_dir / "best_model.pth"
        torch.save({
            'episode': episode,
            'reward': reward,
            'compliance': compliance,
            'policy_net_state_dict': self.agent.policy_net.state_dict(),
            'target_net_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
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
            'policy_net_state_dict': self.agent.policy_net.state_dict(),
            'target_net_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
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
                'model_parameters': sum(p.numel() for p in self.agent.policy_net.parameters())
            },
            'config': self.config,
            'training_metrics': self.training_metrics,
            'final_evaluation': final_metrics,
            'environment_info': {
                'state_dim': self.env.observation_space.shape[0],
                'action_dim': self.env.action_space.n,
                'frequency_bands': list(self.env.band_names)
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
            
                             
            if self.training_metrics['kpis']:
                kpis = self.training_metrics['kpis']
                throughputs = [kpi.get('current_throughput_gbps', 0) for kpi in kpis]
                latencies = [kpi.get('current_latency_ms', 0) for kpi in kpis]
                
                ax_kpi = axes[1, 0]
                ax_kpi.plot(episodes[:len(throughputs)], throughputs, 'g-', label='Throughput (Gbps)')
                ax_kpi.axhline(y=100, color='g', linestyle='--', alpha=0.7, label='Target: 100 Gbps')
                ax_kpi.set_ylabel('Throughput (Gbps)', color='g')
                ax_kpi.tick_params(axis='y', labelcolor='g')
                
                ax_lat = ax_kpi.twinx()
                ax_lat.plot(episodes[:len(latencies)], latencies, 'b-', label='Latency (ms)')
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
            axes[1, 1].set_ylabel('Reward per Use')
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
