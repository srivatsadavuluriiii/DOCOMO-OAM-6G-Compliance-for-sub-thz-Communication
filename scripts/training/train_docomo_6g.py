#!/usr/bin/env python3
"""
DOCOMO 6G Training Script - Professional Integration
Train DQN agent for DOCOMO 6G OAM optimization using existing project infrastructure
Targets: 1 Tbps peak, 0.1ms latency, 500 km/h mobility, 99.99999% reliability
"""

import os
import sys
import yaml
import numpy as np
import time
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Ensure project root is on sys.path before importing project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

# Import existing project infrastructure
from environment.docomo_6g.docomo_6g_env import DOCOMO_6G_Environment

print("✅ Successfully imported all DOCOMO 6G components with existing infrastructure")

# Standard imports
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict

class DOCOMOTrainingManager:
    """
    DOCOMO 6G training management system
    Handles multi-objective optimization and DOCOMO KPI tracking
    """
    
    def __init__(self, config_path: str, results_dir: str):
        """Initialize DOCOMO training manager"""
        
        self.config_path = config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.docomo_config = self.config.get('docomo_6g_system', {})
        self.training_config = self.docomo_config.get('reinforcement_learning', {})
        
        # Initialize environment
        self.env = DOCOMO_6G_Environment(config=self.config)
        
        # Training parameters
        self.num_episodes = self.config.get('simulation', {}).get('max_episodes', 10000)
        self.max_steps_per_episode = self.config.get('simulation', {}).get('max_steps_per_episode', 2000)
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=1000)
        self.episode_throughput = deque(maxlen=1000)
        self.episode_latency = deque(maxlen=1000)
        self.episode_compliance = deque(maxlen=1000)
        self.episode_mobility = deque(maxlen=1000)
        
        # DOCOMO KPI tracking
        self.docomo_kpi_history = defaultdict(list)
        self.training_stats = {
            'start_time': datetime.now(),
            'total_episodes': 0,
            'total_steps': 0,
            'best_compliance_score': 0.0,
            'best_throughput_gbps': 0.0,
            'min_latency_ms': float('inf'),
            'max_mobility_kmh': 0.0
        }
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"docomo_6g_run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        print(f"🚀 DOCOMO 6G Training Manager Initialized")
        print(f"   📁 Results Directory: {self.run_dir}")
        print(f"   🎯 Target Episodes: {self.num_episodes}")
        print(f"   📊 DOCOMO KPI Targets:")
        print(f"      - Peak Data Rate: {self.docomo_config.get('kpi_targets', {}).get('peak_data_rate_tbps', 1.0)} Tbps")
        print(f"      - User Data Rate: {self.docomo_config.get('kpi_targets', {}).get('user_data_rate_gbps', 100.0)} Gbps")
        print(f"      - Latency: {self.docomo_config.get('kpi_targets', {}).get('latency_ms', 0.1)} ms")
        print(f"      - Mobility: {self.docomo_config.get('kpi_targets', {}).get('mobility_kmh', 500.0)} km/h")
        print(f"      - Reliability: {self.docomo_config.get('kpi_targets', {}).get('reliability', 0.9999999)}")
    
    def setup_logging(self):
        """Setup training logs"""
        self.log_file = self.run_dir / "training_log.txt"
        self.metrics_file = self.run_dir / "metrics.json"
        self.kpi_file = self.run_dir / "docomo_kpis.json"
        
        # Copy configuration to run directory
        import shutil
        shutil.copy2(self.config_path, self.run_dir / "config.yaml")
    
    def log_message(self, message: str, level: str = "INFO"):
        """Log training message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)
        
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")
    
    def train_agent(self):
        """
        Main training loop for DOCOMO 6G agent
        """
        self.log_message("🎯 Starting DOCOMO 6G training")
        
        # Training loop
        for episode in range(self.num_episodes):
            episode_start_time = time.time()
            
            # Reset environment
            state, info = self.env.reset()
            
            # Episode tracking
            episode_reward = 0.0
            episode_steps = 0
            episode_metrics = {
                'throughput_samples': [],
                'latency_samples': [],
                'mobility_samples': [],
                'compliance_samples': [],
                'energy_samples': []
            }
            
            # Episode loop
            for step in range(self.max_steps_per_episode):
                # Simple policy for now (random actions)
                # In full implementation, this would be the trained agent
                action = self._select_action(state, episode)
                
                # Execute step
                next_state, reward, terminated, truncated, step_info = self.env.step(action)
                
                # Track metrics
                perf_metrics = step_info.get('performance_metrics', {})
                episode_metrics['throughput_samples'].append(perf_metrics.get('throughput_gbps', 0))
                episode_metrics['latency_samples'].append(perf_metrics.get('latency_ms', 0))
                episode_metrics['mobility_samples'].append(perf_metrics.get('mobility_kmh', 0))
                
                compliance_scores = step_info.get('compliance_scores', {})
                episode_metrics['compliance_samples'].append(compliance_scores.get('overall_current', 0))
                episode_metrics['energy_samples'].append(perf_metrics.get('energy_consumption_w', 0))
                
                # Update episode totals
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                # Check for early termination
                if terminated or truncated:
                    break
            
            # Episode completed - process results
            episode_duration = time.time() - episode_start_time
            self._process_episode_results(episode, episode_reward, episode_steps, 
                                        episode_metrics, episode_duration, step_info)
            
            # Periodic evaluation and checkpointing
            if (episode + 1) % 100 == 0:
                self._evaluate_and_checkpoint(episode + 1)
            
            # Early stopping check
            if self._check_early_stopping(episode):
                self.log_message(f"🛑 Early stopping at episode {episode}")
                break
        
        # Training completed
        self._finalize_training()
    
    def _select_action(self, state: np.ndarray, episode: int) -> int:
        """
        Select action using simple policy (placeholder for trained agent)
        """
        # Simple heuristic policy for demonstration
        # In full implementation, this would use trained neural network
        
        distance = state[3] if len(state) > 3 else 100.0
        throughput = state[1] if len(state) > 1 else 0.0
        latency = state[2] if len(state) > 2 else 1.0
        
        # Distance-based band selection
        if distance < 50:
            if np.random.random() < 0.3:  # 30% chance to switch to higher band
                return 3  # BAND_UP
        elif distance > 200:
            if np.random.random() < 0.3:  # 30% chance to switch to lower band
                return 4  # BAND_DOWN
        
        # Throughput-based OAM mode selection
        if throughput < 50.0:  # Low throughput, try higher mode
            if np.random.random() < 0.2:
                return 1  # OAM_UP
        elif throughput > 150.0:  # High throughput, maybe stabilize
            if np.random.random() < 0.1:
                return 0  # STAY
        
        # High mobility - enable prediction and tracking
        velocity_mag = np.sqrt(state[4]**2 + state[5]**2 + state[6]**2) * 3.6 if len(state) > 6 else 0
        if velocity_mag > 100.0:  # High speed
            if np.random.random() < 0.4:
                return np.random.choice([5, 6])  # BEAM_TRACK or PREDICT
        
        # Default: mostly stay or small adjustments
        return np.random.choice([0, 0, 0, 1, 2], p=[0.6, 0.1, 0.1, 0.1, 0.1])
    
    def _process_episode_results(self, episode: int, reward: float, steps: int, 
                               metrics: Dict[str, List], duration: float, final_info: Dict):
        """Process and log episode results"""
        
        # Calculate episode statistics
        avg_throughput = np.mean(metrics['throughput_samples']) if metrics['throughput_samples'] else 0
        avg_latency = np.mean(metrics['latency_samples']) if metrics['latency_samples'] else 0
        avg_mobility = np.mean(metrics['mobility_samples']) if metrics['mobility_samples'] else 0
        avg_compliance = np.mean(metrics['compliance_samples']) if metrics['compliance_samples'] else 0
        avg_energy = np.mean(metrics['energy_samples']) if metrics['energy_samples'] else 0
        
        # Update history
        self.episode_rewards.append(reward)
        self.episode_throughput.append(avg_throughput)
        self.episode_latency.append(avg_latency)
        self.episode_compliance.append(avg_compliance)
        self.episode_mobility.append(avg_mobility)
        
        # Update training statistics
        self.training_stats['total_episodes'] += 1
        self.training_stats['total_steps'] += steps
        self.training_stats['best_compliance_score'] = max(
            self.training_stats['best_compliance_score'], avg_compliance
        )
        self.training_stats['best_throughput_gbps'] = max(
            self.training_stats['best_throughput_gbps'], avg_throughput
        )
        if avg_latency > 0:
            self.training_stats['min_latency_ms'] = min(
                self.training_stats['min_latency_ms'], avg_latency
            )
        self.training_stats['max_mobility_kmh'] = max(
            self.training_stats['max_mobility_kmh'], avg_mobility
        )
        
        # Track DOCOMO KPIs
        kpi_targets = self.docomo_config.get('kpi_targets', {})
        throughput_achievement = avg_throughput / kpi_targets.get('user_data_rate_gbps', 100.0)
        latency_achievement = kpi_targets.get('latency_ms', 0.1) / max(avg_latency, 0.001)
        mobility_achievement = avg_mobility / kpi_targets.get('mobility_kmh', 500.0)
        
        self.docomo_kpi_history['throughput_achievement'].append(throughput_achievement)
        self.docomo_kpi_history['latency_achievement'].append(latency_achievement)
        self.docomo_kpi_history['mobility_achievement'].append(mobility_achievement)
        self.docomo_kpi_history['compliance_score'].append(avg_compliance)
        self.docomo_kpi_history['energy_efficiency'].append(avg_throughput / max(avg_energy, 0.1))
        
        # Periodic logging
        if episode % 50 == 0 or episode < 10:
            self.log_message(
                f"📊 Episode {episode:4d}: "
                f"Reward={reward:8.2f}, Steps={steps:4d}, "
                f"Throughput={avg_throughput:6.2f}Gbps, "
                f"Latency={avg_latency:6.3f}ms, "
                f"Mobility={avg_mobility:6.1f}km/h, "
                f"Compliance={avg_compliance:5.3f}, "
                f"Duration={duration:5.2f}s"
            )
    
    def _evaluate_and_checkpoint(self, episode: int):
        """Perform evaluation and save checkpoint"""
        self.log_message(f"🔍 Evaluation at episode {episode}")
        
        # Calculate recent performance
        recent_episodes = min(100, len(self.episode_rewards))
        if recent_episodes > 0:
            recent_rewards = list(self.episode_rewards)[-recent_episodes:]
            recent_throughput = list(self.episode_throughput)[-recent_episodes:]
            recent_latency = list(self.episode_latency)[-recent_episodes:]
            recent_compliance = list(self.episode_compliance)[-recent_episodes:]
            
            # Performance summary
            performance_summary = {
                'episode': episode,
                'recent_avg_reward': np.mean(recent_rewards),
                'recent_avg_throughput_gbps': np.mean(recent_throughput),
                'recent_avg_latency_ms': np.mean(recent_latency),
                'recent_avg_compliance': np.mean(recent_compliance),
                'recent_throughput_std': np.std(recent_throughput),
                'recent_latency_std': np.std(recent_latency),
                
                # DOCOMO compliance metrics
                'throughput_target_achievement': np.mean(recent_throughput) / self.docomo_config.get('kpi_targets', {}).get('user_data_rate_gbps', 100.0),
                'latency_target_achievement': self.docomo_config.get('kpi_targets', {}).get('latency_ms', 0.1) / max(np.mean(recent_latency), 0.001),
                'docomo_compliant_episodes': sum(1 for c in recent_compliance if c >= 0.95) / len(recent_compliance)
            }
            
            self.log_message(f"   📈 Recent Performance ({recent_episodes} episodes):")
            self.log_message(f"      Average Reward: {performance_summary['recent_avg_reward']:.2f}")
            self.log_message(f"      Average Throughput: {performance_summary['recent_avg_throughput_gbps']:.2f} Gbps")
            self.log_message(f"      Average Latency: {performance_summary['recent_avg_latency_ms']:.3f} ms")
            self.log_message(f"      Average Compliance: {performance_summary['recent_avg_compliance']:.3f}")
            self.log_message(f"      DOCOMO Compliant: {performance_summary['docomo_compliant_episodes']*100:.1f}%")
            
            # Save checkpoint
            checkpoint_path = self.run_dir / f"checkpoint_episode_{episode}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'episode': episode,
                    'performance_summary': performance_summary,
                    'training_stats': self.training_stats,
                    'docomo_kpi_history': dict(self.docomo_kpi_history)
                }, f, indent=2, default=str)
        
        # Generate plots
        self._generate_training_plots(episode)
    
    def _generate_training_plots(self, episode: int):
        """Generate training progress plots"""
        if len(self.episode_rewards) < 10:
            return  # Not enough data for meaningful plots
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'DOCOMO 6G Training Progress - Episode {episode}', fontsize=16)
        
        # Reward evolution
        axes[0,0].plot(self.episode_rewards, alpha=0.7)
        axes[0,0].set_title('Episode Reward')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].grid(True)
        
        # Throughput evolution
        axes[0,1].plot(self.episode_throughput, alpha=0.7, color='green')
        axes[0,1].axhline(y=self.docomo_config.get('kpi_targets', {}).get('user_data_rate_gbps', 100.0), 
                         color='red', linestyle='--', label='DOCOMO Target')
        axes[0,1].set_title('Average Throughput per Episode')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Throughput (Gbps)')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Latency evolution
        axes[0,2].plot(self.episode_latency, alpha=0.7, color='orange')
        axes[0,2].axhline(y=self.docomo_config.get('kpi_targets', {}).get('latency_ms', 0.1), 
                         color='red', linestyle='--', label='DOCOMO Target')
        axes[0,2].set_title('Average Latency per Episode')
        axes[0,2].set_xlabel('Episode')
        axes[0,2].set_ylabel('Latency (ms)')
        axes[0,2].legend()
        axes[0,2].grid(True)
        
        # Compliance score evolution
        axes[1,0].plot(self.episode_compliance, alpha=0.7, color='purple')
        axes[1,0].axhline(y=0.95, color='red', linestyle='--', label='Compliance Threshold')
        axes[1,0].set_title('DOCOMO Compliance Score')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Compliance Score')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Mobility support
        axes[1,1].plot(self.episode_mobility, alpha=0.7, color='brown')
        axes[1,1].axhline(y=self.docomo_config.get('kpi_targets', {}).get('mobility_kmh', 500.0), 
                         color='red', linestyle='--', label='DOCOMO Target')
        axes[1,1].set_title('Average Mobility Support')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Mobility (km/h)')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # KPI achievement rates
        if len(self.docomo_kpi_history['throughput_achievement']) > 0:
            axes[1,2].plot(self.docomo_kpi_history['throughput_achievement'], label='Throughput', alpha=0.7)
            axes[1,2].plot(self.docomo_kpi_history['latency_achievement'], label='Latency', alpha=0.7)
            axes[1,2].plot(self.docomo_kpi_history['mobility_achievement'], label='Mobility', alpha=0.7)
            axes[1,2].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            axes[1,2].set_title('DOCOMO KPI Achievement Rates')
            axes[1,2].set_xlabel('Episode')
            axes[1,2].set_ylabel('Achievement Ratio')
            axes[1,2].legend()
            axes[1,2].grid(True)
        
        plt.tight_layout()
        
        # Save plots
        plot_path = self.run_dir / f"training_progress_episode_{episode}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _check_early_stopping(self, episode: int) -> bool:
        """Check if early stopping criteria are met"""
        if episode < 1000:  # Minimum episodes before early stopping
            return False
        
        # Check if performance has plateaued
        if len(self.episode_compliance) >= 500:
            recent_compliance = list(self.episode_compliance)[-200:]  # Last 200 episodes
            older_compliance = list(self.episode_compliance)[-500:-200]  # Previous 300 episodes
            
            recent_avg = np.mean(recent_compliance)
            older_avg = np.mean(older_compliance)
            
            # Early stopping if no significant improvement and high compliance
            if recent_avg >= 0.98 and abs(recent_avg - older_avg) < 0.01:
                return True
        
        return False
    
    def _finalize_training(self):
        """Finalize training and generate final report"""
        self.log_message("🎉 Training completed!")
        
        # Final statistics
        training_duration = datetime.now() - self.training_stats['start_time']
        
        final_report = {
            'training_summary': {
                'total_episodes': self.training_stats['total_episodes'],
                'total_steps': self.training_stats['total_steps'],
                'training_duration': str(training_duration),
                'episodes_per_hour': self.training_stats['total_episodes'] / (training_duration.total_seconds() / 3600)
            },
            'docomo_kpi_achievements': {
                'best_compliance_score': self.training_stats['best_compliance_score'],
                'best_throughput_gbps': self.training_stats['best_throughput_gbps'],
                'min_latency_ms': self.training_stats['min_latency_ms'],
                'max_mobility_kmh': self.training_stats['max_mobility_kmh']
            },
            'final_performance': {
                'avg_reward_last_100': np.mean(list(self.episode_rewards)[-100:]) if len(self.episode_rewards) >= 100 else 0,
                'avg_throughput_last_100': np.mean(list(self.episode_throughput)[-100:]) if len(self.episode_throughput) >= 100 else 0,
                'avg_latency_last_100': np.mean(list(self.episode_latency)[-100:]) if len(self.episode_latency) >= 100 else 0,
                'avg_compliance_last_100': np.mean(list(self.episode_compliance)[-100:]) if len(self.episode_compliance) >= 100 else 0
            },
            'docomo_compliance_analysis': {
                'total_compliant_episodes': sum(1 for c in self.episode_compliance if c >= 0.95),
                'compliance_rate': sum(1 for c in self.episode_compliance if c >= 0.95) / len(self.episode_compliance) if self.episode_compliance else 0,
                'throughput_target_achievements': sum(1 for t in self.episode_throughput if t >= self.docomo_config.get('kpi_targets', {}).get('user_data_rate_gbps', 100.0)),
                'latency_target_achievements': sum(1 for l in self.episode_latency if l <= self.docomo_config.get('kpi_targets', {}).get('latency_ms', 0.1))
            }
        }
        
        # Save final report
        final_report_path = self.run_dir / "final_training_report.json"
        with open(final_report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Generate final plots
        self._generate_training_plots(self.training_stats['total_episodes'])
        
        # Log final summary
        self.log_message("📊 Final Training Summary:")
        self.log_message(f"   🎯 Episodes: {final_report['training_summary']['total_episodes']}")
        self.log_message(f"   ⏱️ Duration: {final_report['training_summary']['training_duration']}")
        self.log_message(f"   🏆 Best Compliance: {final_report['docomo_kpi_achievements']['best_compliance_score']:.3f}")
        self.log_message(f"   📈 Best Throughput: {final_report['docomo_kpi_achievements']['best_throughput_gbps']:.2f} Gbps")
        self.log_message(f"   ⚡ Min Latency: {final_report['docomo_kpi_achievements']['min_latency_ms']:.3f} ms")
        self.log_message(f"   🚀 Max Mobility: {final_report['docomo_kpi_achievements']['max_mobility_kmh']:.1f} km/h")
        self.log_message(f"   ✅ Compliance Rate: {final_report['docomo_compliance_analysis']['compliance_rate']*100:.1f}%")
        
        print(f"\n🎉 DOCOMO 6G Training Complete!")
        print(f"📁 Results saved to: {self.run_dir}")
        print(f"📊 Final report: {final_report_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="DOCOMO 6G Training Script")
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to DOCOMO 6G configuration file')
    parser.add_argument('--results-dir', type=str, default='results/docomo_6g',
                       help='Directory to save training results')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Override number of training episodes')
    
    args = parser.parse_args()
    
    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    print("🚀 Starting DOCOMO 6G Training")
    print(f"📁 Config: {args.config}")
    print(f"📁 Results: {args.results_dir}")
    
    # Initialize training manager
    trainer = DOCOMOTrainingManager(args.config, args.results_dir)
    
    # Override episodes if specified
    if args.episodes is not None:
        trainer.num_episodes = args.episodes
        print(f"🎯 Override episodes: {args.episodes}")
    
    try:
        # Start training
        trainer.train_agent()
        
    except KeyboardInterrupt:
        trainer.log_message("🛑 Training interrupted by user")
        trainer._finalize_training()
        
    except Exception as e:
        trainer.log_message(f"❌ Training failed with error: {str(e)}", level="ERROR")
        raise

if __name__ == "__main__":
    main()
