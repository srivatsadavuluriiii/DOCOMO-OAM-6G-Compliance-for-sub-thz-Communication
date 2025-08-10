#!/usr/bin/env python3
import os
import argparse
from typing import List, Dict, Any, Optional
import sys

# Use centralized path management instead of sys.path.append
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

# Import hierarchical config system
from utils.hierarchical_config import load_hierarchical_config

from scripts.training.train_rl import train as train_agent
from scripts.training.train_stable_rl import train as train_stable_agent
from scripts.evaluation.evaluate_rl import evaluate as evaluate_agent




def parse_args(args: List[str] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OAM Handover with Deep Q-Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a DQN agent")
    train_parser.add_argument("--config", type=str, default="rl_config_new", help="Path to RL configuration (hierarchical name or legacy .yaml file)")
    train_parser.add_argument("--sim-config", type=str, default="config/simulation_params.yaml", help="Path to simulation configuration file (only used with legacy configs)")
    train_parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    train_parser.add_argument("--log-interval", type=int, default=10, help="Episodes between logging")
    train_parser.add_argument("--eval-interval", type=int, default=100, help="Episodes between evaluations")
    train_parser.add_argument("--num-episodes", type=int, default=None, help="Override number of episodes from config")
    train_parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per episode from config")
    
    # Train with stable rewards command
    train_stable_parser = subparsers.add_parser("train-stable", help="Train a DQN agent with stable rewards")
    train_stable_parser.add_argument("--config", type=str, default="rl_config_new", help="Path to RL configuration (hierarchical name or legacy .yaml file)")
    train_stable_parser.add_argument("--sim-config", type=str, default="config/simulation_params.yaml", help="Path to simulation configuration file (only used with legacy configs)")
    train_stable_parser.add_argument("--stable-config", type=str, default="stable_reward_config_new", help="Path to stable reward configuration (hierarchical name or legacy .yaml file)")
    train_stable_parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    train_stable_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_stable_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    train_stable_parser.add_argument("--log-interval", type=int, default=10, help="Episodes between logging")
    train_stable_parser.add_argument("--eval-interval", type=int, default=100, help="Episodes between evaluations")
    train_stable_parser.add_argument("--num-episodes", type=int, default=None, help="Override number of episodes from config")
    train_stable_parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per episode from config")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained DQN agent")
    eval_parser.add_argument("--model-dir", type=str, required=True, help="Directory containing the trained model")
    eval_parser.add_argument("--config", type=str, default=None, help="Path to configuration file (defaults to config.yaml in model directory)")
    eval_parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation results (defaults to model directory)")
    eval_parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    eval_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    eval_parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    eval_parser.add_argument("--verbose", action="store_true", help="Print detailed information during evaluation")
    
    # Parse arguments
    if args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args)


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "train":
        train_agent(args)
    elif args.command == "train-stable":
        train_stable_agent(args)
    elif args.command == "evaluate":
        evaluate_agent(args)
    else:
        print("Please specify a command: train, train-stable, or evaluate")
        print("Example: python main.py train")
        print("Example: python main.py train-stable")
        print("Example: python main.py evaluate --model-dir results/train_20230101_120000")
        sys.exit(1)


if __name__ == "__main__":
    main() 