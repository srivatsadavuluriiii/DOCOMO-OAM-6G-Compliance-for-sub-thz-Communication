
import optuna
import yaml
import subprocess
import os
import json
from typing import Dict, Any

# Define the search space for hyperparameters
def define_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the hyperparameter search space using Optuna's trial object."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),
        'epsilon_decay': trial.suggest_float('epsilon_decay', 0.99, 0.9999),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'replay_buffer_size': trial.suggest_categorical('replay_buffer_size', [10000, 50000, 100000]),
        'target_update_frequency': trial.suggest_categorical('target_update_frequency', [100, 500, 1000]),
        # Use strings for hidden_layers to avoid Optuna warnings and ensure serializability
        'hidden_layers': trial.suggest_categorical('hidden_layers', ['128,128', '256,256', '512,256,128'])
    }
    # Convert the hidden_layers string back to a list of ints for the config
    params['hidden_layers'] = [int(x) for x in params['hidden_layers'].split(',')]
    return params

# Objective function for Optuna to optimize
def objective(trial: optuna.Trial) -> float:
    """
    The objective function to be maximized by Optuna.
    Trains an agent with a given set of hyperparameters and returns its performance.
    """
    params = define_search_space(trial)
    
    # Create a temporary config file for this trial
    temp_config_path = f"config/temp_trial_{trial.number}.yaml"
    
    # Load the base config
    with open("config/config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)
        
    # Robustly create nested config structure if it doesn't exist
    if 'docomo_6g_system' not in base_config:
        base_config['docomo_6g_system'] = {}
    if 'reinforcement_learning' not in base_config['docomo_6g_system']:
        base_config['docomo_6g_system']['reinforcement_learning'] = {}
    if 'training' not in base_config['docomo_6g_system']['reinforcement_learning']:
        base_config['docomo_6g_system']['reinforcement_learning']['training'] = {}
        
    # Update the config with the suggested hyperparameters
    rl_training_config = base_config.get('docomo_6g_system', {}).get('reinforcement_learning', {}).get('training', {})
    rl_training_config.update(params)
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(base_config, f)
        
    # Define output directory for this trial
    output_dir = f"results/tuning/trial_{trial.number}"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Run Training ---
    training_command = [
        "python", "scripts/training/train_compliance.py",
        "--config", temp_config_path,
        "--output-dir", output_dir,
        "--episodes", "500",  # Shorter training for faster tuning
        "--max-steps", "300",
        "--no-gpu" # Use CPU for parallel trials
    ]
    
    print(f"--- Starting Trial {trial.number} with params: {params} ---")
    try:
        training_result = subprocess.run(training_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training for Trial {trial.number}:")
        print(f"  Command: {e.cmd}")
        print(f"  Return Code: {e.returncode}")
        print(f"  Stdout:\n{e.stdout}")
        print(f"  Stderr:\n{e.stderr}")
        raise # Re-raise the exception to mark the trial as failed
    
    # --- Run Evaluation ---
    eval_output_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    evaluation_command = [
        "python", "scripts/evaluation/evaluate_rl.py",
        "--model-dir", output_dir,
        "--config", temp_config_path,
        "--output-dir", eval_output_dir,
        "--episodes", "50" # Evaluate over 50 episodes for a stable score
    ]
    
    eval_result = subprocess.run(evaluation_command, check=True, capture_output=True, text=True)
    
    # --- Parse Evaluation Results ---
    # Find the line with "Overall Avg Throughput" and extract the value
    avg_throughput = 0.0
    for line in eval_result.stdout.splitlines():
        if "Overall Avg Throughput" in line:
            # Expected format: "  Overall Avg Throughput (per-episode): X.XXe+XX bps"
            parts = line.split(":")
            if len(parts) > 1:
                throughput_str = parts[1].strip().split(" ")[0]
                try:
                    avg_throughput = float(throughput_str)
                except ValueError:
                    print(f"Warning: Could not parse throughput from line: {line}")
                    avg_throughput = 0.0
    
    # Clean up the temporary config file
    os.remove(temp_config_path)
    
    print(f"--- Trial {trial.number} finished. Average Throughput: {avg_throughput / 1e9:.2f} Gbps ---")
    
    return avg_throughput

# Main execution block
if __name__ == "__main__":
    study_name = "docomo-6g-agent-tuning"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )
    
    print(f"Starting/Resuming study '{study_name}'. Using storage: '{storage_name}'")
    
    try:
        study.optimize(objective, n_trials=100)
    except KeyboardInterrupt:
        print("Study interrupted by user. Saving current results.")

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"Best trial number: {best_trial.number}")
    print(f"Best value (Average Throughput): {best_trial.value / 1e9:.2f} Gbps")
    
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # You can visualize the results with:
    # optuna-dashboard sqlite:///docomo-6g-agent-tuning.db
    print("\nTo visualize the results, run the following command:")
    print(f"pip install optuna-dashboard")
    print(f"optuna-dashboard {storage_name}")
