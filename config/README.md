# System Configuration Files

This directory contains system configuration files for the OAM 6G project.

## Files

### Base Configuration
- **`base_config_new.yaml`**: Main base configuration with common parameters
- **`extended_config_new.yaml`**: Extended configuration with additional parameters
- **`simulation_params.yaml`**: Simulation-specific parameters

### RL Configuration
- **`rl_config_new.yaml`**: Reinforcement learning configuration
- **`stable_reward_config_new.yaml`**: Stable reward function configuration
- **`stable_reward_params.yaml`**: Stable reward parameters

### Distance Optimization
- **`distance_optimization_config.yaml`**: Distance optimization specific configuration

### Requirements
- **`requirements.txt`**: Python package dependencies

## Usage

### Loading Configurations
```python
from utils.config_utils import load_config

# Load base configuration
base_config = load_config('config/base_config_new.yaml')

# Load specific configuration
rl_config = load_config('config/rl_config_new.yaml')

# Load distance optimization configuration
distance_config = load_config('config/distance_optimization_config.yaml')
```

### Configuration Hierarchy
1. **Base Config**: Common parameters across all simulations
2. **Extended Config**: Additional parameters for complex scenarios
3. **RL Config**: Reinforcement learning specific parameters
4. **Distance Config**: Distance optimization parameters

## Configuration Structure

Each configuration file contains:
- **System Parameters**: Frequency, bandwidth, power settings
- **OAM Parameters**: Mode ranges, beam characteristics
- **Environment Parameters**: Turbulence, mobility settings
- **Training Parameters**: Learning rates, episode settings
- **Reward Parameters**: Reward function weights and penalties 