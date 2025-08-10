# OAM 6G API Documentation

This document provides comprehensive API documentation for the OAM 6G reinforcement learning system.

## Table of Contents

1. [Environment API](#environment-api)
2. [Agent API](#agent-api)
3. [Simulator API](#simulator-api)
4. [Configuration API](#configuration-api)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## Environment API

### OAM_Env

The main Gymnasium environment for OAM mode handover decisions.

#### Constructor

```python
OAM_Env(config: Optional[Dict[str, Any]] = None, simulator: Optional[SimulatorInterface] = None)
```

**Parameters:**
- `config` (Optional[Dict[str, Any]]): Configuration dictionary containing environment parameters. If None, uses default values.
- `simulator` (Optional[SimulatorInterface]): Channel simulator instance for dependency injection. If None, creates default ChannelSimulator.

**Returns:**
- `OAM_Env`: Initialized environment instance.

**Example:**
```python
# Basic usage with default configuration
env = OAM_Env()

# With custom configuration
config = {
    'system': {'frequency': 28.0e9, 'bandwidth': 400e6},
    'oam': {'min_mode': 1, 'max_mode': 6}
}
env = OAM_Env(config)

# With custom simulator for testing
from environment.mock_simulator import MockChannelSimulator
simulator = MockChannelSimulator(config)
env = OAM_Env(config, simulator)
```

#### Methods

##### reset()

```python
reset(seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]
```

Resets the environment to initial state.

**Parameters:**
- `seed` (Optional[int]): Random seed for reproducible initialization. Default: None.
- `options` (Optional[Dict[str, Any]]): Additional reset options. Default: None.

**Returns:**
- `Tuple[np.ndarray, Dict[str, Any]]`: (initial_state, info_dict)
  - `initial_state`: 8-dimensional state vector [SINR, distance, vx, vy, vz, current_mode, min_mode, max_mode]
  - `info_dict`: Additional information including position, velocity, throughput, handovers

**Example:**
```python
obs, info = env.reset(seed=42)
print(f"Initial SINR: {obs[0]:.2f} dB")
print(f"Initial position: {info['position']}")
```

##### step()

```python
step(action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
```

Takes an action in the environment.

**Parameters:**
- `action` (int): Action to take (0: STAY, 1: UP, 2: DOWN)

**Returns:**
- `Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]`: (next_state, reward, terminated, truncated, info)
  - `next_state`: Updated state vector
  - `reward`: Reward value for the action
  - `terminated`: True if episode ended naturally
  - `truncated`: True if episode was artificially terminated
  - `info`: Additional information including SINR, mode, throughput, handovers

**Example:**
```python
action = env.action_space.sample()  # Random action
next_obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward:.3f}")
print(f"Current mode: {info['mode']}")
```

#### State Space

The environment uses an 8-dimensional state space:

```python
state = [SINR, distance, velocity_x, velocity_y, velocity_z, current_mode, min_mode, max_mode]
```

**Components:**
- `SINR` (float): Signal-to-Interference-plus-Noise Ratio in dB [-30, 50]
- `distance` (float): Distance from transmitter in meters [50, 300]
- `velocity_x/y/z` (float): User velocity components in m/s [-5, 5]
- `current_mode` (int): Current OAM mode [min_mode, max_mode]
- `min_mode` (int): Minimum available OAM mode
- `max_mode` (int): Maximum available OAM mode

#### Action Space

Discrete action space with 3 actions:

```python
action_space = spaces.Discrete(3)
```

**Actions:**
- `0`: STAY - Keep current OAM mode
- `1`: UP - Increase OAM mode (if possible)
- `2`: DOWN - Decrease OAM mode (if possible)

---

## Agent API

### Agent

Deep Q-Network (DQN) agent for OAM mode selection.

#### Constructor

```python
Agent(state_dim: int, action_dim: int, hidden_layers: List[int] = [128, 128], 
      learning_rate: float = 1e-4, gamma: float = 0.99, buffer_capacity: int = 50000,
      batch_size: int = 64, target_update_freq: int = 10, device: Optional[torch.device] = None,
      replay_buffer: Optional[ReplayBufferInterface] = None)
```

**Parameters:**
- `state_dim` (int): Dimension of state space (default: 8)
- `action_dim` (int): Number of possible actions (default: 3)
- `hidden_layers` (List[int]): List of hidden layer sizes [128, 128]
- `learning_rate` (float): Learning rate for optimizer (default: 1e-4)
- `gamma` (float): Discount factor for future rewards (default: 0.99)
- `buffer_capacity` (int): Maximum replay buffer size (default: 50000)
- `batch_size` (int): Batch size for training (default: 64)
- `target_update_freq` (int): Episodes between target network updates (default: 10)
- `device` (Optional[torch.device]): Device for computation (CPU/GPU)
- `replay_buffer` (Optional[ReplayBufferInterface]): Custom replay buffer for dependency injection

**Example:**
```python
# Basic agent
agent = Agent(state_dim=8, action_dim=3)

# Custom configuration
agent = Agent(
    state_dim=8,
    action_dim=3,
    hidden_layers=[256, 128, 64],
    learning_rate=1e-3,
    gamma=0.95,
    buffer_capacity=100000,
    batch_size=128
)
```

#### Methods

##### choose_action()

```python
choose_action(state: np.ndarray, epsilon: float = 0.0) -> int
```

Selects an action using epsilon-greedy policy.

**Parameters:**
- `state` (np.ndarray): Current state vector
- `epsilon` (float): Exploration probability (0.0 = greedy, 1.0 = random)

**Returns:**
- `int`: Selected action (0, 1, or 2)

**Example:**
```python
action = agent.choose_action(state, epsilon=0.1)  # 10% exploration
```

##### learn()

```python
learn() -> Optional[Dict[str, float]]
```

Performs one learning step using experience replay.

**Returns:**
- `Optional[Dict[str, float]]`: Training metrics (loss, etc.) or None if buffer too small

**Example:**
```python
metrics = agent.learn()
if metrics:
    print(f"Training loss: {metrics['loss']:.4f}")
```

##### save_model()

```python
save_model(filepath: str) -> None
```

Saves the agent's policy network.

**Parameters:**
- `filepath` (str): Path to save the model

**Example:**
```python
agent.save_model("models/agent_final.pth")
```

##### load_model()

```python
load_model(filepath: str) -> None
```

Loads a saved policy network.

**Parameters:**
- `filepath` (str): Path to the saved model

**Example:**
```python
agent.load_model("models/agent_final.pth")
```

---

## Simulator API

### ChannelSimulator

Physics-based simulator for OAM wireless channels.

#### Constructor

```python
ChannelSimulator(config: Optional[Dict[str, Any]] = None)
```

**Parameters:**
- `config` (Optional[Dict[str, Any]]): Configuration dictionary with physics parameters

**Example:**
```python
config = {
    'system': {
        'frequency': 28.0e9,      # 28 GHz
        'bandwidth': 400e6,       # 400 MHz
        'tx_power_dBm': 30.0,     # 30 dBm
        'noise_figure_dB': 8.0,   # 8 dB
        'noise_temp': 290.0       # 290 K
    },
    'environment': {
        'humidity': 50.0,         # 50%
        'temperature': 20.0,      # 20°C
        'pressure': 101.3,        # 101.3 kPa
        'turbulence_strength': 1e-14,  # m^(-2/3)
        'pointing_error_std': 0.005,   # 5 mrad
        'rician_k_factor': 8.0    # 8 dB
    },
    'oam': {
        'min_mode': 1,            # Minimum OAM mode
        'max_mode': 6,            # Maximum OAM mode
        'beam_width': 0.03        # 30 mrad
    }
}
simulator = ChannelSimulator(config)
```

#### Methods

##### run_step()

```python
run_step(user_position: np.ndarray, current_oam_mode: int) -> Tuple[np.ndarray, float]
```

Simulates one step of the wireless channel.

**Parameters:**
- `user_position` (np.ndarray): 3D position [x, y, z] in meters
- `current_oam_mode` (int): Current OAM mode being used

**Returns:**
- `Tuple[np.ndarray, float]`: (channel_matrix, sinr_dB)
  - `channel_matrix`: Complex channel matrix for all OAM modes
  - `sinr_dB`: Signal-to-Interference-plus-Noise Ratio in dB

**Example:**
```python
position = np.array([100.0, 50.0, 0.0])  # 100m x, 50m y, 0m z
current_mode = 3
H, sinr = simulator.run_step(position, current_mode)
print(f"SINR: {sinr:.2f} dB")
```

---

## Configuration API

### Configuration Structure

The system uses a hierarchical configuration system with the following structure:

```yaml
# Base configuration (config/base_config_new.yaml)
system:
  frequency: 28.0e9        # Hz
  bandwidth: 400e6         # Hz
  tx_power_dBm: 30.0       # dBm
  noise_figure_dB: 8.0     # dB
  noise_temp: 290.0        # K

environment:
  humidity: 50.0           # %
  temperature: 20.0        # °C
  pressure: 101.3          # kPa
  turbulence_strength: 1e-14  # m^(-2/3)
  pointing_error_std: 0.005   # rad
  rician_k_factor: 8.0     # dB

oam:
  min_mode: 1              # Minimum OAM mode
  max_mode: 8              # Maximum OAM mode
  beam_width: 0.03         # rad

rl_base:
  network:
    state_dim: 8           # State dimension
    action_dim: 3          # Action dimension
    hidden_layers: [128, 128]  # Hidden layer sizes
    activation: 'relu'     # Activation function

  replay_buffer:
    capacity: 50000        # Buffer capacity
    min_samples_to_learn: 1000  # Minimum samples for learning

  exploration:
    epsilon_start: 1.0     # Initial exploration rate
    epsilon_end: 0.01      # Final exploration rate
    epsilon_decay: 0.99    # Exploration decay rate

  training:
    num_episodes: 1000     # Number of training episodes
    max_steps_per_episode: 500  # Maximum steps per episode
    batch_size: 128        # Training batch size
    learning_rate: 0.0001  # Learning rate
    gamma: 0.99            # Discount factor
    target_update_freq: 10 # Target network update frequency

reward:
  throughput_factor: 1.0   # Throughput reward weight
  handover_penalty: 0.2    # Handover penalty weight
  outage_penalty: 1.0      # Outage penalty weight
  sinr_threshold: -5.0     # SINR threshold for outage
```

### Configuration Loading

```python
from utils.hierarchical_config import HierarchicalConfig

# Load configuration with inheritance
config_manager = HierarchicalConfig("config")
config = config_manager.load_config_with_inheritance("rl_config_new")

# Validate configuration
is_valid, errors = config_manager.validate_config(config)
if not is_valid:
    print("Configuration errors:", errors)
```

---

## Usage Examples

### Basic Training Loop

```python
import numpy as np
from environment.oam_env import OAM_Env
from models.agent import Agent

# Initialize environment and agent
env = OAM_Env()
agent = Agent(state_dim=8, action_dim=3)

# Training loop
for episode in range(1000):
    obs, _ = env.reset()
    episode_reward = 0
    
    for step in range(500):
        # Choose action
        action = agent.choose_action(obs, epsilon=0.1)
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store experience
        agent.replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
        
        # Learn
        if len(agent.replay_buffer) >= 1000:
            agent.learn()
        
        obs = next_obs
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode}: Reward = {episode_reward:.2f}")
```

### Custom Configuration

```python
# Custom configuration
config = {
    'system': {
        'frequency': 60.0e9,      # 60 GHz
        'bandwidth': 800e6,       # 800 MHz
        'tx_power_dBm': 40.0      # 40 dBm
    },
    'oam': {
        'min_mode': 1,
        'max_mode': 10            # More OAM modes
    },
    'rl_base': {
        'training': {
            'num_episodes': 2000,  # More episodes
            'batch_size': 256      # Larger batch
        }
    }
}

# Initialize with custom config
env = OAM_Env(config)
agent = Agent(state_dim=8, action_dim=3, buffer_capacity=100000)
```

### Evaluation

```python
# Evaluation mode
agent.load_model("models/agent_final.pth")

eval_rewards = []
for episode in range(100):
    obs, _ = env.reset()
    episode_reward = 0
    
    for step in range(500):
        action = agent.choose_action(obs, epsilon=0.0)  # No exploration
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    eval_rewards.append(episode_reward)

print(f"Average evaluation reward: {np.mean(eval_rewards):.2f}")
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'utils'`

**Solution:**
```python
# Add project root to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Or use the path utility
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()
```

#### 2. Configuration Errors

**Problem:** `KeyError: 'system' not found in config`

**Solution:**
```python
# Ensure configuration has required sections
config = {
    'system': {
        'frequency': 28.0e9,
        'bandwidth': 400e6,
        'tx_power_dBm': 30.0,
        'noise_figure_dB': 8.0,
        'noise_temp': 290.0
    },
    'oam': {
        'min_mode': 1,
        'max_mode': 6
    }
}
```

#### 3. Validation Errors

**Problem:** `ValueError: Parameter validation failed`

**Solution:**
```python
# Check parameter ranges
config = {
    'system': {
        'frequency': 28.0e9,      # Must be 1e9 to 1000e9
        'tx_power_dBm': 30.0,     # Must be -20 to 50
        'noise_temp': 290.0       # Must be 50 to 500
    },
    'oam': {
        'min_mode': 1,            # Must be >= 1
        'max_mode': 6,            # Must be > min_mode
        'beam_width': 0.03        # Must be 0.001 to 1.0
    }
}
```

#### 4. Training Issues

**Problem:** Agent not learning (constant loss)

**Solutions:**
```python
# 1. Check learning rate
agent = Agent(learning_rate=1e-3)  # Try higher learning rate

# 2. Check batch size
agent = Agent(batch_size=32)        # Try smaller batch size

# 3. Check exploration
action = agent.choose_action(state, epsilon=0.5)  # Increase exploration

# 4. Check buffer size
agent = Agent(buffer_capacity=10000)  # Ensure sufficient buffer
```

#### 5. Performance Issues

**Problem:** Slow training or high memory usage

**Solutions:**
```python
# 1. Reduce episode length
config['rl_base']['training']['max_steps_per_episode'] = 200

# 2. Reduce buffer size
agent = Agent(buffer_capacity=10000)

# 3. Use smaller network
agent = Agent(hidden_layers=[64, 64])

# 4. Use GPU if available
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = Agent(device=device)
```

#### 6. Environment Issues

**Problem:** Environment produces NaN or infinite values

**Solutions:**
```python
# 1. Check physics parameters
config['environment']['turbulence_strength'] = 1e-14  # Use reasonable values

# 2. Check distance ranges
config['environment']['distance_min'] = 10.0  # Minimum 1m
config['environment']['distance_max'] = 1000.0  # Maximum 10km

# 3. Check frequency ranges
config['system']['frequency'] = 28.0e9  # 1-1000 GHz
```

### Debugging Tips

1. **Enable Validation:**
```python
# Set environment variable for detailed validation
import os
os.environ['OAM_DEBUG'] = '1'
```

2. **Check State Values:**
```python
obs, _ = env.reset()
print(f"State range: [{obs.min():.3f}, {obs.max():.3f}]")
print(f"State components: {obs}")
```

3. **Monitor Training:**
```python
# Add logging
import logging
logging.basicConfig(level=logging.INFO)

# Monitor agent learning
metrics = agent.learn()
if metrics:
    print(f"Loss: {metrics['loss']:.4f}")
```

4. **Validate Physics:**
```python
# Test simulator directly
simulator = ChannelSimulator(config)
position = np.array([100.0, 0.0, 0.0])
H, sinr = simulator.run_step(position, 3)
print(f"SINR: {sinr:.2f} dB")
```

### Performance Optimization

1. **Use GPU:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = Agent(device=device)
```

2. **Optimize Batch Size:**
```python
# Larger batches for GPU, smaller for CPU
batch_size = 256 if torch.cuda.is_available() else 64
agent = Agent(batch_size=batch_size)
```

3. **Reduce Validation:**
```python
# Disable detailed validation for faster training
os.environ['OAM_FAST'] = '1'
```

4. **Use Efficient Data Types:**
```python
# Use float32 for faster computation
obs = obs.astype(np.float32)
```

---

## API Reference Summary

### Key Classes

| Class | Purpose | Main Methods |
|-------|---------|--------------|
| `OAM_Env` | RL Environment | `reset()`, `step()`, `render()` |
| `Agent` | DQN Agent | `choose_action()`, `learn()`, `save_model()` |
| `ChannelSimulator` | Physics Simulator | `run_step()`, `_validate_parameters()` |
| `HierarchicalConfig` | Configuration | `load_config_with_inheritance()`, `validate_config()` |

### Key Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `frequency` | float | 1e9-1000e9 | Carrier frequency (Hz) |
| `tx_power_dBm` | float | -20-50 | Transmit power (dBm) |
| `min_mode` | int | 1-20 | Minimum OAM mode |
| `max_mode` | int | 1-20 | Maximum OAM mode |
| `epsilon` | float | 0.0-1.0 | Exploration rate |
| `learning_rate` | float | 1e-6-1e-1 | Learning rate |

### Key Return Values

| Method | Returns | Description |
|--------|---------|-------------|
| `env.reset()` | (obs, info) | Initial state and info |
| `env.step()` | (obs, reward, done, truncated, info) | Step result |
| `agent.choose_action()` | int | Selected action |
| `agent.learn()` | Dict or None | Training metrics |
| `simulator.run_step()` | (H, sinr) | Channel matrix and SINR | 