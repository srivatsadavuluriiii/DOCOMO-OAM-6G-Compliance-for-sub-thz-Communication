# OAM 6G Project Structure

## Overview
This project implements a physics-based reinforcement learning framework for optimizing OAM mode handover decisions in 6G wireless communications.

## Directory Structure

### Core Components
```
 environment/           # RL environment implementations
    oam_env.py       # Base OAM environment
    stable_oam_env.py # Enhanced environment with stable rewards
    distance_optimized_env.py # Distance-aware optimization
    physics_calculator.py # Physics calculations
    reward_calculator.py # Reward computation
    mobility_model.py # User mobility simulation
    distance_optimizer.py # Distance optimization logic
 models/               # Neural network models
    agent.py         # RL agent implementation
    dqn_model.py     # DQN neural network
 simulator/            # Channel simulation
    channel_simulator.py # Main simulator
    optimized_channel_simulator.py # Optimized version
 utils/                # Utility functions
    config_utils.py  # Configuration management
    path_utils.py    # Path utilities
    input_sanitizer.py # Input validation
    visualization_unified.py # Plotting utilities
```

### Configuration
```
 config/               # Configuration files
    base_config_new.yaml # Base configuration
    rl_config_new.yaml # RL-specific config
    distance_optimization_config.yaml # Distance optimization
    stable_reward_config_new.yaml # Stable reward config
```

### Scripts
```
 scripts/              # Executable scripts
    training/        # Training scripts
       train_distance_optimization.py
       train_stable_rl.py
    analysis/        # Analysis scripts
       analyze_distance_optimization.py
       analyze_three_way_relationship.py
    verification/    # Verification scripts
        verify_oam_physics.py
```

### Tests
```
 tests/                # Test suite
    debug/           # Debug scripts
       debug_position.py
    unit/            # Unit tests
       test_distance_categorization.py
    integration/     # Integration tests
       test_distance_performance.py
    README.md        # Test documentation
```

### Analysis
```
 analysis/             # Analysis outputs
    images/          # Generated visualizations
       three_way_relationship_analysis.png
       throughput_handover_tradeoff_analysis.png
       README.md
    analyze_*.py     # Analysis scripts
```

### Documentation
```
 docs/                 # Documentation
    DISTANCE_OPTIMIZATION_GUIDE.md
    API_REFERENCE.md
```

### Results
```
 results/              # Training results
    distance_optimization_*/ # Timestamped results
```

## Key Features

### 1. Distance Optimization
- **Multi-criteria optimization** balancing distance, throughput, and stability
- **Adaptive thresholds** that adjust based on performance
- **Distance-aware reward calculation** incorporating distance penalties/bonuses

### 2. Physics-Based Simulation
- **Realistic channel modeling** with path loss, turbulence, crosstalk
- **OAM mode simulation** with proper beam patterns
- **Optimized caching** for performance

### 3. Reinforcement Learning
- **DQN agent** with experience replay and target networks
- **Stable reward functions** to prevent training instability
- **Distance-aware mode selection** using optimization scores

### 4. Analysis & Visualization
- **Comprehensive analysis scripts** for performance evaluation
- **Multi-dimensional visualizations** showing relationships between metrics
- **Distance category performance** tracking

## Usage

### Training
```bash
# Distance optimization training
python scripts/training/train_distance_optimization.py

# Stable RL training
python scripts/training/train_stable_rl.py
```

### Analysis
```bash
# Distance optimization analysis
python scripts/analysis/analyze_distance_optimization.py

# Three-way relationship analysis
python analysis/analyze_three_way_relationship.py
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python tests/debug/debug_position.py
python tests/unit/test_distance_categorization.py
python tests/integration/test_distance_performance.py
```

## Configuration

The project uses a hierarchical configuration system:
- `base_config_new.yaml`: Common parameters
- `rl_config_new.yaml`: RL-specific parameters
- `distance_optimization_config.yaml`: Distance optimization parameters
- `stable_reward_config_new.yaml`: Stable reward parameters

## Performance Insights

Based on analysis results:
1. **Distance is the primary factor** affecting throughput
2. **Near distances (50-100m)** show highest throughput potential
3. **Higher handover frequencies** can improve performance at shorter distances
4. **OAM modes 1-3** perform best at near distances
5. **Distance optimization** provides significant performance improvements 