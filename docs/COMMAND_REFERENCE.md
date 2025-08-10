# OAM 6G Project - Complete Command Reference

This document provides a comprehensive guide to all available commands for the OAM 6G project, organized by category and use case.

## 📋 Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Training Commands](#training-commands)
3. [Analysis Commands](#analysis-commands)
4. [Evaluation Commands](#evaluation-commands)
5. [Verification Commands](#verification-commands)
6. [Testing Commands](#testing-commands)
7. [Configuration Commands](#configuration-commands)
8. [Utility Commands](#utility-commands)
9. [Development Commands](#development-commands)
10. [Troubleshooting](#troubleshooting)

---

## 🚀 Setup and Installation

### Install Dependencies
```bash
# Install all required packages
pip install -r config/requirements.txt

# Or install with specific Python version
python3.12 -m pip install -r config/requirements.txt
```

### Verify Installation
```bash
# Test core imports
python -c "import sys; sys.path.append('.'); from environment.oam_env import OAM_Env; print('✅ Core environment imported')"

# Test distance optimization
python -c "import sys; sys.path.append('.'); from environment.distance_optimizer import DistanceOptimizer; print('✅ Distance optimization imported')"

# Test simulator
python -c "import sys; sys.path.append('.'); from simulator.channel_simulator import ChannelSimulator; print('✅ Simulator imported')"
```

---

## 🎯 Training Commands

### Distance Optimization Training
```bash
# Train distance-optimized agent
python scripts/training/train_distance_optimization.py

# Train with custom parameters
python scripts/training/train_distance_optimization.py --num-episodes 2000 --output-dir results/distance_opt_v2

# Train with specific seed
python scripts/training/train_distance_optimization.py --seed 42 --num-episodes 1000
```

### Standard RL Training
```bash
# Train standard RL agent
python scripts/training/train_rl.py

# Train with custom config
python scripts/training/train_rl.py --config rl_config_new --num-episodes 1500

# Train with GPU acceleration
python scripts/training/train_rl.py --no-gpu false --batch-size 128
```

### Stable RL Training
```bash
# Train stable RL agent
python scripts/training/train_stable_rl.py

# Train with stable reward config
python scripts/training/train_stable_rl.py --stable-config stable_reward_config_new

# Train with evaluation intervals
python scripts/training/train_stable_rl.py --eval-interval 50 --log-interval 10
```

### Training with Custom Parameters
```bash
# Override episode count
python scripts/training/train_distance_optimization.py --num-episodes 5000

# Override max steps per episode
python scripts/training/train_rl.py --max-steps 200

# Use specific configuration
python scripts/training/train_stable_rl.py --config extended_config_new
```

---

## 📊 Analysis Commands

### Three-Way Relationship Analysis
```bash
# Run three-way relationship analysis
python analysis/analyze_three_way_relationship.py

# Analyze with custom parameters
python analysis/analyze_three_way_relationship.py --num-episodes 500 --output-dir results/analysis_v2
```

### Throughput-Handover Tradeoff Analysis
```bash
# Run tradeoff analysis
python analysis/analyze_throughput_handover_tradeoff.py

# Analyze with specific distance ranges
python analysis/analyze_throughput_handover_tradeoff.py --distance-range 50-300
```

### Distance Optimization Analysis
```bash
# Compare distance optimization vs baseline
python scripts/analysis/analyze_distance_optimization.py

# Analyze with custom episodes
python scripts/analysis/analyze_distance_optimization.py --num-episodes 200
```

---

## 🔍 Evaluation Commands

### RL Agent Evaluation
```bash
# Evaluate trained agent
python scripts/evaluation/evaluate_rl.py

# Evaluate with specific model
python scripts/evaluation/evaluate_rl.py --model-path results/trained_models/agent_best.pth

# Evaluate with custom episodes
python scripts/evaluation/evaluate_rl.py --num-episodes 100 --epsilon 0.0
```

### Performance Benchmarking
```bash
# Run comprehensive evaluation
python scripts/evaluation/evaluate_rl.py --benchmark --save-plots

# Evaluate multiple models
python scripts/evaluation/evaluate_rl.py --model-dir results/trained_models/ --compare
```

---

## ✅ Verification Commands

### Environment Verification
```bash
# Verify environment functionality
python scripts/verification/verify_environment.py

# Verify with custom config
python scripts/verification/verify_environment.py --config base_config_new.yaml
```

### OAM Physics Verification
```bash
# Verify OAM physics implementation
python scripts/verification/verify_oam_physics.py

# Generate verification plots
python scripts/verification/verify_oam_physics.py --save-plots --output-dir plots/verification
```

### System Integration Verification
```bash
# Verify all components work together
python -c "
import sys; sys.path.append('.')
from environment.distance_optimized_env import DistanceOptimizedEnv
from models.agent import Agent
from utils.config_utils import load_config
config = load_config('config/base_config_new.yaml')
env = DistanceOptimizedEnv(config)
agent = Agent(state_dim=8, action_dim=3)
print('✅ All components integrated successfully')
"
```

---

## 🧪 Testing Commands

### Run All Tests
```bash
# Run complete test suite
python -m pytest

# Run with coverage
python -m pytest --cov=. --cov-report=html

# Run with verbose output
python -m pytest -v
```

### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest tests/unit/

# Physics tests only
python -m pytest tests/physics/

# Integration tests only
python -m pytest tests/integration/

# Benchmark tests only
python -m pytest tests/benchmarks/
```

### Run Individual Test Files
```bash
# Test agent functionality
python -m pytest tests/unit/test_agent.py -v

# Test configuration system
python -m pytest tests/unit/test_config.py -v

# Test replay buffer
python -m pytest tests/unit/test_replay_buffer.py -v

# Test environment-simulator integration
python -m pytest tests/integration/test_env_simulator.py -v
```

### Run Physics Tests
```bash
# Test atmospheric absorption
python -m pytest tests/physics/test_atmospheric_absorption.py -v

# Test beam width evolution
python -m pytest tests/physics/test_beam_width_evolution.py -v

# Test turbulence effects
python -m pytest tests/physics/test_kolmogorov_turbulence.py -v

# Test OAM crosstalk
python -m pytest tests/physics/test_oam_crosstalk.py -v
```

### Run Performance Tests
```bash
# Test agent performance
python -m pytest tests/benchmarks/test_agent_benchmarks.py -v

# Test environment performance
python -m pytest tests/benchmarks/test_environment_benchmarks.py -v

# Test simulator performance
python -m pytest tests/benchmarks/test_simulator_benchmarks.py -v
```

### Run Regression Tests
```bash
# Test training regression
python -m pytest tests/regression/test_training.py -v
```

---

## ⚙️ Configuration Commands

### Validate Configuration
```bash
# Validate base configuration
python -c "from utils.config_utils import load_config; load_config('config/base_config_new.yaml'); print('✅ Config valid')"

# Validate distance optimization config
python -c "from utils.config_utils import load_config; load_config('config/distance_optimization_config.yaml'); print('✅ Distance config valid')"

# Validate all configurations
python -c "
from utils.config_utils import load_config
configs = ['base_config_new.yaml', 'rl_config_new.yaml', 'stable_reward_config_new.yaml', 'distance_optimization_config.yaml']
for config in configs:
    load_config(f'config/{config}')
    print(f'✅ {config} valid')
"
```

### Load Hierarchical Configuration
```bash
# Load with inheritance
python -c "from utils.hierarchical_config import load_hierarchical_config; config = load_hierarchical_config('rl_config_new'); print('✅ Hierarchical config loaded')"
```

---

## 🛠️ Utility Commands

### Path Management
```bash
# Get project root
python -c "from utils.path_utils import get_project_root; print(get_project_root())"

# Create results directory
python -c "from utils.path_utils import create_results_dir; print(create_results_dir('test_run'))"
```

### Configuration Migration
```bash
# Migrate old configs to new format
python scripts/config_migration.py

# Migrate specific config
python scripts/config_migration.py --input old_config.yaml --output new_config.yaml
```

### Main Script
```bash
# Run main project script
python scripts/main.py

# Run with arguments
python scripts/main.py --mode training --config base_config_new.yaml
```

---

## 🔧 Development Commands

### Code Quality Checks
```bash
# Run linter
python -m flake8 . --max-line-length=120

# Run linter on specific file
python -m flake8 scripts/verification/verify_oam_physics.py --max-line-length=120

# Run linter with specific rules
python -m flake8 . --select=E741,W293,W291,W292,F541
```

### Type Checking
```bash
# Run mypy (if installed)
mypy . --ignore-missing-imports

# Check specific module
mypy environment/ --ignore-missing-imports
```

### Code Formatting
```bash
# Format with black (if installed)
black .

# Format specific file
black scripts/training/train_distance_optimization.py
```

### Documentation Generation
```bash
# Generate API documentation (if sphinx installed)
sphinx-build -b html docs/source docs/build/html

# Generate coverage report
python -m pytest --cov=. --cov-report=html
```

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### Import Errors
```bash
# Fix import path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to path in script
python -c "import sys; sys.path.append('.'); from environment.oam_env import OAM_Env"
```

#### Configuration Errors
```bash
# Validate configuration manually
python -c "
from utils.config_utils import load_config
from utils.input_sanitizer import InputSanitizer
config = load_config('config/base_config_new.yaml')
sanitizer = InputSanitizer()
sanitizer.validate_config(config)
print('✅ Configuration validated')
"
```

#### Memory Issues
```bash
# Run with reduced batch size
python scripts/training/train_rl.py --batch-size 32

# Run with CPU only
python scripts/training/train_rl.py --no-gpu

# Monitor memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

#### Performance Issues
```bash
# Profile training script
python -m cProfile -o profile.stats scripts/training/train_distance_optimization.py

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(10)
"
```

### Debug Commands
```bash
# Debug environment creation
python -c "
import sys; sys.path.append('.')
from environment.oam_env import OAM_Env
from utils.config_utils import load_config
config = load_config('config/base_config_new.yaml')
env = OAM_Env(config)
print(f'Environment created: {env.observation_space.shape}')
"

# Debug agent creation
python -c "
import sys; sys.path.append('.')
from models.agent import Agent
agent = Agent(state_dim=8, action_dim=3)
print(f'Agent created: state_dim={agent.state_dim}, action_dim={agent.action_dim}')
"

# Debug simulator
python -c "
import sys; sys.path.append('.')
from simulator.channel_simulator import ChannelSimulator
from utils.config_utils import load_config
config = load_config('config/base_config_new.yaml')
sim = ChannelSimulator(config)
print(f'Simulator created: frequency={sim.frequency} Hz')
"
```

---

## 📈 Monitoring and Logging

### Training Monitoring
```bash
# Monitor training progress
tail -f results/training.log

# Monitor GPU usage (if available)
nvidia-smi -l 1

# Monitor system resources
htop
```

### Log Analysis
```bash
# Extract training metrics
grep "Episode" results/training.log | tail -20

# Extract reward information
grep "Reward" results/training.log | tail -10

# Extract throughput data
grep "Throughput" results/training.log | tail -10
```

---

## 🎯 Quick Start Commands

### Complete Workflow
```bash
# 1. Setup
pip install -r config/requirements.txt

# 2. Verify installation
python -c "import sys; sys.path.append('.'); from environment.oam_env import OAM_Env; print('✅ Ready')"

# 3. Run tests
python -m pytest tests/unit/ -v

# 4. Train distance-optimized agent
python scripts/training/train_distance_optimization.py --num-episodes 1000

# 5. Evaluate results
python scripts/evaluation/evaluate_rl.py

# 6. Run analysis
python analysis/analyze_three_way_relationship.py
```

### Development Workflow
```bash
# 1. Code quality check
python -m flake8 . --max-line-length=120

# 2. Run tests
python -m pytest tests/unit/ -v

# 3. Verify physics
python scripts/verification/verify_oam_physics.py

# 4. Train and test
python scripts/training/train_distance_optimization.py --num-episodes 100
python scripts/evaluation/evaluate_rl.py
```

---

## 📝 Notes

- All commands assume you're in the project root directory
- Use `python3` instead of `python` if needed
- Add `--help` to any script for detailed options
- Check `docs/` for additional documentation
- Monitor system resources during training
- Backup results before major changes

---

*Last updated: August 2024*
*OAM 6G Project - Complete Command Reference* 