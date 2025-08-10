# OAM 6G Project - Quick Reference

## 🚀 Essential Commands

### Setup & Verification
```bash
# Install dependencies
pip install -r config/requirements.txt

# Verify installation
python -c "import sys; sys.path.append('.'); from environment.oam_env import OAM_Env; print('✅ Ready')"
```

### Training
```bash
# Train distance-optimized agent
python scripts/training/train_distance_optimization.py

# Train standard RL agent
python scripts/training/train_rl.py

# Train stable RL agent
python scripts/training/train_stable_rl.py
```

### Testing
```bash
# Run all tests
python -m pytest

# Run unit tests only
python -m pytest tests/unit/ -v

# Run physics tests
python -m pytest tests/physics/ -v
```

### Analysis
```bash
# Three-way relationship analysis
python analysis/analyze_three_way_relationship.py

# Distance optimization analysis
python scripts/analysis/analyze_distance_optimization.py
```

### Evaluation
```bash
# Evaluate trained agent
python scripts/evaluation/evaluate_rl.py
```

### Verification
```bash
# Verify OAM physics
python scripts/verification/verify_oam_physics.py

# Verify environment
python scripts/verification/verify_environment.py
```

## 🔧 Development Commands

### Code Quality
```bash
# Lint code
python -m flake8 . --max-line-length=120

# Run specific test
python -m pytest tests/unit/test_agent.py -v
```

### Debug
```bash
# Test core components
python -c "
import sys; sys.path.append('.')
from environment.distance_optimized_env import DistanceOptimizedEnv
from models.agent import Agent
from utils.config_utils import load_config
config = load_config('config/base_config_new.yaml')
env = DistanceOptimizedEnv(config)
agent = Agent(state_dim=8, action_dim=3)
print('✅ All components working')
"
```

## 📊 Monitoring

### Training Progress
```bash
# Monitor logs
tail -f results/training.log

# Check GPU usage
nvidia-smi -l 1
```

## 🎯 Quick Workflow

```bash
# 1. Setup
pip install -r config/requirements.txt

# 2. Test
python -m pytest tests/unit/ -v

# 3. Train
python scripts/training/train_distance_optimization.py --num-episodes 1000

# 4. Evaluate
python scripts/evaluation/evaluate_rl.py

# 5. Analyze
python analysis/analyze_three_way_relationship.py
```

## ⚠️ Common Issues

### Import Errors
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Memory Issues
```bash
python scripts/training/train_rl.py --batch-size 32 --no-gpu
```

### Configuration Issues
```bash
python -c "from utils.config_utils import load_config; load_config('config/base_config_new.yaml'); print('✅ Config valid')"
```

---

*For detailed commands, see `docs/COMMAND_REFERENCE.md`* 