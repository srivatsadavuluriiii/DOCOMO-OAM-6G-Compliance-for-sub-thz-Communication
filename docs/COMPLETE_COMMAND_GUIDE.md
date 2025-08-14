#  Complete Command Guide - 6G OAM Deep Reinforcement Learning Project

##  Table of Contents

1. [Project Setup](#-project-setup)
2. [Environment Management](#-environment-management)
3. [Configuration Management](#-configuration-management)
4. [Training Commands](#-training-commands)
5. [Evaluation Commands](#-evaluation-commands)
6. [Analysis Commands](#-analysis-commands)
7. [Verification Commands](#-verification-commands)
8. [Testing Commands](#-testing-commands)
9. [Visualization Commands](#-visualization-commands)
10. [Physics Validation](#-physics-validation)
11. [Performance Tuning](#-performance-tuning)
12. [Debugging Commands](#-debugging-commands)
13. [Data Management](#-data-management)
14. [Advanced Usage](#-advanced-usage)
15. [Troubleshooting](#-troubleshooting)

---

##  Project Setup

### Initial Setup
```bash
# Clone and setup (if from repository)
git clone <repository-url>
cd "OAM 6G"

# Create virtual environment
python3 -m venv oam_rl_env
source oam_rl_env/bin/activate  # On Windows: oam_rl_env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib seaborn
pip install gymnasium pyyaml optuna
pip install pytest pytest-cov
pip install jupyter notebook
pip install pandas scikit-learn
```

### Directory Structure Check
```bash
# Verify project structure
ls -la
tree . -L 2  # If tree is installed
find . -name "*.py" | head -10  # Check Python files
find . -name "*.yaml" | head -5  # Check config files
```

### Dependencies Verification
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(torch|numpy|gymnasium|pyyaml)"

# Verify CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

##  Environment Management

### Virtual Environment Commands
```bash
# Activate environment
source oam_rl_env/bin/activate

# Deactivate environment
deactivate

# Check environment location
which python
echo $VIRTUAL_ENV

# Install new packages
pip install <package-name>

# Export requirements
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

### Environment Variables
```bash
# Set project root (optional)
export OAM_PROJECT_ROOT="$(pwd)"

# Set CUDA device (if multiple GPUs)
export CUDA_VISIBLE_DEVICES=0

# Set Python path (if needed)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check environment variables
env | grep -E "(OAM|CUDA|PYTHON)"
```

---

##  Configuration Management

### Configuration File Commands
```bash
# View main configuration
cat config/config.yaml

# Validate configuration syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# Check specific config sections
grep -A 10 "physics:" config/config.yaml
grep -A 10 "docomo_6g_system:" config/config.yaml
grep -A 10 "reinforcement_learning:" config/config.yaml
```

### Configuration Validation
```bash
# Validate all configs
python -c "
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(' Configuration loaded successfully')
print(f'Available sections: {list(config.keys())}')
"

# Check for required sections
python scripts/verification/verify_environment.py --config config/config.yaml
```

---

##  Training Commands

### Basic Training
```bash
# Basic DOCOMO compliance training
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/basic_training \
    --episodes 1000 \
    --max-steps 900

# Quick smoke test (fast training)
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/smoke_test \
    --episodes 10 \
    --max-steps 100 \
    --eval-episodes 2
```

### Advanced Training Options
```bash
# GPU training with evaluation
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/gpu_training \
    --episodes 5000 \
    --max-steps 900 \
    --eval-episodes 50 \
    --device cuda

# CPU-only training
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/cpu_training \
    --episodes 1000 \
    --max-steps 900 \
    --no-gpu

# Training with curriculum learning
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/curriculum_training \
    --episodes 3000 \
    --max-steps 900 \
    --curriculum

# Resume from checkpoint
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/resumed_training \
    --episodes 2000 \
    --resume-from results/previous_training/checkpoint_best.pth
```

### Multi-Agent Training
```bash
# Single user (default)
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/single_user \
    --episodes 1000 \
    --num-users 1

# Multi-user training (2 users)
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/multi_user_2 \
    --episodes 1000 \
    --num-users 2

# Dense network (4 users)
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/dense_network \
    --episodes 2000 \
    --num-users 4
```

### Network Slicing Training
```bash
# eMBB slice training
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/embb_slice \
    --episodes 1000 \
    --slice-type embb

# uRLLC slice training
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/urllc_slice \
    --episodes 1000 \
    --slice-type urllc

# mMTC slice training
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/mmtc_slice \
    --episodes 1000 \
    --slice-type mmtc
```

---

##  Evaluation Commands

### Basic Evaluation
```bash
# Evaluate trained model
python scripts/evaluation/evaluate_rl.py \
    --model-path results/basic_training/model_best.pth \
    --config config/config.yaml \
    --episodes 100 \
    --output-dir results/evaluation

# Quick evaluation
python scripts/evaluation/evaluate_rl.py \
    --model-path results/basic_training/model_best.pth \
    --config config/config.yaml \
    --episodes 10 \
    --output-dir results/quick_eval
```

### Comprehensive Evaluation
```bash
# Full performance evaluation
python scripts/evaluation/evaluate_rl.py \
    --model-path results/basic_training/model_best.pth \
    --config config/config.yaml \
    --episodes 1000 \
    --output-dir results/full_evaluation \
    --save-trajectories \
    --detailed-metrics

# Compare multiple models
python scripts/evaluation/evaluate_rl.py \
    --model-path results/model1/model_best.pth \
    --baseline-path results/model2/model_best.pth \
    --config config/config.yaml \
    --episodes 500 \
    --output-dir results/model_comparison

# Evaluate with different scenarios
python scripts/evaluation/evaluate_rl.py \
    --model-path results/basic_training/model_best.pth \
    --config config/config.yaml \
    --episodes 200 \
    --scenario urban_dense \
    --output-dir results/urban_evaluation

python scripts/evaluation/evaluate_rl.py \
    --model-path results/basic_training/model_best.pth \
    --config config/config.yaml \
    --episodes 200 \
    --scenario highway_mobility \
    --output-dir results/highway_evaluation
```

---

##  Analysis Commands

### Performance Analysis
```bash
# Distance vs metrics analysis
python scripts/analysis/plot_distance_vs_metrics.py \
    --results-dir results/basic_training \
    --output-dir analysis/distance_analysis

# Generate comprehensive plots
python scripts/analysis/plot_distance_vs_metrics.py \
    --results-dir results/basic_training \
    --output-dir analysis/comprehensive \
    --plot-types throughput,handover,latency,energy

# Compare before/after training
python scripts/analysis/plot_distance_vs_metrics.py \
    --results-dir results/basic_training \
    --baseline-dir results/untrained_baseline \
    --output-dir analysis/training_comparison
```

### Advanced Analysis
```bash
# Three-way relationship analysis
python scripts/analysis/analyze_three_way_relationship.py \
    --data-dir results/basic_training \
    --output-dir analysis/relationships

# Performance correlation analysis
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv('results/basic_training/episode_metrics.csv')

# Correlation matrix
corr = df[['throughput', 'latency', 'energy_efficiency', 'handovers']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Performance Metrics Correlation')
plt.tight_layout()
plt.savefig('analysis/correlation_matrix.png', dpi=300)
plt.show()
"

# Statistical analysis
python -c "
import pandas as pd
import numpy as np

df = pd.read_csv('results/basic_training/episode_metrics.csv')
print('=== PERFORMANCE STATISTICS ===')
print(df[['throughput', 'latency', 'handovers']].describe())
print('\n=== IMPROVEMENT OVER TIME ===')
print(f'Initial throughput: {df.throughput.head(10).mean():.2f} Gbps')
print(f'Final throughput: {df.throughput.tail(10).mean():.2f} Gbps')
print(f'Improvement: {((df.throughput.tail(10).mean() / df.throughput.head(10).mean()) - 1) * 100:.1f}%')
"
```

---

##  Verification Commands

### Environment Verification
```bash
# Basic environment check
python scripts/verification/verify_environment.py

# Detailed environment verification
python scripts/verification/verify_environment.py \
    --config config/config.yaml \
    --detailed \
    --check-gpu

# DOCOMO baseline verification
python scripts/verification/verify_docomo_100m.py \
    --config config/config.yaml

# Physics verification
python scripts/verification/verify_oam_physics.py \
    --config config/config.yaml \
    --frequency 300e9 \
    --bandwidth 20e9
```

### System Health Checks
```bash
# Check all imports
python -c "
import sys
modules = [
    'environment.docomo_6g_env',
    'models.agent', 
    'models.dqn_model',
    'simulator.channel_simulator',
    'simulator.oam_beam_physics',
    'environment.physics_calculator',
    'environment.advanced_physics',
    'environment.docomo_compliance'
]

for module in modules:
    try:
        __import__(module)
        print(f' {module}')
    except ImportError as e:
        print(f' {module}: {e}')
"

# Memory usage check
python -c "
import psutil
import torch

print(f'CPU cores: {psutil.cpu_count()}')
print(f'RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')
"
```

---

## Testing Commands

### Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/physics/ -v
pytest tests/physics/test_oam_beam_physics.py -v
pytest tests/physics/test_mathematical_accuracy.py -v
pytest tests/physics/test_advanced_physics_compliance.py -v

# Run tests with coverage
pytest tests/ --cov=environment --cov=models --cov=simulator --cov-report=html

# Run specific test methods
pytest tests/physics/test_oam_beam_physics.py::TestOAMBeamPhysics::test_fundamental_gaussian_mode -v
```

### Physics Tests
```bash
# OAM beam physics tests
pytest tests/physics/test_oam_beam_physics.py::TestOAMBeamPhysics::test_beam_divergence_evolution -v
pytest tests/physics/test_oam_beam_physics.py::TestOAMBeamPhysics::test_mode_orthogonality -v
pytest tests/physics/test_oam_beam_physics.py::TestOAMBeamPhysics::test_atmospheric_mode_coupling -v

# Mathematical accuracy tests  
pytest tests/physics/test_mathematical_accuracy.py::TestMathematicalAccuracy::test_beam_divergence_physics -v
pytest tests/physics/test_mathematical_accuracy.py::TestMathematicalAccuracy::test_enhanced_throughput_adaptive_modulation -v

# Advanced physics tests
pytest tests/physics/test_advanced_physics_compliance.py::TestAdvancedPhysicsModels::test_doppler_shift_extreme_mobility -v
pytest tests/physics/test_advanced_physics_compliance.py::TestDOCOMOCompliance::test_energy_efficiency_100x_improvement -v
```

### Custom Physics Validation
```bash
# Validate OAM beam physics directly
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('oam_beam_physics', 'simulator/oam_beam_physics.py')
oam_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oam_module)

oam = oam_module.OAMBeamPhysics()
params = oam_module.OAMBeamParameters(
    wavelength=1e-3, waist_radius=0.01, aperture_radius=0.05, oam_mode_l=1
)

import numpy as np
r = np.linspace(0, 0.02, 20)
phi = np.linspace(0, 2*np.pi, 20)
R, PHI = np.meshgrid(r, phi, indexing='ij')

result = oam.laguerre_gaussian_field(R, PHI, 0.0, params)
print(f' OAM field calculation successful')
print(f'Beam radius: {result.beam_radius:.4f} m')
print(f'Mode purity: {result.mode_purity:.4f}')
"

# Test DOCOMO compliance
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('docomo_compliance', 'environment/docomo_compliance.py')
docomo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docomo_module)

docomo = docomo_module.DOCOMOComplianceManager()

# Test energy efficiency
ee = docomo.calculate_energy_efficiency(1000e9, 10, 300e9)
print(f' Energy efficiency: {ee.improvement_factor_vs_5g:.0f}x improvement')

# Test sensing accuracy
sa = docomo.calculate_sensing_accuracy(600e9, 50e9, 30, 256)
print(f' Sensing accuracy: {sa.position_accuracy_cm:.2f} cm')
"
```

---

##  Visualization Commands

### Basic Plots
```bash
# Generate OAM mode gallery
python scripts/visualization/generate_oam_gallery.py \
    --output-dir visualizations/oam_gallery \
    --modes 1,2,3,4,5,6,7,8

# Visualize specific OAM modes
python scripts/visualization/visualize_oam_modes.py \
    --mode 1 \
    --frequency 300e9 \
    --output-dir visualizations/mode_1

# Generate training progress plots
python -c "
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/basic_training/episode_metrics.csv')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0,0].plot(df.episode, df.throughput)
axes[0,0].set_title('Throughput vs Episode')
axes[0,0].set_ylabel('Throughput (Gbps)')

axes[0,1].plot(df.episode, df.latency)
axes[0,1].set_title('Latency vs Episode')
axes[0,1].set_ylabel('Latency (ms)')

axes[1,0].plot(df.episode, df.handovers)
axes[1,0].set_title('Handovers vs Episode')
axes[1,0].set_ylabel('Handovers')

axes[1,1].plot(df.episode, df.reward)
axes[1,1].set_title('Reward vs Episode')
axes[1,1].set_ylabel('Reward')

plt.tight_layout()
plt.savefig('visualizations/training_progress.png', dpi=300)
plt.show()
"
```

### Advanced Visualizations
```bash
# 3D performance surface plots
python -c "
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Load data
df = pd.read_csv('results/basic_training/episode_metrics.csv')

# Create 3D surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Sample data for surface
episodes = df.episode.values[::10]  # Every 10th episode
throughput = df.throughput.values[::10]
latency = df.latency.values[::10]

ax.scatter(episodes, throughput, latency, c=episodes, cmap='viridis')
ax.set_xlabel('Episode')
ax.set_ylabel('Throughput (Gbps)')
ax.set_zlabel('Latency (ms)')
ax.set_title('3D Performance Evolution')

plt.savefig('visualizations/3d_performance.png', dpi=300)
plt.show()
"

# Heatmap of performance metrics
python -c "
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('results/basic_training/episode_metrics.csv')

# Create performance matrix
metrics = ['throughput', 'latency', 'energy_efficiency', 'handovers', 'reward']
data = df[metrics].values

# Normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_norm = scaler.fit_transform(data)

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data_norm.T, 
           xticklabels=range(0, len(data_norm), len(data_norm)//10),
           yticklabels=metrics,
           cmap='RdYlBu_r',
           center=0)
plt.title('Performance Metrics Heatmap (Normalized)')
plt.xlabel('Episode')
plt.tight_layout()
plt.savefig('visualizations/performance_heatmap.png', dpi=300)
plt.show()
"
```

---

##  Physics Validation

### OAM Beam Physics
```bash
# Test Laguerre-Gaussian beam generation
python -c "
import sys
sys.path.insert(0, '.')
from simulator.oam_beam_physics import OAMBeamPhysics, OAMBeamParameters
import numpy as np

oam = OAMBeamPhysics()
params = OAMBeamParameters(
    wavelength=1e-3, waist_radius=0.01, aperture_radius=0.05,
    oam_mode_l=2, radial_mode_p=0
)

# Test beam divergence
distances = np.array([0, 100, 500, 1000])
evolution = oam.beam_divergence_evolution(distances, params)

print('=== BEAM DIVERGENCE TEST ===')
for i, d in enumerate(distances):
    print(f'Distance {d:4.0f}m: beam width = {evolution[\"beam_radius\"][i]:.4f}m')

# Test mode orthogonality
overlap = oam.mode_overlap_integral(1, 0, 2, 0, params, params)
print(f'\\nMode orthogonality (l=1,2): {abs(overlap):.6f} (should be ~0)')

overlap_same = oam.mode_overlap_integral(1, 0, 1, 0, params, params)
print(f'Same mode overlap (l=1,1): {abs(overlap_same):.6f} (should be ~1)')
"

# Test atmospheric coupling
python -c "
import sys
sys.path.insert(0, '.')
from simulator.oam_beam_physics import OAMBeamPhysics, OAMBeamParameters

oam = OAMBeamPhysics()
params = OAMBeamParameters(wavelength=1e-3, waist_radius=0.01, aperture_radius=0.05)

# Test atmospheric mode coupling
cn2 = 1e-14  # Moderate turbulence
coupling = oam.atmospheric_mode_coupling(cn2, 3, 300e12, 1000, params)

print('=== ATMOSPHERIC COUPLING TEST ===')
print(f'Turbulence strength (Cn²): {cn2:.0e} m^(-2/3)')
print(f'Coupling matrix shape: {coupling.shape}')
print(f'Diagonal elements (self-coupling): {np.diag(coupling)}')
print(f'Off-diagonal max (cross-coupling): {np.max(np.abs(coupling - np.diag(np.diag(coupling)))):.4f}')
"
```

### Advanced Physics Validation
```bash
# Test Doppler effects at extreme mobility
python -c "
import sys
sys.path.insert(0, '.')
from environment.advanced_physics import AdvancedPhysicsModels, DopplerParameters

physics = AdvancedPhysicsModels()

# DOCOMO extreme mobility test
params = DopplerParameters(
    velocity_kmh=500,  # 500 km/h target
    frequency_hz=600e9,  # 600 GHz
    angle_degrees=0.0
)

result = physics.calculate_doppler_shift(params)

print('=== EXTREME MOBILITY DOPPLER TEST ===')
print(f'Velocity: {params.velocity_kmh} km/h')
print(f'Frequency: {params.frequency_hz/1e9:.0f} GHz')
print(f'Doppler shift: {result[\"doppler_shift_hz\"]/1000:.1f} kHz')
print(f'Coherence time: {result[\"coherence_time_ms\"]:.3f} ms')
print(f'Severity: {result[\"doppler_severity\"]}')
"

# Test THz phase noise
python -c "
import sys
sys.path.insert(0, '.')
from environment.advanced_physics import AdvancedPhysicsModels, PhaseNoiseParameters

physics = AdvancedPhysicsModels()

frequencies = [140e9, 300e9, 600e9]
for freq in frequencies:
    params = PhaseNoiseParameters(
        frequency_hz=freq,
        phase_noise_dbchz=-85 + (freq/100e9 - 1)*5,
        oscillator_type='crystal'
    )
    
    result = physics.calculate_phase_noise_impact(params, 1e9)
    
    print(f'=== {freq/1e9:.0f} GHz PHASE NOISE ===')
    print(f'Phase error: {result[\"phase_error_rms_deg\"]:.1f}°')
    print(f'SNR degradation: {result[\"snr_degradation_db\"]:.1f} dB')
    print(f'Quality: {result[\"oscillator_quality\"]}')
    print()
"
```

---

##  Performance Tuning

### Hyperparameter Optimization
```bash
# Basic hyperparameter tuning
python scripts/tuning/tune_agent.py \
    --config config/config.yaml \
    --trials 50 \
    --study-name basic_tuning

# Extensive hyperparameter search
python scripts/tuning/tune_agent.py \
    --config config/config.yaml \
    --trials 200 \
    --study-name extensive_tuning \
    --timeout 3600  # 1 hour timeout

# Resume existing study
python scripts/tuning/tune_agent.py \
    --config config/config.yaml \
    --trials 100 \
    --study-name basic_tuning \
    --resume

# Tune specific parameters
python scripts/tuning/tune_agent.py \
    --config config/config.yaml \
    --trials 100 \
    --study-name learning_rate_tuning \
    --param-focus learning_rate
```

### Performance Profiling
```bash
# Profile training performance
python -m cProfile -o training_profile.prof scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/profile_test \
    --episodes 10 \
    --max-steps 100

# Analyze profile
python -c "
import pstats
p = pstats.Stats('training_profile.prof')
p.sort_stats('cumulative').print_stats(20)
"

# Memory profiling
python -c "
import tracemalloc
import subprocess
import sys

tracemalloc.start()

# Run a small training
subprocess.run([
    sys.executable, 'scripts/training/train_compliance.py',
    '--config', 'config/config.yaml',
    '--output-dir', 'results/memory_test',
    '--episodes', '5',
    '--max-steps', '50'
])

current, peak = tracemalloc.get_traced_memory()
print(f'Current memory usage: {current / 1024 / 1024:.1f} MB')
print(f'Peak memory usage: {peak / 1024 / 1024:.1f} MB')
"
```

---

##  Debugging Commands

### Debug Training Issues
```bash
# Debug mode training (verbose output)
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/debug_training \
    --episodes 5 \
    --max-steps 50 \
    --debug \
    --verbose

# Single-step debugging
python -c "
import sys
sys.path.insert(0, '.')
from environment.docomo_6g_env import DOCOMO_6G_Environment
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

env = DOCOMO_6G_Environment(config=config)
obs = env.reset()
print(f'Initial observation shape: {obs[0].shape if isinstance(obs, tuple) else obs.shape}')
print(f'Action space: {env.action_space}')
print(f'Observation space: {env.observation_space}')

# Test single step
action = env.action_space.sample()
step_result = env.step(action)
print(f'Step result length: {len(step_result)}')
"

# Environment debugging
python -c "
import sys
sys.path.insert(0, '.')
from environment.docomo_6g_env import DOCOMO_6G_Environment
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

env = DOCOMO_6G_Environment(config=config)

print('=== ENVIRONMENT DEBUG INFO ===')
print(f'State space shape: {env.observation_space.shape}')
print(f'Action space size: {env.action_space.n}')
print(f'Frequency bands: {list(env.frequency_bands.keys())}')
print(f'OAM modes: {env.oam_modes}')

# Test episode
obs = env.reset()
for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f'Step {i}: reward={reward:.3f}, done={done}, info_keys={list(info.keys())}')
    if done:
        break
"
```

### Debug Physics Calculations
```bash
# Debug channel simulator
python -c "
import sys
sys.path.insert(0, '.')
from simulator.channel_simulator import ChannelSimulator
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

sim = ChannelSimulator(config)

print('=== CHANNEL SIMULATOR DEBUG ===')
print(f'Frequency range: {sim.min_freq/1e9:.0f}-{sim.max_freq/1e9:.0f} GHz')
print(f'OAM modes: {sim.min_mode}-{sim.max_mode}')
print(f'Turbulence enabled: {hasattr(sim, \"turbulence_strength\")}')

# Test signal calculation
distance = 100  # meters
frequency = 300e9  # 300 GHz
oam_mode = 1
beam_width = 0.01  # 1 cm

result = sim.run_step(distance, frequency, oam_mode, beam_width)
print(f'\\nTest calculation at {distance}m, {frequency/1e9:.0f}GHz:')
print(f'SINR: {result[\"sinr_db\"]:.2f} dB')
print(f'Path loss: {result[\"path_loss_db\"]:.2f} dB')
print(f'Atmospheric loss: {result.get(\"atmospheric_loss_db\", \"N/A\"):.2f} dB')
"

# Debug physics calculator
python -c "
import sys
sys.path.insert(0, '.')
from environment.physics_calculator import PhysicsCalculator
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

physics = PhysicsCalculator(config)

print('=== PHYSICS CALCULATOR DEBUG ===')
print(f'Bandwidth: {physics.bandwidth/1e9:.1f} GHz')

# Test calculations
sinr_db = 20
throughput = physics.calculate_throughput(sinr_db)
print(f'Throughput at {sinr_db} dB SINR: {throughput/1e9:.2f} Gbps')

# Test enhanced throughput
enhanced = physics.calculate_enhanced_throughput(sinr_db, 300e9, 'adaptive', 0.8)
print(f'Enhanced throughput: {enhanced[\"practical_throughput\"]/1e9:.2f} Gbps')
print(f'Modulation: {enhanced[\"modulation\"]}')
print(f'Link status: {enhanced[\"link_status\"]}')
"
```

---

##  Data Management

### Results Management
```bash
# List all results
find results/ -name "*.csv" | head -10
find results/ -name "*.png" | head -10
find results/ -name "*.pth" | head -10

# Archive results
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/

# Clean old results (be careful!)
find results/ -name "*.log" -mtime +7 -delete
find results/ -name "checkpoint_*.pth" -not -name "*best*" -mtime +3 -delete

# Check disk usage
du -sh results/
df -h .  # Overall disk space
```

### Data Analysis Scripts
```bash
# Extract metrics from all results
python -c "
import os
import pandas as pd
import glob

result_dirs = glob.glob('results/*/')
all_metrics = []

for result_dir in result_dirs:
    metrics_file = os.path.join(result_dir, 'episode_metrics.csv')
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        df['experiment'] = os.path.basename(result_dir.rstrip('/'))
        all_metrics.append(df)

if all_metrics:
    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv('analysis/all_experiments.csv', index=False)
    print(f'Combined {len(all_metrics)} experiments with {len(combined)} total episodes')
    print(f'Experiments: {combined.experiment.unique()}')
else:
    print('No experiment data found')
"

# Generate summary report
python -c "
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('analysis/all_experiments.csv')
    
    print('=== EXPERIMENT SUMMARY REPORT ===')
    
    # Group by experiment
    summary = df.groupby('experiment').agg({
        'throughput': ['mean', 'std', 'max'],
        'latency': ['mean', 'std', 'min'],
        'handovers': ['mean', 'std'],
        'reward': ['mean', 'std', 'max'],
        'episode': 'count'
    }).round(3)
    
    print(summary)
    
    # Best performing experiment
    best_exp = df.groupby('experiment')['reward'].mean().idxmax()
    print(f'\\nBest experiment: {best_exp}')
    
    best_data = df[df.experiment == best_exp]
    print(f'Average reward: {best_data.reward.mean():.3f}')
    print(f'Average throughput: {best_data.throughput.mean():.3f} Gbps')
    print(f'Average latency: {best_data.latency.mean():.3f} ms')
    
except FileNotFoundError:
    print('No combined experiment data found. Run data extraction first.')
"
```

---

##  Advanced Usage

### Custom Training Scenarios
```bash
# Urban dense scenario
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/urban_dense \
    --episodes 2000 \
    --scenario urban_dense \
    --num-users 4

# Highway mobility scenario
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/highway_mobility \
    --episodes 1500 \
    --scenario highway_mobility \
    --max-speed 500  # km/h

# Indoor hotspot scenario
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/indoor_hotspot \
    --episodes 1000 \
    --scenario indoor_hotspot \
    --frequency-focus sub_thz_300

# Industrial IoT scenario
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/industrial_iot \
    --episodes 3000 \
    --scenario industrial_iot \
    --reliability-focus
```

### Research Extensions
```bash
# Train with different RL algorithms (if implemented)
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/double_dqn \
    --episodes 2000 \
    --algorithm double_dqn

python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/dueling_dqn \
    --episodes 2000 \
    --algorithm dueling_dqn

# Experiment with different reward functions
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/energy_focused \
    --episodes 1500 \
    --reward-weights energy:0.4,throughput:0.3,latency:0.3

# Physics sensitivity analysis
python -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import yaml
from environment.docomo_6g_env import DOCOMO_6G_Environment

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('=== PHYSICS SENSITIVITY ANALYSIS ===')

# Test different turbulence strengths
turbulence_values = [1e-16, 1e-15, 1e-14, 1e-13]
for turb in turbulence_values:
    config['physics']['atmospheric']['turbulence_strength'] = turb
    env = DOCOMO_6G_Environment(config=config)
    
    # Run a few steps
    obs = env.reset()
    rewards = []
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        if done:
            obs = env.reset()
    
    print(f'Turbulence {turb:.0e}: avg reward = {np.mean(rewards):.3f}')
"
```

### Jupyter Notebook Integration
```bash
# Start Jupyter notebook
jupyter notebook

# Run in notebook cell:
# %load_ext autoreload
# %autoreload 2
# 
# import sys
# sys.path.insert(0, '.')
# from environment.docomo_6g_env import DOCOMO_6G_Environment
# import yaml
# 
# with open('config/config.yaml', 'r') as f:
#     config = yaml.safe_load(f)
# 
# env = DOCOMO_6G_Environment(config=config)
# # Interactive experimentation...

# Create analysis notebook
cat > analysis_notebook.py << 'EOF'
# %%
import sys
sys.path.insert(0, '.')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load training results
df = pd.read_csv('results/basic_training/episode_metrics.csv')
df.head()

# %%
# Plot training progress
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0,0].plot(df.episode, df.throughput)
axes[0,0].set_title('Throughput vs Episode')

axes[0,1].plot(df.episode, df.latency)
axes[0,1].set_title('Latency vs Episode')

axes[1,0].plot(df.episode, df.handovers)
axes[1,0].set_title('Handovers vs Episode')

axes[1,1].plot(df.episode, df.reward)
axes[1,1].set_title('Reward vs Episode')

plt.tight_layout()
plt.show()

# %%
# Physics analysis
from environment.advanced_physics import AdvancedPhysicsModels
physics = AdvancedPhysicsModels()

# Test different scenarios...
EOF

# Convert to notebook
jupyter nbconvert --to notebook analysis_notebook.py
```

---

##  Troubleshooting

### Common Issues and Solutions

#### Import Errors
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check module structure
python -c "import sys; print('\\n'.join(sys.path))"

# Test individual imports
python -c "from environment.docomo_6g_env import DOCOMO_6G_Environment; print(' Environment OK')"
python -c "from models.agent import Agent; print(' Agent OK')"
python -c "from simulator.channel_simulator import ChannelSimulator; print(' Simulator OK')"
```

#### Memory Issues
```bash
# Check available memory
free -h  # Linux
vm_stat | grep "Pages free"  # macOS

# Reduce batch size in training
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/low_memory \
    --episodes 100 \
    --batch-size 16 \
    --memory-limit 4000000000  # 4GB limit
```

#### GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU memory
nvidia-smi  # If NVIDIA GPU

# Force CPU training
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/cpu_only \
    --episodes 100 \
    --no-gpu

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

#### Configuration Issues
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml')); print(' Config valid')"

# Check for missing sections
python -c "
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

required = ['physics', 'docomo_6g_system', 'simulation']
missing = [sec for sec in required if sec not in config]
if missing:
    print(f' Missing sections: {missing}')
else:
    print(' All required sections present')
"

# Reset to default config (backup first!)
cp config/config.yaml config/config_backup.yaml
# Then restore from template or repository
```

#### Training Convergence Issues
```bash
# Check learning rate
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/debug_lr \
    --episodes 50 \
    --learning-rate 1e-5  # Lower learning rate

# Check reward scaling
python -c "
import sys
sys.path.insert(0, '.')
from environment.docomo_6g_env import DOCOMO_6G_Environment
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

env = DOCOMO_6G_Environment(config=config)
obs = env.reset()

rewards = []
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    rewards.append(reward)
    if done:
        obs = env.reset()

import numpy as np
print(f'Reward range: [{np.min(rewards):.3f}, {np.max(rewards):.3f}]')
print(f'Reward mean: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}')
"

# Enable detailed logging
python scripts/training/train_compliance.py \
    --config config/config.yaml \
    --output-dir results/detailed_debug \
    --episodes 20 \
    --log-level DEBUG \
    --save-every 5
```

---

##  Quick Reference Summary

### Most Common Commands
```bash
# 1. Basic training
python scripts/training/train_compliance.py --config config/config.yaml --output-dir results/basic --episodes 1000

# 2. Run all tests
pytest tests/ -v

# 3. Evaluate model
python scripts/evaluation/evaluate_rl.py --model-path results/basic/model_best.pth --config config/config.yaml --episodes 100

# 4. Generate plots
python scripts/analysis/plot_distance_vs_metrics.py --results-dir results/basic --output-dir analysis/

# 5. Verify environment
python scripts/verification/verify_environment.py

# 6. Hyperparameter tuning
python scripts/tuning/tune_agent.py --config config/config.yaml --trials 50

# 7. Quick physics test
python -c "from environment.docomo_6g_env import DOCOMO_6G_Environment; print(' Environment working')"

# 8. View results
ls -la results/*/
cat results/*/training_log.txt | tail -20
```

### Directory Quick Access
```bash
# Key directories
cd config/          # Configuration files
cd environment/     # RL environment and physics
cd models/          # DQN and agent models  
cd simulator/       # Channel simulation
cd scripts/         # All executable scripts
cd tests/           # Test suite
cd results/         # Training results
cd docs/            # Documentation
```

### File Patterns
```bash
# Important file types
*.yaml              # Configuration files
*_env.py           # Environment implementations
*_model.py         # Neural network models
*_simulator.py     # Physics simulators
test_*.py          # Test files
train_*.py         # Training scripts
evaluate_*.py      # Evaluation scripts
plot_*.py          # Plotting scripts
verify_*.py        # Verification scripts
```

---

##  Performance Optimization Tips

1. **GPU Usage**: Always use `--device cuda` for faster training
2. **Batch Size**: Increase batch size if you have more GPU memory
3. **Episodes**: Start with fewer episodes (100-500) for testing
4. **Evaluation**: Use fewer evaluation episodes during training (`--eval-episodes 10`)
5. **Checkpointing**: Save checkpoints frequently (`--save-every 100`)
6. **Hyperparameters**: Use Optuna for systematic hyperparameter search
7. **Memory**: Monitor memory usage and adjust batch sizes accordingly
8. **Parallel**: Run multiple experiments in parallel if you have resources

This comprehensive guide covers every aspect of running and using the 6G OAM Deep Reinforcement Learning project. Bookmark this file and refer to specific sections as needed!
