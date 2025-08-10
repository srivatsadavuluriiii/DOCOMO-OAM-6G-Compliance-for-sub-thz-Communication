# Distance Optimization Guide for OAM 6G Systems

## Overview

The Distance Optimization system provides intelligent, distance-aware optimization strategies for OAM mode selection in 6G communication systems. This system enhances performance by automatically selecting optimal OAM modes based on distance, throughput, and system stability considerations.

## Key Concepts

### Distance Categories

The system categorizes distances into three main categories:

1. **Near Range (0-50m)**
   - Optimal for modes 1-3
   - High beam focus and low crosstalk
   - High throughput potential
   - Sensitive to pointing errors

2. **Medium Range (50-150m)**
   - Optimal for modes 4-6
   - Balanced performance characteristics
   - Moderate crosstalk
   - Good coverage and stability

3. **Far Range (150m+)**
   - Optimal for modes 7-8
   - Wide coverage area
   - Higher crosstalk but robust to pointing errors
   - Lower throughput but better range

### Optimization Strategy

The distance optimization system uses a multi-criteria optimization approach:

- **Distance Weight (30%)**: Optimizes for distance-based mode selection
- **Throughput Weight (50%)**: Prioritizes throughput performance
- **Stability Weight (20%)**: Reduces unnecessary handovers

## Architecture

### Core Components

1. **DistanceOptimizer**: Main optimization engine
2. **DistanceOptimizedEnv**: Enhanced environment with distance optimization
3. **DistanceAwareRewardCalculator**: Reward calculation with distance considerations
4. **DistanceOptimizationConfig**: Configuration management

### Class Hierarchy

```
DistanceOptimizer
├── DistanceOptimizationConfig
├── DistanceAwareRewardCalculator
└── DistanceOptimizedEnv
    └── OAM_Env (base)
```

## Configuration

### Distance Optimization Configuration

```yaml
# config/distance_optimization_config.yaml

# Distance thresholds
distance_thresholds:
  near_threshold: 50.0      # meters
  medium_threshold: 150.0    # meters
  far_threshold: 300.0       # meters

# Mode preferences by distance
mode_preferences:
  near_modes: [1, 2, 3]
  medium_modes: [4, 5, 6]
  far_modes: [7, 8]

# Optimization weights
optimization_weights:
  distance_weight: 0.3
  throughput_weight: 0.5
  stability_weight: 0.2

# Adaptive parameters
adaptive_parameters:
  enabled: true
  learning_rate: 0.01
```

### Integration with Base Configuration

The distance optimization system integrates with the existing OAM configuration:

```python
from utils.config_utils import load_config, merge_configs

# Load configurations
base_config = load_config('config/base_config_new.yaml')
distance_config = load_config('config/distance_optimization_config.yaml')

# Merge configurations
config = merge_configs(base_config, distance_config)
```

## Usage

### Basic Usage

```python
from environment.distance_optimized_env import DistanceOptimizedEnv
from environment.distance_optimizer import DistanceOptimizer

# Initialize environment with distance optimization
env = DistanceOptimizedEnv(config)

# Get distance optimization statistics
stats = env.get_distance_optimization_stats()
print(f"Optimization success rate: {stats['success_rate']:.3f}")

# Get performance by distance category
category_performance = env.get_distance_category_performance()
for category, performance in category_performance.items():
    print(f"{category}: {performance['avg_throughput']/1e6:.1f} Mbps")
```

### Training with Distance Optimization

```python
from scripts.training.train_distance_optimization import train_distance_optimization_agent

# Train agent with distance optimization
training_results = train_distance_optimization_agent(config, results_dir)

# Access training metrics
episode_rewards = training_results['episode_rewards']
final_stats = training_results['final_distance_optimization_stats']
```

### Analysis and Comparison

```python
from scripts.analysis.analyze_distance_optimization import main as run_analysis

# Run comprehensive analysis
run_analysis()
```

## Optimization Strategies

### 1. Threshold-Based Mode Selection

The system uses distance thresholds to categorize optimal modes:

```python
def get_optimal_mode_by_distance(self, distance: float, current_mode: int, available_modes: List[int]) -> int:
    category = self.get_distance_category(distance)
    
    if category == "near":
        preferred_modes = self.config.near_modes
    elif category == "medium":
        preferred_modes = self.config.medium_modes
    else:  # far
        preferred_modes = self.config.far_modes
    
    # Return first available preferred mode
    available_preferred = [m for m in preferred_modes if m in available_modes]
    return available_preferred[0] if available_preferred else current_mode
```

### 2. Multi-Criteria Optimization

The system combines multiple factors for optimal decision-making:

```python
optimization_score = (
    self.config.distance_weight * distance_score +
    self.config.throughput_weight * throughput_score +
    self.config.stability_weight * stability_score
)
```

### 3. Adaptive Thresholds

The system can adapt thresholds based on performance:

```python
def _update_adaptive_thresholds(self, distance: float, optimization_score: float):
    if optimization_score > 0.8:  # Good performance
        self.adaptive_medium_threshold += self.config.learning_rate * 10
    elif optimization_score < 0.4:  # Poor performance
        self.adaptive_medium_threshold -= self.config.learning_rate * 10
```

## Performance Metrics

### Key Performance Indicators

1. **Optimization Success Rate**: Percentage of successful optimizations
2. **Average Optimization Score**: Overall optimization performance
3. **Distance Category Performance**: Performance by distance range
4. **Handover Frequency**: Rate of mode changes
5. **Throughput Improvement**: Performance gains over baseline

### Monitoring and Analysis

```python
# Get comprehensive statistics
stats = env.get_distance_optimization_stats()

print(f"Total optimizations: {stats['total_optimizations']}")
print(f"Success rate: {stats['success_rate']:.3f}")
print(f"Average optimization score: {stats['average_optimization_score']:.3f}")

# Get performance by category
category_performance = env.get_distance_category_performance()
for category, performance in category_performance.items():
    print(f"{category}: {performance['avg_throughput']/1e6:.1f} Mbps")
```

## Advanced Features

### 1. Distance-Aware Reward Calculation

The system provides enhanced reward calculation that considers distance optimization:

```python
# Calculate distance-aware reward
reward, optimization_info = distance_aware_reward_calculator.calculate_distance_aware_reward(
    throughput, sinr_dB, handover_occurred, distance, current_mode, available_modes
)
```

### 2. Handover Optimization

The system includes intelligent handover management:

```python
# Check handover constraints
if len(self.mode_change_history) == 0 or not any(self.mode_change_history[-self.min_handover_interval:]):
    should_change = True
```

### 3. Performance Tracking

Comprehensive performance tracking and history:

```python
# Track optimization history
self.distance_optimization_history.append({
    'distance': current_distance,
    'mode': self.current_mode,
    'throughput': current_throughput,
    'optimization_scores': optimization_scores,
    'mode_change': mode_changed
})
```

## Configuration Options

### Distance Thresholds

```yaml
distance_thresholds:
  near_threshold: 50.0      # Adjust for your deployment
  medium_threshold: 150.0    # Typical indoor/outdoor boundary
  far_threshold: 300.0       # Long-range communication
```

### Mode Preferences

```yaml
mode_preferences:
  near_modes: [1, 2, 3]     # Lower modes for near distances
  medium_modes: [4, 5, 6]   # Balanced modes for medium distances
  far_modes: [7, 8]         # Higher modes for far distances
```

### Optimization Weights

```yaml
optimization_weights:
  distance_weight: 0.3       # Weight for distance optimization
  throughput_weight: 0.5     # Weight for throughput maximization
  stability_weight: 0.2      # Weight for stability (fewer handovers)
```

### Adaptive Parameters

```yaml
adaptive_parameters:
  enabled: true              # Enable adaptive threshold adjustment
  learning_rate: 0.01        # Learning rate for adaptation
  min_threshold: 50.0        # Minimum threshold value
  max_threshold: 400.0       # Maximum threshold value
```

## Best Practices

### 1. Configuration Tuning

- Start with default thresholds and adjust based on your deployment
- Monitor optimization success rate and adjust weights accordingly
- Use adaptive thresholds for dynamic environments

### 2. Performance Monitoring

- Regularly check optimization statistics
- Monitor distance category performance
- Track handover frequency and stability

### 3. Integration Guidelines

- Integrate with existing OAM environment seamlessly
- Use dependency injection for simulator components
- Maintain backward compatibility with existing configurations

### 4. Training Considerations

- Use sufficient episodes for convergence
- Monitor optimization scores during training
- Validate performance across different distance ranges

## Troubleshooting

### Common Issues

1. **Low Optimization Success Rate**
   - Check distance thresholds for your deployment
   - Adjust optimization weights
   - Verify mode availability

2. **Excessive Handovers**
   - Increase stability weight
   - Adjust handover interval constraints
   - Check optimization threshold

3. **Poor Performance in Specific Distance Ranges**
   - Review mode preferences for that range
   - Check adaptive threshold settings
   - Analyze category-specific performance

### Debugging Tools

```python
# Enable detailed logging
env.set_distance_optimization_enabled(True)

# Get detailed optimization info
optimized_mode, optimization_info = env._get_distance_optimized_mode(
    distance, throughput, current_mode
)

# Print optimization details
print(f"Distance category: {optimization_info['distance_category']}")
print(f"Optimization scores: {optimization_info['optimization_scores']}")
print(f"Should change mode: {optimization_info['should_change']}")
```

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - Neural network-based mode selection
   - Reinforcement learning for optimization
   - Predictive distance modeling

2. **Advanced Analytics**
   - Real-time performance monitoring
   - Predictive maintenance
   - Automated threshold adjustment

3. **Multi-Objective Optimization**
   - Energy efficiency considerations
   - Quality of Service guarantees
   - Fairness and resource allocation

### Extension Points

The system is designed for easy extension:

```python
# Custom optimization strategy
class CustomDistanceOptimizer(DistanceOptimizer):
    def optimize_mode_selection(self, distance, throughput, mode, available_modes):
        # Implement custom optimization logic
        pass

# Custom reward calculation
class CustomDistanceAwareRewardCalculator(DistanceAwareRewardCalculator):
    def calculate_distance_aware_reward(self, throughput, sinr, handover, distance, mode, available_modes):
        # Implement custom reward logic
        pass
```

## Conclusion

The Distance Optimization system provides a comprehensive solution for intelligent OAM mode selection based on distance considerations. By combining threshold-based selection with multi-criteria optimization and adaptive learning, it significantly improves system performance across different distance ranges.

The modular design allows for easy integration with existing OAM systems while providing extensive configuration options for different deployment scenarios. The comprehensive monitoring and analysis tools enable continuous optimization and performance improvement. 