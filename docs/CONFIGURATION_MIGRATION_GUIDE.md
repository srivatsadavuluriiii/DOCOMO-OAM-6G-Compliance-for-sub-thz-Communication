# Configuration Migration Guide

## Overview

This guide explains how to migrate from the old configuration files to the new hierarchical configuration system.

## Old Configuration System

The old configuration system used multiple flat YAML files:

- `config/rl_params.yaml` - Basic RL parameters
- `config/stable_reward_params.yaml` - Stable reward parameters
- `config/simulation_params.yaml` - Simulation parameters
- `config/extended_training_config.yaml` - Extended training parameters

These files had significant duplication and inconsistencies.

## New Hierarchical Configuration System

The new system uses a hierarchical approach with inheritance:

- `config/base_config_new.yaml` - Base configuration with common parameters
- `config/rl_config_new.yaml` - RL-specific parameters (inherits from base)
- `config/stable_reward_config_new.yaml` - Stable reward parameters (inherits from rl)
- `config/extended_config_new.yaml` - Extended training parameters (inherits from rl)

## How to Use the New System

### In Python Code

```python
from utils.hierarchical_config import load_hierarchical_config

# Load a configuration with inheritance
config = load_hierarchical_config("rl_config_new")  # Note: no .yaml extension
```

### Command Line

```bash
# Train with the new RL config
python scripts/main.py train --config rl_config_new

# Train with stable rewards
python scripts/main.py train-stable --config rl_config_new --stable-config stable_reward_config_new

# Extended training
python scripts/main.py train --config extended_config_new
```

## Benefits of the New System

1. **Single Source of Truth** - Common parameters are defined once in the base config
2. **Reduced Duplication** - No more copy-pasting parameters between files
3. **Consistent Values** - Parameters like `max_mode` are guaranteed to be consistent
4. **Clearer Structure** - Each config file has a clear purpose and inheritance path
5. **Easier Maintenance** - Changes to common parameters only need to be made in one place

## Backward Compatibility

The system still supports loading legacy .yaml files directly:

```python
from utils.config_utils import load_config

# Load a legacy config file
legacy_config = load_config("path/to/legacy_config.yaml")
```

## Migration Script

A migration script is available to help update your code:

```bash
python scripts/config_migration.py --update-scripts
```

This will update imports and config loading in the training scripts.
