#!/usr/bin/env python3
"""
Migration script to update from old configuration files to the new hierarchical system.

This script provides utilities to migrate from the old configuration files
(rl_params.yaml, stable_reward_params.yaml) to the new hierarchical configuration
system (base_config_new.yaml, rl_config_new.yaml, stable_reward_config_new.yaml).
"""

import os
import sys
import shutil
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from utils.hierarchical_config import HierarchicalConfig, load_hierarchical_config
from utils.config_utils import load_config, save_config


def create_backup(file_path: str) -> str:
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to the file to back up
        
    Returns:
        Path to the backup file
    """
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    return backup_path


def get_config_mapping() -> Dict[str, str]:
    """
    Get mapping from old config files to new hierarchical config files.
    
    Returns:
        Dictionary mapping old config files to new config files
    """
    return {
        "config/rl_params.yaml": "rl_config_new",
        "config/stable_reward_params.yaml": "stable_reward_config_new",
        "config/extended_training_config.yaml": "extended_config_new",
    }


def update_script(script_path: str, dry_run: bool = False) -> List[str]:
    """
    Update a script to use the new hierarchical configuration system.
    
    Args:
        script_path: Path to the script to update
        dry_run: If True, don't actually modify the file
        
    Returns:
        List of changes made to the script
    """
    with open(script_path, 'r') as f:
        content = f.read()
    
    changes = []
    config_mapping = get_config_mapping()
    
    # Replace imports if needed
    if "from utils.config_utils import load_config" in content and "from utils.hierarchical_config import" not in content:
        new_content = content.replace(
            "from utils.config_utils import load_config",
            "from utils.config_utils import load_config\nfrom utils.hierarchical_config import load_hierarchical_config"
        )
        changes.append("Added import for hierarchical config")
        content = new_content
    
    # Replace default config paths
    for old_path, new_name in config_mapping.items():
        if f'default="{old_path}"' in content:
            new_content = content.replace(
                f'default="{old_path}"',
                f'default="{new_name}"'
            )
            changes.append(f"Updated default config path: {old_path} -> {new_name}")
            content = new_content
    
    # Replace load_config calls with load_hierarchical_config
    for old_path, new_name in config_mapping.items():
        if f'load_config("{old_path}")' in content:
            new_content = content.replace(
                f'load_config("{old_path}")',
                f'load_hierarchical_config("{new_name}")'
            )
            changes.append(f"Updated config loading: {old_path} -> {new_name}")
            content = new_content
        
        if f"load_config(args.config)" in content and old_path in content:
            new_content = content.replace(
                "load_config(args.config)",
                "load_hierarchical_config(args.config) if not args.config.endswith('.yaml') else load_config(args.config)"
            )
            changes.append("Updated config loading to support both hierarchical and legacy configs")
            content = new_content
    
    # Write the updated content if not a dry run
    if not dry_run and changes:
        backup_path = create_backup(script_path)
        print(f"Created backup at {backup_path}")
        
        with open(script_path, 'w') as f:
            f.write(content)
    
    return changes


def update_all_scripts(dry_run: bool = False) -> Dict[str, List[str]]:
    """
    Update all training scripts to use the new hierarchical configuration system.
    
    Args:
        dry_run: If True, don't actually modify the files
        
    Returns:
        Dictionary mapping script paths to lists of changes made
    """
    script_paths = [
        "scripts/training/train_rl.py",
        "scripts/training/train_stable_rl.py",
        "scripts/main.py"
    ]
    
    all_changes = {}
    
    for script_path in script_paths:
        changes = update_script(script_path, dry_run)
        all_changes[script_path] = changes
        
        if changes:
            print(f"Updated {script_path}:")
            for change in changes:
                print(f"  - {change}")
        else:
            print(f"No changes needed for {script_path}")
    
    return all_changes


def create_migration_guide() -> None:
    """Create a migration guide markdown file."""
    guide_content = """# Configuration Migration Guide

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
"""
    
    guide_path = "docs/CONFIGURATION_MIGRATION_GUIDE.md"
    with open(guide_path, 'w') as f:
        f.write(guide_content)
    
    print(f"Created migration guide at {guide_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Migrate from old config files to new hierarchical system")
    
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify any files")
    parser.add_argument("--update-scripts", action="store_true", help="Update training scripts to use the new config system")
    parser.add_argument("--create-guide", action="store_true", help="Create a migration guide markdown file")
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if args.update_scripts:
        update_all_scripts(args.dry_run)
    
    if args.create_guide:
        create_migration_guide()
    
    if not (args.update_scripts or args.create_guide):
        print("No actions specified. Use --update-scripts or --create-guide.")
        print("Example: python config_migration.py --update-scripts --create-guide")


if __name__ == "__main__":
    main()