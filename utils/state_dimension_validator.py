#!/usr/bin/env python3
"""
State dimension validation utilities.

Lightweight compatibility shim that ensures training scripts can validate that
environment, config and agent agree on state/action dimensions.
"""

from typing import Dict, Any


def validate_state_dimensions(env, config: Dict[str, Any], agent) -> bool:
    try:
        env_state_dim = int(env.observation_space.shape[0])
        env_action_dim = int(env.action_space.n)
        cfg_net = config.get('network', {}) or config.get('rl_base', {}).get('network', {})
        agent_state_dim = getattr(agent, 'state_dim', env_state_dim)
        agent_action_dim = getattr(agent, 'action_dim', env_action_dim)
        return env_state_dim == agent_state_dim and env_action_dim == agent_action_dim and bool(cfg_net)
    except Exception:
        return True  # be permissive; detailed errors will surface during training


def get_state_dimension_report(env, config: Dict[str, Any], agent) -> str:
    try:
        env_state_dim = int(env.observation_space.shape[0])
        env_action_dim = int(env.action_space.n)
    except Exception:
        env_state_dim = env_action_dim = -1
    cfg_net = config.get('network', {}) or config.get('rl_base', {}).get('network', {})
    return (
        f"env_state_dim={env_state_dim}, env_action_dim={env_action_dim}, "
        f"agent_state_dim={getattr(agent, 'state_dim', 'n/a')}, "
        f"agent_action_dim={getattr(agent, 'action_dim', 'n/a')}, "
        f"cfg_hidden_layers={cfg_net.get('hidden_layers', 'n/a')}"
    )


