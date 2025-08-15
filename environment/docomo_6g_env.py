#!/usr/bin/env python3
"""
DOCOMO 6G Environment - Main RL Environment
Complete 6G OAM environment with DOCOMO KPI compliance
Supports 1 Tbps peak, ultra-low latency, massive connectivity
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import yaml
import time
import math
from dataclasses import dataclass
from collections import deque
import warnings

                          
from .docomo_kpi_tracker import DOCOMOKPITracker, PerformanceMeasurement, DOCOMOKPIs
from .docomo_atmospheric_models import DOCOMOAtmosphericModels, AtmosphericParameters, AtmosphericCondition
from .ultra_high_mobility import UltraHighMobilityModel, MobilityState, BeamTrackingState

                                     
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
models_dir = os.path.join(project_root, 'models', 'docomo_6g')
sys.path.insert(0, project_root)

                                                         
try:
    from models.multi_objective_reward import MultiObjectiveReward
except ImportError:

    from ..models.multi_objective_reward import MultiObjectiveReward                

                                           
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from simulator.channel_simulator import ChannelSimulator
from environment.physics_calculator import PhysicsCalculator

class DOCOMOFrequencyBands:
    """DOCOMO 6G frequency bands with specifications"""
    
    BANDS = {
        'sub_6ghz': {
            'frequency': 6.0e9,
            'bandwidth': 100.0e6,
            'max_range_km': 10.0,
            'target_throughput_gbps': 1.0,
            'target_latency_ms': 1.0,
            'oam_modes': [1, 2, 3]
        },
        'mmwave_28': {
            'frequency': 28.0e9,
            'bandwidth': 800.0e6,
            'max_range_km': 1.0,
            'target_throughput_gbps': 10.0,
            'target_latency_ms': 0.5,
            'oam_modes': [1, 2, 3, 4]
        },
        'mmwave_39': {
            'frequency': 39.0e9,
            'bandwidth': 800.0e6,
            'max_range_km': 0.8,
            'target_throughput_gbps': 15.0,
            'target_latency_ms': 0.3,
            'oam_modes': [1, 2, 3, 4, 5]
        },
        'mmwave_60': {
            'frequency': 60.0e9,
            'bandwidth': 2.0e9,
            'max_range_km': 0.5,
            'target_throughput_gbps': 30.0,
            'target_latency_ms': 0.2,
            'oam_modes': [1, 2, 3, 4, 5, 6]
        },
        'sub_thz_100': {
            'frequency': 100.0e9,
            'bandwidth': 5.0e9,
            'max_range_km': 0.2,
            'target_throughput_gbps': 100.0,
            'target_latency_ms': 0.15,
            'oam_modes': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        'sub_thz_140': {
            'frequency': 140.0e9,
            'bandwidth': 10.0e9,
            'max_range_km': 0.15,
            'target_throughput_gbps': 200.0,
            'target_latency_ms': 0.1,
            'oam_modes': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        'sub_thz_220': {
            'frequency': 220.0e9,
            'bandwidth': 20.0e9,
            'max_range_km': 0.1,
            'target_throughput_gbps': 500.0,
            'target_latency_ms': 0.08,
            'oam_modes': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        'sub_thz_300': {
            'frequency': 300.0e9,
            'bandwidth': 50.0e9,
            'max_range_km': 0.05,
            'target_throughput_gbps': 50.0,  # Realistic target
            'target_latency_ms': 0.05,
            'oam_modes': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        'thz_600': {
            'frequency': 600.0e9,
            'bandwidth': 100.0e9,
            'max_range_km': 0.01,
            'target_throughput_gbps': 100.0,  # Realistic target
            'target_latency_ms': 0.01,
            'oam_modes': [1, 2, 3, 4, 5, 6, 7, 8]
        }
    }

class DOCOMO_6G_Environment(gym.Env):
    """
    DOCOMO-aligned 6G OAM environment with comprehensive KPI tracking
    
    State Space (19 dimensions):
    [0] SINR (dB)
    [1] Throughput (Gbps) 
    [2] Latency (ms)
    [3] Distance (m)
    [4] Velocity X (m/s)
    [5] Velocity Y (m/s)
    [6] Velocity Z (m/s)
    [7] Current Band (0-8)
    [8] Current OAM Mode (1-8)
    [9] Doppler Shift (Hz)
    [10] Atmospheric Loss (dB)
    [11] Energy Consumption (W)
    [12] Reliability Score (0-1)
    [13] Interference Level (dB)
    [14] Prediction Confidence (0-1)
    [15] Beam Alignment Error (deg)
    [16] Connection Count
    [17] Mobility Prediction (m/s change)
    [18] Weather Factor (0-1)
    
    Action Space (8 discrete actions):
    [0] STAY - Maintain current configuration
    [1] OAM_UP - Increase OAM mode
    [2] OAM_DOWN - Decrease OAM mode  
    [3] BAND_UP - Switch to higher frequency band
    [4] BAND_DOWN - Switch to lower frequency band
    [5] BEAM_TRACK - Optimize beam tracking
    [6] PREDICT - Enable mobility prediction
    [7] OPTIMIZE_BAND - Automatically switch to the optimal band for the current distance
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config_path: str = None, config: Dict[str, Any] = None, num_users: int = 1):
        """
        Initialize DOCOMO 6G environment
        
        Args:
            config_path: Path to DOCOMO configuration file
            config: Configuration dictionary (alternative to file)
            num_users: Number of simultaneous users (agents) in the environment
        """
        super().__init__()
        
        self.num_users = num_users
        self.agent_ids = [f"agent_{i}" for i in range(self.num_users)]
                            
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("Either config_path or config must be provided")
        
        self.docomo_config = self.config.get('docomo_6g_system', {})
        network_cfg = self.docomo_config.get('network', {})
        self.base_stations = [tuple(bs) for bs in network_cfg.get('base_stations', [[0, 0, 30]])]

        rl_training_cfg = (
            self.docomo_config.get('reinforcement_learning', {})
            .get('training', {})
        )
        self.high_freq_start_prob = float(rl_training_cfg.get('high_freq_start_prob', 0.9))  # Strongly favor THz bands
        self.optimal_start_band = bool(rl_training_cfg.get('optimal_start_band', False))  # Force probability-based for THz
        
        # Interference configuration (now also includes other agents as interferers)
        interference_cfg = self.docomo_config.get('interference', {})
        self.interference_enabled = interference_cfg.get('enabled', False)
        # num_interfering_users now refers to non-learning, external interferers
        self.num_external_interferers = interference_cfg.get('num_interfering_users', 0)
        self.external_interferers = []
        if self.interference_enabled and self.num_external_interferers > 0:
            self._initialize_external_interferers()
            
        # Load frequency bands - prioritize config over hardcoded defaults
        cfg_bands = self.docomo_config.get('frequency_bands', {})
        if isinstance(cfg_bands, dict) and cfg_bands:
            # Use ONLY config-specified bands (don't merge with hardcoded)
            self.frequency_bands = {}
            for band_name, band_cfg in cfg_bands.items():
                self.frequency_bands[band_name] = dict(band_cfg)
            print(f"  Using config-specified bands: {list(self.frequency_bands.keys())}")
        else:
            # Fallback to hardcoded bands if no config specified
            self.frequency_bands = dict(DOCOMOFrequencyBands.BANDS)
            print(f"  Using default hardcoded bands: {list(self.frequency_bands.keys())}")
                                             
        self.band_names = list(self.frequency_bands.keys())
        
        # Initialize per-agent components
        self.kpi_trackers: Dict[str, DOCOMOKPITracker] = {agent_id: DOCOMOKPITracker(self.config) for agent_id in self.agent_ids}
        self.mobility_models: Dict[str, UltraHighMobilityModel] = {agent_id: UltraHighMobilityModel(self.config) for agent_id in self.agent_ids}
        self.multi_objective_rewards: Dict[str, MultiObjectiveReward] = {agent_id: MultiObjectiveReward(self.config) for agent_id in self.agent_ids}
        
        self.atmospheric_models = DOCOMOAtmosphericModels() # Shared
        # Use first available band for physics calculator initialization
        first_band = list(self.frequency_bands.keys())[0]
        default_bandwidth = float(self.frequency_bands[first_band]['bandwidth'])
        self.physics_calculator = PhysicsCalculator(
            config=self.config,
            bandwidth=default_bandwidth
        ) # Shared
        self.channel_simulator = ChannelSimulator(config=self.config) # Shared
        
        # Network slicing configuration
        slicing_cfg = self.config.get('network_slicing', {})
        self.slicing_enabled = slicing_cfg.get('enabled', False)
        self.slice_types = slicing_cfg.get('slice_types', {})
        self.slice_selection_distribution = slicing_cfg.get('selection_distribution', {})
        # Per-agent current slice and QoS targets
        self.current_slice_names: Dict[str, str] = {agent_id: "default" for agent_id in self.agent_ids}
        self.current_qos_targets: Dict[str, Dict[str, float]] = {agent_id: {} for agent_id in self.agent_ids}
        
        # Initial speed configuration (config-driven)
        # docomo_6g_system.mobility.initial_speed_kmh supports:
        #   distribution: 'uniform' or 'normal'
        #   uniform: min, max
        #   normal: mean, std
        # Values are clamped to [0, max_speed_kmh]
        self.initial_speed_distribution = 'uniform'
        self.initial_speed_min_kmh = 0.0
        # cap default upper to mobility model's max
        try:
            self.initial_speed_max_kmh = float(getattr(list(self.mobility_models.values())[0], 'max_speed_kmh', 120.0))
        except Exception:
            self.initial_speed_max_kmh = 120.0
        self.initial_speed_mean_kmh = 60.0
        self.initial_speed_std_kmh = 20.0
        try:
            mobility_cfg = self.docomo_config.get('mobility', {}) if isinstance(self.docomo_config, dict) else {}
            init_cfg = mobility_cfg.get('initial_speed_kmh', {}) if isinstance(mobility_cfg, dict) else {}
            dist = str(init_cfg.get('distribution', self.initial_speed_distribution)).lower()
            if dist in ('uniform', 'normal'):
                self.initial_speed_distribution = dist
            if 'min' in init_cfg:
                self.initial_speed_min_kmh = float(init_cfg.get('min', self.initial_speed_min_kmh))
            if 'max' in init_cfg:
                self.initial_speed_max_kmh = float(init_cfg.get('max', self.initial_speed_max_kmh))
            if 'mean' in init_cfg:
                self.initial_speed_mean_kmh = float(init_cfg.get('mean', self.initial_speed_mean_kmh))
            if 'std' in init_cfg:
                self.initial_speed_std_kmh = float(init_cfg.get('std', self.initial_speed_std_kmh))
            # Ensure sensible bounds
            cap = float(getattr(list(self.mobility_models.values())[0], 'max_speed_kmh', self.initial_speed_max_kmh))
            self.initial_speed_min_kmh = max(0.0, min(self.initial_speed_min_kmh, cap))
            self.initial_speed_max_kmh = max(self.initial_speed_min_kmh, min(self.initial_speed_max_kmh, cap))
            self.initial_speed_mean_kmh = max(0.0, min(self.initial_speed_mean_kmh, cap))
            self.initial_speed_std_kmh = max(0.0, float(self.initial_speed_std_kmh))
        except Exception:
            pass
        
        # Initialize per-agent dynamic states
        # Use first available band from config instead of hardcoded mmwave_28
        first_band = list(self.frequency_bands.keys())[0] if self.frequency_bands else 'mmwave_28'
        self.current_bands: Dict[str, str] = {agent_id: first_band for agent_id in self.agent_ids}
        self.current_oam_modes: Dict[str, int] = {agent_id: 1 for agent_id in self.agent_ids}
        self.beam_alignment_errors_deg: Dict[str, float] = {agent_id: np.random.uniform(0.1, 2.0) for agent_id in self.agent_ids}
        self.beam_tracking_enabled_flags: Dict[str, bool] = {agent_id: False for agent_id in self.agent_ids}
        self.prediction_confidences: Dict[str, float] = {agent_id: 0.5 for agent_id in self.agent_ids}
        
        self._pending_band_dir: Dict[str, Optional[int]] = {agent_id: None for agent_id in self.agent_ids}
        self._pending_band_counter: Dict[str, int] = {agent_id: 0 for agent_id in self.agent_ids}
        self._last_band_switch_step: Dict[str, int] = {agent_id: -9999 for agent_id in self.agent_ids}
        self._band_stickiness_steps: Dict[str, int] = {agent_id: 0 for agent_id in self.agent_ids}
        self.state_histories: Dict[str, deque] = {agent_id: deque(maxlen=100) for agent_id in self.agent_ids}
        self.throughput_histories: Dict[str, deque] = {agent_id: deque(maxlen=50) for agent_id in self.agent_ids}
        self.handover_histories: Dict[str, deque] = {agent_id: deque(maxlen=20) for agent_id in self.agent_ids}
        self.episode_stats: Dict[str, Dict[str, Any]] = {agent_id: {
            'total_reward': 0.0,
            'peak_throughput_gbps': 0.0,
            'min_latency_ms': float('inf'),
            'handover_count': 0,
            'compliance_score': 0.0
        } for agent_id in self.agent_ids}
        
        # Use first available band for initialization  
        first_band_config = self.frequency_bands.get(first_band, {})
        
        self.channel_config = {
            'oam': {
                'min_mode': 1,
                'max_mode': 8,
            },
            'system': {
                'frequency': first_band_config.get('frequency', '28.0e9'),
                'bandwidth': first_band_config.get('bandwidth', '1.0e9'),
                'tx_power_dBm': first_band_config.get('power_dbm', 30.0),
            }
        }
        
        # Action Space: Each agent has 8 discrete actions
        self.action_space = spaces.Dict({agent_id: spaces.Discrete(8) for agent_id in self.agent_ids})
        
        # Observation Space (23 dimensions per agent + other agent info)
        # Original 19: SINR, Throughput, Latency, Distance, VelX, VelY, VelZ, Band, OAM, Doppler, AtmosLoss, Energy, Reliability, Interference, PredConfidence, BeamAlignError, ConnectionCount, Acceleration, WeatherFactor
        # New 4: TargetThroughput, TargetLatency, TargetReliability, TargetMobility (per agent)
        # Plus 3*num_other_agents for other agent positions (x,y,z)
        # Plus 1*num_other_agents for other agent current band (index)
        # Plus 1*num_other_agents for other agent current OAM mode
        # Total: 23 + (5 * (num_users - 1))
        
        single_agent_obs_dim = 23 # 19 base + 4 QoS
        other_agent_info_dim = 5 # x, y, z, band_idx, oam_mode
        total_obs_dim = single_agent_obs_dim + (other_agent_info_dim * (self.num_users - 1))
        
        single_obs_space = spaces.Box(
            low=np.array([-30.0, 0.0, 0.0, 1.0, -139.0, -139.0, -139.0,
                         0, 1, -1000.0, 0.0, 0.0, 0.0, -50.0,
                         0.0, 0.0, 0, -100.0, 0.0, # Original 19
                         0.0, 0.0, 0.0, 0.0]), # New 4 for QoS targets
            high=np.array([50.0, 2000.0, 10.0, 10000.0, 139.0, 139.0, 139.0,
                          8, 8, 1000.0, 50.0, 100.0, 1.0, 50.0,
                          1.0, 10.0, 1000000, 100.0, 1.0, # Original 19
                          2000.0, 100.0, 1.0, 500.0]), # New 4 for QoS targets
            dtype=np.float32
        )
        
        if self.num_users > 1:
            # Expand bounds for other agent info (position, band, oam_mode)
            low_other_agents = np.tile(np.array([-10000.0, -10000.0, -10000.0, 0, 1]), (self.num_users - 1))
            high_other_agents = np.tile(np.array([10000.0, 10000.0, 10000.0, 8, 8]), (self.num_users - 1))
            
            low = np.concatenate((single_obs_space.low, low_other_agents))
            high = np.concatenate((single_obs_space.high, high_other_agents))
            
            self.observation_space = spaces.Dict({
                agent_id: spaces.Box(low=low, high=high, dtype=np.float32) 
                for agent_id in self.agent_ids
            })
        else:
            self.observation_space = spaces.Dict({
                self.agent_ids[0]: single_obs_space
            })
        
        # Initial state (per-agent)
        self.step_count = 0
        self.episode_count = 0
        self.max_steps = self.config.get('simulation', {}).get('max_steps_per_episode', 2000)
                                                                        
        self.min_band_switch_interval = int(self.docomo_config.get('band_switch_optimization', {}).get('min_interval_steps', 3))
        self.band_time_to_trigger_steps = int(self.docomo_config.get('band_switch_optimization', {}).get('time_to_trigger_steps', 2))
        self.band_sinr_hysteresis_db = float(self.docomo_config.get('band_switch_optimization', {}).get('sinr_hysteresis_db', 1.0))
        self.band_switch_min_gain_gbps = float(self.docomo_config.get('band_switch_optimization', {}).get('min_gain_gbps', 1.0))
        self.min_band_dwell_steps = int(self.docomo_config.get('band_switch_optimization', {}).get('min_dwell_steps', 3))
        self._early_exemption_steps: int = int(self.docomo_config.get('band_switch_optimization', {}).get('early_exemption_steps', 10))
        self._early_exemption_remaining: int = 999999 if self._early_exemption_steps <= 0 else 3                                
        
        self._band_stickiness_window: int = int(self.docomo_config.get('band_switch_optimization', {}).get('stickiness_window', 20))
        self._band_stickiness_bonus: float = float(self.docomo_config.get('band_switch_optimization', {}).get('stickiness_bonus', 0.1))
        
        self.base_station_position = self.base_stations[0]
        
        self.atmospheric_params = AtmosphericParameters()
        self._update_atmospheric_conditions()
        
        # Do not return anything from __init__. Initial observation is provided by reset().
        
    def reconfigure(self, new_config: Dict[str, Any]):
        """Reconfigure the environment with new settings for curriculum learning."""
        self.config = new_config
        # Re-initialize all per-agent components
        for agent_id in self.agent_ids:
            self.kpi_trackers[agent_id] = DOCOMOKPITracker(self.config)
            self.mobility_models[agent_id] = UltraHighMobilityModel(self.config)
            self.multi_objective_rewards[agent_id] = MultiObjectiveReward(self.config)
        
        # Re-initialize shared components if their config changes (e.g., bands)
        self.frequency_bands = dict(DOCOMOFrequencyBands.BANDS)
        cfg_bands = self.docomo_config.get('frequency_bands', {})
        if isinstance(cfg_bands, dict) and cfg_bands:
            for band_name, band_cfg in cfg_bands.items():
                if band_name in self.frequency_bands:
                    for key in [
                        'frequency', 'bandwidth', 'max_range_km',
                        'target_throughput_gbps', 'target_latency_ms',
                        'oam_modes', 'antenna_gain_dbi', 'tx_power_dbm'
                    ]:
                        if key in band_cfg:
                            self.frequency_bands[band_name][key] = band_cfg[key]
                else:
                    self.frequency_bands[band_name] = band_cfg
        self.band_names = list(self.frequency_bands.keys())
        
        # Re-initialize physics and channel sim if needed (they take full config)
        self.physics_calculator = PhysicsCalculator(config=self.config)
        self.channel_simulator = ChannelSimulator(config=self.config)
        
        # Update initial speed distribution parameters if they changed
        # This logic is copied from __init__ to ensure reconfigurability
        mobility_cfg = self.docomo_config.get('mobility', {}) if isinstance(self.docomo_config, dict) else {}
        init_cfg = mobility_cfg.get('initial_speed_kmh', {}) if isinstance(mobility_cfg, dict) else {}
        dist = str(init_cfg.get('distribution', self.initial_speed_distribution)).lower()
        if dist in ('uniform', 'normal'):
            self.initial_speed_distribution = dist
        if 'min' in init_cfg:
            self.initial_speed_min_kmh = float(init_cfg.get('min', self.initial_speed_min_kmh))
        if 'max' in init_cfg:
            self.initial_speed_max_kmh = float(init_cfg.get('max', self.initial_speed_max_kmh))
        if 'mean' in init_cfg:
            self.initial_speed_mean_kmh = float(init_cfg.get('mean', self.initial_speed_mean_kmh))
        if 'std' in init_cfg:
            self.initial_speed_std_kmh = float(init_cfg.get('std', self.initial_speed_std_kmh))
        cap = float(getattr(list(self.mobility_models.values())[0], 'max_speed_kmh', self.initial_speed_max_kmh))
        self.initial_speed_min_kmh = max(0.0, min(self.initial_speed_min_kmh, cap))
        self.initial_speed_max_kmh = max(self.initial_speed_min_kmh, min(self.initial_speed_max_kmh, cap))
        self.initial_speed_mean_kmh = max(0.0, min(self.initial_speed_mean_kmh, cap))
        self.initial_speed_std_kmh = max(0.0, float(self.initial_speed_std_kmh))
        
        # Re-initialize network slicing parameters as well
        slicing_cfg = self.config.get('network_slicing', {})
        self.slicing_enabled = slicing_cfg.get('enabled', False)
        self.slice_types = slicing_cfg.get('slice_types', {})
        self.slice_selection_distribution = slicing_cfg.get('selection_distribution', {})
        
        # Re-initialize external interferers if their config changed
        interference_cfg = self.docomo_config.get('interference', {})
        self.interference_enabled = interference_cfg.get('enabled', False)
        self.num_external_interferers = interference_cfg.get('num_interfering_users', 0)
        if self.interference_enabled and self.num_external_interferers > 0:
            self._initialize_external_interferers()
        else:
            self.external_interferers = []
                                                
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state and return (observation, info)."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_count += 1
        
        observations = {}
        infos = {}
        
        # Reset per-agent states
        for agent_id in self.agent_ids:
            self._pending_band_dir[agent_id] = None
            self._pending_band_counter[agent_id] = 0
            self._last_band_switch_step[agent_id] = -9999
            self._band_stickiness_steps[agent_id] = 0
            
            self.beam_alignment_errors_deg[agent_id] = np.random.uniform(0.1, 2.0)
            self.beam_tracking_enabled_flags[agent_id] = False
            self.prediction_confidences[agent_id] = 0.5
            
            # Select network slice for this agent
            if self.slicing_enabled and self.slice_types and self.slice_selection_distribution:
                slice_names = list(self.slice_selection_distribution.keys())
                probabilities = list(self.slice_selection_distribution.values())
                probabilities_norm = np.array(probabilities) / np.sum(probabilities) # Normalize probabilities
                
                selected_slice = np.random.choice(slice_names, p=probabilities_norm)
                self.current_slice_names[agent_id] = selected_slice
                self.current_qos_targets[agent_id] = self.slice_types[selected_slice].get('qos_targets', {})
                self.multi_objective_rewards[agent_id].set_qos_targets(self.current_qos_targets[agent_id])
                self.multi_objective_rewards[agent_id].set_reward_weights(self.slice_types[selected_slice].get('reward_weights', {}))
            else:
                self.current_slice_names[agent_id] = "default"
                self.current_qos_targets[agent_id] = {}
                self.multi_objective_rewards[agent_id].set_qos_targets({})
                self.multi_objective_rewards[agent_id].set_reward_weights({})
                
            # Sample initial position close to base station for THz performance
            initial_distance = np.random.uniform(5.0, 20.0)  # Very close for THz consistency
        initial_angle = np.random.uniform(0, 2*np.pi)
        initial_position = (
            initial_distance * np.cos(initial_angle),
            initial_distance * np.sin(initial_angle),
                1.5
            )
            # Sample initial speed per episode using configured distribution
        try:
            cap = float(getattr(self.mobility_models[agent_id], 'max_speed_kmh', self.initial_speed_max_kmh))
        except Exception:
            cap = self.initial_speed_max_kmh
        if self.initial_speed_distribution == 'normal':
            initial_speed_kmh = float(np.random.normal(self.initial_speed_mean_kmh, self.initial_speed_std_kmh))
            if not np.isfinite(initial_speed_kmh):
                initial_speed_kmh = self.initial_speed_mean_kmh
            initial_speed_kmh = max(0.0, min(cap, initial_speed_kmh))
        else:
            lo = float(min(self.initial_speed_min_kmh, self.initial_speed_max_kmh))
            hi = float(max(self.initial_speed_min_kmh, self.initial_speed_max_kmh))
            initial_speed_kmh = float(np.random.uniform(lo, min(hi, cap)))
        initial_speed_ms = initial_speed_kmh / 3.6
        velocity_angle = np.random.uniform(0, 2*np.pi)
        initial_velocity = (
            initial_speed_ms * np.cos(velocity_angle),
            initial_speed_ms * np.sin(velocity_angle),
            0.0
        )
        self.mobility_models[agent_id].update_mobility_state(
        position=initial_position,
        velocity=initial_velocity,
        timestamp=time.time()
        )

            # Force THz probability-based selection for consistent high performance
        start_rand = np.random.random()
        if True:  # Enable optimal_start_band to force THz selection for lab
            start_pos = np.array(initial_position, dtype=float)
            distance_km = float(np.linalg.norm(start_pos)) / 1000.0
            per_band_tp = {}
            for b in self.band_names:
                try:
                    max_range_km = float(self.frequency_bands[b].get('max_range_km', 0.0))
                    if max_range_km and distance_km > max_range_km:
                        per_band_tp[b] = 0.0
                        continue
                            # Need to temporarily set current_band for _estimate_throughput_for_band
                    original_band = self.current_bands[agent_id]
                    self.current_bands[agent_id] = b
                    self._apply_band_to_simulator(agent_id) # Apply per-agent
                    per_band_tp[b] = self._estimate_throughput_for_band(agent_id, b)
                    self.current_bands[agent_id] = original_band # Restore
                    self._apply_band_to_simulator(agent_id) # Apply per-agent
                except Exception:
                    per_band_tp[b] = 0.0
            # Get high frequency bands dynamically (>= 100 GHz)
            high_freq_bands = [b for b in self.band_names 
                             if float(self.frequency_bands[b].get('frequency', 0)) >= 100e9]
            
            # If no high freq bands available, use all bands
            candidate_bands = high_freq_bands if high_freq_bands else self.band_names
            
            # For lab scenario (close distance), FORCE highest frequency bands
            if distance_km <= 0.01:  # Lab conditions 
                # FORCE 300+ GHz bands in lab for maximum throughput
                thz_candidates = [b for b in candidate_bands 
                                if float(self.frequency_bands[b].get('frequency', 0)) >= 300e9]
                if thz_candidates:
                    candidates = thz_candidates  # Use ONLY THz bands in lab
                else:
                    # Fallback to any available band if no THz
                    candidates = [b for b in candidate_bands if per_band_tp.get(b, 0.0) >= 1.0]
            else:
                candidates = [b for b in candidate_bands if per_band_tp.get(b, 0.0) >= 100.0]
            if candidates:
                # For lab, prefer highest frequency first
                if distance_km <= 0.01:  
                    chosen = max(candidates, key=lambda b: float(self.frequency_bands[b].get('frequency', 0)))
                else:
                    chosen = max(candidates, key=lambda b: per_band_tp.get(b, 0.0))
            else:
                # For lab, even in fallback prefer highest frequency
                if distance_km <= 0.01:  
                    best_candidate = max(candidate_bands, key=lambda b: float(self.frequency_bands[b].get('frequency', 0)))
                else:
                    best_candidate = max(candidate_bands, key=lambda b: per_band_tp.get(b, 0.0))
                
                if per_band_tp.get(best_candidate, 0.0) > 0.0:
                    chosen = best_candidate
                else:
                    # Last resort - for lab, prefer THz bands 
                    if distance_km <= 0.01:
                        thz_bands = [b for b in self.band_names if float(self.frequency_bands[b].get('frequency', 0)) >= 300e9]
                        chosen = thz_bands[0] if thz_bands else max(self.band_names, key=lambda b: float(self.frequency_bands[b].get('frequency', 0)))
                    else:
                        chosen = max(self.band_names, key=lambda b: per_band_tp.get(b, 0.0))
                    # Use first available band as fallback
                    fallback_band = list(self.frequency_bands.keys())[0]
                    self.current_bands[agent_id] = chosen if chosen else fallback_band
        else:
            # Select from available bands only (no hardcoded band names)
            available_bands = list(self.frequency_bands.keys())
            
            if start_rand < self.high_freq_start_prob:
                # Try to prefer higher frequency bands if available
                sorted_bands = sorted(available_bands, 
                                    key=lambda b: self.frequency_bands[b].get('frequency', 0), 
                                    reverse=True)
                # Use top 60% of frequency bands with higher probability
                high_freq_count = max(1, int(len(sorted_bands) * 0.6))
                high_freq_bands = sorted_bands[:high_freq_count]
                self.current_bands[agent_id] = np.random.choice(high_freq_bands)
            else:
                # Random selection from all available bands
                self.current_bands[agent_id] = np.random.choice(available_bands)

        # Clamp current OAM mode to available range for the band
        available_modes = self.frequency_bands[self.current_bands[agent_id]]['oam_modes']
        try:
            min_mode = int(min(available_modes))
            max_mode = int(max(available_modes))
            favored = [m for m in available_modes if m >= max_mode - 2]
            self.current_oam_modes[agent_id] = int(np.random.choice(favored if favored else available_modes))
            self.current_oam_modes[agent_id] = max(min_mode, min(max_mode, int(self.current_oam_modes[agent_id])))
        except Exception:
                    self.current_oam_modes[agent_id] = available_modes[-1] if isinstance(available_modes, list) and available_modes else 1
                
                # Apply band to simulator/physics (per-agent)
        self._apply_band_to_simulator(agent_id)
                
                # Clear histories (per-agent)
        self.state_histories[agent_id].clear()
        self.throughput_histories[agent_id].clear()
        self.handover_histories[agent_id].clear()
                
                # Reset episode stats (per-agent)
        self.episode_stats[agent_id] = {
            'total_reward': 0.0,
            'peak_throughput_gbps': 0.0,
            'min_latency_ms': float('inf'),
            'handover_count': 0,
            'compliance_score': 0.0
        }
        
        # Update atmospheric conditions (shared)
        self._update_atmospheric_conditions()

        # Compute initial performance and build observations for all agents
        all_performance_metrics = {agent_id: self._calculate_performance(agent_id) for agent_id in self.agent_ids}
        
        for agent_id in self.agent_ids:
            observation = self._get_observation(agent_id, all_performance_metrics)
            observations[agent_id] = observation
            
            infos[agent_id] = {
            'episode': self.episode_count,
            'docomo_compliance': True,
                'kpi_targets': self.kpi_trackers[agent_id].docomo_targets.__dict__,
                'frequency_band': self.current_bands[agent_id],
                'oam_mode': self.current_oam_modes[agent_id],
                'slice_name': self.current_slice_names[agent_id],
                'qos_targets': self.current_qos_targets[agent_id]
            }
        
        return observations, infos
    
    def _initialize_external_interferers(self):
        """Initialize the external interfering users."""
        self.external_interferers = []
        for _ in range(self.num_external_interferers):
            interferer_mobility = UltraHighMobilityModel(self.config)
            
            # Initialize interferer position and velocity
            initial_distance = np.random.uniform(50.0, 500.0)
            initial_angle = np.random.uniform(0, 2*np.pi)
            initial_position = (
                initial_distance * np.cos(initial_angle),
                initial_distance * np.sin(initial_angle),
                1.5
            )
            initial_speed_kmh = np.random.uniform(30, 120)
            initial_speed_ms = initial_speed_kmh / 3.6
            velocity_angle = np.random.uniform(0, 2*np.pi)
            initial_velocity = (
                initial_speed_ms * np.cos(velocity_angle),
                initial_speed_ms * np.sin(velocity_angle),
                0.0
            )
            interferer_mobility.update_mobility_state(
                position=initial_position,
                velocity=initial_velocity,
                timestamp=time.time()
            )
            self.external_interferers.append(interferer_mobility)

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Execute environment step for all agents.
        
        Args:
            actions: Dictionary of actions for each agent (agent_id: action)
            
        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
        """
        self.step_count += 1
        
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        # 1. Update mobility and atmospheric conditions (shared for external interferers, per-agent for main agents)
        for agent_id in self.agent_ids:
            self.mobility_models[agent_id].step_mobility(dt=0.1)
        if self.interference_enabled:
            for interferer in self.external_interferers:
                interferer.step_mobility(dt=0.1)
        self._update_atmospheric_conditions()
        
        # 2. Execute actions for each agent and calculate initial performance metrics
        all_performance_metrics = {agent_id: self._calculate_performance(agent_id) for agent_id in self.agent_ids}
        
        action_infos: Dict[str, Dict[str, Any]] = {}
        for agent_id, action in actions.items():
            action_infos[agent_id] = self._execute_action(agent_id, action, all_performance_metrics[agent_id])
        
        # 3. Recalculate performance after all actions (especially band changes) are applied
        all_performance_metrics = {agent_id: self._calculate_performance(agent_id) for agent_id in self.agent_ids}

        for agent_id in self.agent_ids:
            performance_metrics = all_performance_metrics[agent_id]
            action_info = action_infos[agent_id]
            
            # Create measurement for KPI tracker
        from datetime import datetime
        measurement = PerformanceMeasurement(
            timestamp=datetime.now(),
            throughput_gbps=performance_metrics['throughput_gbps'],
            latency_ms=performance_metrics['latency_ms'],
            sinr_db=performance_metrics['sinr_db'],
            distance_m=performance_metrics['distance_m'],
            mobility_kmh=performance_metrics['mobility_kmh'],
                band=self.current_bands[agent_id],
                oam_mode=self.current_oam_modes[agent_id],
            energy_consumption_w=performance_metrics['energy_consumption_w'],
            reliability_score=performance_metrics['reliability_score'],
            handover_count=action_info.get('handover_occurred', 0),
            beam_alignment_error_deg=performance_metrics['beam_alignment_error_deg'],
            doppler_shift_hz=performance_metrics['doppler_shift_hz'],
            atmospheric_loss_db=performance_metrics['atmospheric_loss_db']
        )
        
        compliance_scores = self.kpi_trackers[agent_id].update(measurement)
        
            # Get next observation (pass pre-calculated metrics and other agents' states)
        next_observation = self._get_observation(agent_id, all_performance_metrics)
        observations[agent_id] = next_observation
        
        # Prepare info for reward calculation
        reward_info = {
            **performance_metrics,
            **action_info,
            'compliance_scores': compliance_scores,
            'handover_occurred': action_info.get('handover_occurred', False),
            'handover_successful': action_info.get('handover_successful', True),
                'current_band': self.current_bands[agent_id],
                'current_oam_mode': self.current_oam_modes[agent_id],
                'slice_name': self.current_slice_names[agent_id],
                'qos_targets': self.current_qos_targets[agent_id]
            }
            
            # Calculate reward
        current_state_history = self.state_histories[agent_id]
        current_state_obs = current_state_history[-1] if current_state_history else np.zeros(self.observation_space[agent_id].shape[0])
        reward, reward_breakdown = self.multi_objective_rewards[agent_id].calculate(
                state=current_state_obs,
                action=actions[agent_id],
            next_state=next_observation,
            info=reward_info
        )
        rewards[agent_id] = reward
        
        self._update_episode_stats(agent_id, performance_metrics, reward, compliance_scores)
        
        self.state_histories[agent_id].append(next_observation.copy())
        self.throughput_histories[agent_id].append(performance_metrics['throughput_gbps'])
        
        terminated, truncated = self._check_termination(agent_id, performance_metrics)
        terminateds[agent_id] = terminated
        truncateds[agent_id] = truncated
        
        infos[agent_id] = {
            'step': self.step_count,
            'episode': self.episode_count,
            'performance_metrics': performance_metrics,
            'action_info': action_info,
            'reward_breakdown': reward_breakdown,
            'compliance_scores': compliance_scores,
            'docomo_compliance': compliance_scores.get('overall_current', 0.0) >= 0.95,
                'kpi_report': self.kpi_trackers[agent_id].get_current_kpis(),
                'mobility_stats': self.mobility_models[agent_id].get_mobility_statistics(),
            'atmospheric_conditions': self.atmospheric_params.__dict__,
                'episode_stats': self.episode_stats[agent_id].copy(),
            'terminated_reason': self._get_termination_reason(terminated, truncated, performance_metrics)
        }
        
        return observations, rewards, terminateds, truncateds, infos
    
    def _execute_action(self, agent_id: str, action: int, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specified action for a given agent and return action information"""
        action_info = {
            'action': action,
            'action_name': self._get_action_name(action),
            'handover_occurred': False,
            'handover_successful': True,
            'mode_changed': False,
            'band_changed': False,
            'beam_tracking_enabled': False
        }
        
        prev_mode = self.current_oam_modes[agent_id]
        prev_band = self.current_bands[agent_id]
        
        if action == 0:        
            pass              
            
        elif action == 1:          
            if self.current_oam_modes[agent_id] < 8:
                available_modes = self.frequency_bands[self.current_bands[agent_id]]['oam_modes']
                if self.current_oam_modes[agent_id] + 1 <= max(available_modes):
                    self.current_oam_modes[agent_id] += 1
                    action_info['mode_changed'] = True
                    
        elif action == 2:            
            if self.current_oam_modes[agent_id] > 1:
                available_modes = self.frequency_bands[self.current_bands[agent_id]]['oam_modes']
                if self.current_oam_modes[agent_id] - 1 >= min(available_modes):
                    self.current_oam_modes[agent_id] -= 1
                    action_info['mode_changed'] = True
                    
        elif action == 3:                                                                         
            current_band_idx = self.band_names.index(self.current_bands[agent_id])
            if current_band_idx < len(self.band_names) - 1:
                new_band = self.band_names[current_band_idx + 1]
                                                                                 
                if new_band == 'thz_600':
                    action_info['ignored_band_switch'] = True
                    return action_info
                                                                         
                early_exempt = False
                if (self._early_exemption_steps > 0 and self._early_exemption_remaining > 0 and self.step_count <= self._early_exemption_steps):
                                                                  
                    current_tp = self._estimate_throughput_for_band(agent_id, self.current_bands[agent_id])
                    candidate_tp = self._estimate_throughput_for_band(agent_id, new_band)
                    if candidate_tp - current_tp >= self.band_switch_min_gain_gbps:
                        early_exempt = True
                        self._early_exemption_remaining -= 1
                                                                    
                if not early_exempt:
                    if (self.step_count - self._last_band_switch_step[agent_id]) < max(self.min_band_switch_interval, self.min_band_dwell_steps):
                        action_info['ignored_band_switch'] = True
                        return action_info
                                                      
                if self._pending_band_dir[agent_id] == +1:
                    self._pending_band_counter[agent_id] += 1
                else:
                    self._pending_band_dir[agent_id] = +1
                    self._pending_band_counter[agent_id] = 1
                if not early_exempt and self._pending_band_counter[agent_id] < self.band_time_to_trigger_steps:
                    action_info['pending_band_switch'] = True
                    return action_info
                                                                      
                current_tp = self._estimate_throughput_for_band(agent_id, self.current_bands[agent_id])
                candidate_tp = self._estimate_throughput_for_band(agent_id, new_band)
                if candidate_tp - current_tp < self.band_switch_min_gain_gbps:
                    action_info['insufficient_tp_gain'] = candidate_tp - current_tp
                    return action_info
                                                                    
                distance_km = self.mobility_models[agent_id].current_state.position_x**2 + self.mobility_models[agent_id].current_state.position_y**2
                distance_km = np.sqrt(distance_km) / 1000.0
                
                                                                                                   
                max_range = self.frequency_bands[new_band]['max_range_km']
                
                if distance_km <= max_range:
                    self.current_bands[agent_id] = new_band
                                                                                               
                    available_modes = self.frequency_bands[new_band]['oam_modes']
                                                                           
                                                                                     
                    try:
                        min_mode = int(min(available_modes))
                        max_mode = int(max(available_modes))
                        favored = [m for m in available_modes if m >= max_mode - 2]
                        if favored:
                            self.current_oam_modes[agent_id] = int(np.random.choice(favored))
                        else:
                            self.current_oam_modes[agent_id] = max_mode
                                          
                        self.current_oam_modes[agent_id] = max(min_mode, min(max_mode, int(self.current_oam_modes[agent_id])))
                    except Exception:
                        self.current_oam_modes[agent_id] = available_modes[-1] if isinstance(available_modes, list) and available_modes else 1
                    
                    action_info['band_changed'] = True
                    action_info['handover_occurred'] = True
                    self._last_band_switch_step[agent_id] = self.step_count
                    self._pending_band_dir[agent_id] = None
                    self._pending_band_counter[agent_id] = 0
                    self._band_stickiness_steps[agent_id] = 0

                                                                    
                    self._apply_band_to_simulator(agent_id)
                    
                                                                            
                    try:
                        freq_val = self.frequency_bands[new_band].get('frequency', 28.0e9)
                        freq_ghz = float(freq_val) / 1e9
                    except Exception:
                        freq_ghz = 28.0
                    if freq_ghz >= 300:             
                        action_info['high_freq_bonus'] = 5.0
                    elif freq_ghz >= 140:                
                        action_info['high_freq_bonus'] = 3.0  
                    elif freq_ghz >= 100:                          
                        action_info['high_freq_bonus'] = 2.0
                    else:
                        action_info['high_freq_bonus'] = 1.0
                    
        elif action == 4:                        
            current_band_idx = self.band_names.index(self.current_bands[agent_id])
            if current_band_idx > 0:
                new_band = self.band_names[current_band_idx - 1]
                                                                    
                if (self.step_count - self._last_band_switch_step[agent_id]) < self.min_band_switch_interval:
                    action_info['ignored_band_switch'] = True
                    return action_info
                                                        
                if self._pending_band_dir[agent_id] == -1:
                    self._pending_band_counter[agent_id] += 1
                else:
                    self._pending_band_dir[agent_id] = -1
                    self._pending_band_counter[agent_id] = 1
                if self._pending_band_counter[agent_id] < self.band_time_to_trigger_steps:
                    action_info['pending_band_switch'] = True
                    return action_info
                                                                                       
                current_tp = self._estimate_throughput_for_band(agent_id, self.current_bands[agent_id])
                candidate_tp = self._estimate_throughput_for_band(agent_id, new_band)
                                                                                                 
                if current_tp > 5.0 and (current_tp - candidate_tp) > (self.band_switch_min_gain_gbps / 2.0):
                    action_info['downswitch_avoided_due_tp_loss'] = current_tp - candidate_tp
                    return action_info
                self.current_bands[agent_id] = new_band
                                                                        
                available_modes = self.frequency_bands[new_band]['oam_modes']
                try:
                    min_mode = int(min(available_modes))
                    max_mode = int(max(available_modes))
                    if self.current_oam_modes[agent_id] < min_mode:
                        self.current_oam_modes[agent_id] = min_mode
                    elif self.current_oam_modes[agent_id] > max_mode:
                        self.current_oam_modes[agent_id] = max_mode
                except Exception:
                    self.current_oam_modes[agent_id] = available_modes[0] if isinstance(available_modes, list) and available_modes else 1
                action_info['band_changed'] = True
                action_info['handover_occurred'] = True
                self._last_band_switch_step[agent_id] = self.step_count
                self._pending_band_dir[agent_id] = None
                self._pending_band_counter[agent_id] = 0
                self._band_stickiness_steps[agent_id] = 0

                                                                
                self._apply_band_to_simulator(agent_id)
                
        elif action == 5:                                           
                                                                     
            action_info['beam_tracking_enabled'] = True
                                                    
            self.beam_alignment_errors_deg[agent_id] *= 0.8
                                        
            action_info['beam_optimization_bonus'] = 1.5
            
        elif action == 6:                                          
                                                               
            self.mobility_models[agent_id].beam_prediction_enabled = True
                                                         
            velocity_vector = np.array([
                self.mobility_models[agent_id].current_state.velocity_x,
                self.mobility_models[agent_id].current_state.velocity_y,
                self.mobility_models[agent_id].current_state.velocity_z
            ])
                                                                      
            prediction_accuracy = np.exp(-np.linalg.norm(velocity_vector) / 50.0)                        
            self.beam_alignment_errors_deg[agent_id] *= (1.0 - 0.3 * prediction_accuracy)
            action_info['prediction_bonus'] = prediction_accuracy
            
        elif action == 7: # OPTIMIZE_BAND
            distance_m = performance_metrics['distance_m']
            optimal_band = self._get_optimal_band_for_distance(agent_id, distance_m)
            
            if self.current_bands[agent_id] != optimal_band:
                self.current_bands[agent_id] = optimal_band
                action_info['band_changed'] = True
                action_info['handover_occurred'] = True
                self._last_band_switch_step[agent_id] = self.step_count
                self._apply_band_to_simulator(agent_id)
        
                               
        if action_info['handover_occurred']:
            self.episode_stats[agent_id]['handover_count'] += 1
        else:
                                                                         
            self._band_stickiness_steps[agent_id] += 1
            if self._band_stickiness_steps[agent_id] >= self._band_stickiness_window:
                action_info['band_stickiness_bonus'] = self._band_stickiness_bonus
                                          
                self._band_stickiness_steps[agent_id] = 0
        
        return action_info

    def _estimate_throughput_for_band(self, agent_id: str, band_name: str) -> float:
        """Estimate instantaneous throughput (Gbps) if using the given band at current position.

        Temporarily applies the candidate band's settings, probes SINR, computes throughput,
        then restores the current band's configuration. Safe and lightweight for single-step estimates.
        """
        prev_band = self.current_bands[agent_id]
        try:
            # Switch to candidate band
            self.current_bands[agent_id] = band_name
            self._apply_band_to_simulator(agent_id)

            # Current position and mode (clamped within candidate band's supported modes)
            cs = self.mobility_models[agent_id].current_state
            pos = np.array([cs.position_x, cs.position_y, cs.position_z], dtype=float)

            oam_modes = self.frequency_bands[self.current_bands[agent_id]]['oam_modes']
            if isinstance(oam_modes, list) and oam_modes:
                min_mode = int(min(oam_modes))
                max_mode = int(max(oam_modes))
                cand_mode = int(max(min(self.current_oam_modes[agent_id], max_mode), min_mode))
            else:
                cand_mode = int(self.current_oam_modes[agent_id])

            # Consider interference from other agents and external interferers
            interferer_positions = []
            if self.interference_enabled:
                for other_agent_id in self.agent_ids:
                    if other_agent_id != agent_id:
                        interferer_positions.append(self.mobility_models[other_agent_id].current_state.position)
                interferer_positions.extend([interferer.current_state.position for interferer in self.external_interferers])
            
            _, sinr_db = self.channel_simulator.run_step(pos, cand_mode, cs.speed_kmh, interferer_positions=interferer_positions)

            # Convert SINR to throughput and return in Gbps
            tp_bps = self.physics_calculator.calculate_throughput(sinr_db) * 0.95
            return float(tp_bps / 1e9)
        except Exception:
            return 0.0
        finally:
            # Restore previous band configuration
            try:
                self.current_bands[agent_id] = prev_band
                self._apply_band_to_simulator(agent_id)
            except Exception:
                pass

    def _handle_handover_decision(self, agent_id: str) -> Tuple[bool, bool, Optional[int]]:
        """
        Checks if a handover is necessary and returns the decision for a given agent.
        Called when a handover-related action is considered.
        """
        candidate_base_stations = [bs for bs in self.base_stations if bs != self.base_station_position]
        if not candidate_base_stations:
            return False, True, None

        handover_decision = self.mobility_models[agent_id].should_trigger_handover(
            current_bs_position=self.base_station_position,
            candidate_bs_positions=candidate_base_stations,
            prediction_time_ms=50.0
        )
        
        if handover_decision['should_handover']:
            successful = np.random.random() > 0.05 # 95% success rate
            best_candidate_index = handover_decision['best_candidate_index']
            if successful and best_candidate_index is not None:
                self.base_station_position = candidate_base_stations[best_candidate_index]
            return True, successful, best_candidate_index
            
        return False, True, None

    def _update_mobility(self, agent_id: str):
        """Update mobility state by stepping the mobility model for a given agent."""
        self.mobility_models[agent_id].step_mobility(dt=0.1)
    
    def _update_atmospheric_conditions(self):
        """Update atmospheric conditions (could be weather-based) - shared for all agents"""
        
        self.atmospheric_params.temperature_c += np.random.normal(0, 0.1)
        self.atmospheric_params.humidity_percent = np.clip(
            self.atmospheric_params.humidity_percent + np.random.normal(0, 1), 0, 100
        )
        
        if np.random.random() < 0.01:                      
            conditions = list(AtmosphericCondition)
            self.atmospheric_params.condition = np.random.choice(conditions)
            
            if self.atmospheric_params.condition == AtmosphericCondition.LIGHT_RAIN:
                self.atmospheric_params.rain_rate_mm_h = np.random.uniform(1, 5)
            elif self.atmospheric_params.condition == AtmosphericCondition.MODERATE_RAIN:
                self.atmospheric_params.rain_rate_mm_h = np.random.uniform(5, 15)
            elif self.atmospheric_params.condition == AtmosphericCondition.HEAVY_RAIN:
                self.atmospheric_params.rain_rate_mm_h = np.random.uniform(15, 50)
            else:
                self.atmospheric_params.rain_rate_mm_h = 0.0
    
    def _calculate_performance(self, agent_id: str) -> Dict[str, float]:
        """Calculate comprehensive system performance metrics for a given agent"""
        current_state = self.mobility_models[agent_id].current_state
        distance_m = np.sqrt(current_state.position_x**2 + current_state.position_y**2)

        band_spec = self.frequency_bands[self.current_bands[agent_id]]
        frequency_hz = float(band_spec['frequency'])
        frequency_ghz = frequency_hz / 1e9
        bandwidth_hz = float(band_spec['bandwidth'])

        # Use realistic atmospheric losses instead of the overly pessimistic DOCOMO model
        distance_km = distance_m / 1000.0
        if frequency_ghz >= 300:  # THz bands
            realistic_atm_loss_db = 10.0 * distance_km  # 10 dB/km for THz
        elif frequency_ghz >= 100:  # Sub-THz bands  
            realistic_atm_loss_db = 2.0 * distance_km   # 2 dB/km for sub-THz
        elif frequency_ghz >= 60:   # mmWave high freq
            realistic_atm_loss_db = 0.5 * distance_km   # 0.5 dB/km for 60+ GHz
        elif frequency_ghz >= 28:   # mmWave
            realistic_atm_loss_db = 0.1 * distance_km   # 0.1 dB/km for mmWave
        else:  # Sub-6 GHz
            realistic_atm_loss_db = 0.01 * distance_km  # 0.01 dB/km for sub-6
        
        # Create realistic atmospheric losses dict for compatibility
        atmospheric_losses = {
            'total_mean_db': realistic_atm_loss_db,
            'molecular_total_db': realistic_atm_loss_db * 0.8,
            'weather_total_db': realistic_atm_loss_db * 0.2
        }

        beam_angles = self.mobility_models[agent_id].predict_beam_angles(
            base_station_position=self.base_station_position,
            prediction_time_ms=10.0
        )
        beam_alignment_error_deg = self.beam_alignment_errors_deg[agent_id] # Use stored value for consistency

        user_position = np.array([
            current_state.position_x,
            current_state.position_y,
            current_state.position_z,
        ], dtype=float)
        
        try:
            oam_modes = self.frequency_bands[self.current_bands[agent_id]].get('oam_modes', [1, 8])
            if isinstance(oam_modes, list) and oam_modes:
                min_mode = int(min(oam_modes))
                max_mode = int(max(oam_modes))
                if self.current_oam_modes[agent_id] < min_mode:
                    self.current_oam_modes[agent_id] = min_mode
                elif self.current_oam_modes[agent_id] > max_mode:
                    self.current_oam_modes[agent_id] = max_mode
        except Exception:
            pass
        
        # Calculate interference from other active agents and external interferers
        interferer_positions = []
        if self.interference_enabled:
            for other_agent_id in self.agent_ids:
                if other_agent_id != agent_id:
                    interferer_positions.append(self.mobility_models[other_agent_id].current_state.position)
            interferer_positions.extend([interferer.current_state.position for interferer in self.external_interferers])
        
        _, sinr_db = self.channel_simulator.run_step(user_position, int(self.current_oam_modes[agent_id]), current_state.speed_kmh, interferer_positions=interferer_positions)

        # Calculate throughput using band-specific bandwidth for accurate 500+ Gbps capability
        current_band = self.current_bands[agent_id]
        band_bandwidth_hz = float(self.frequency_bands[current_band]['bandwidth'])
        
        # Use enhanced throughput calculation with proper modulation and coding
        current_band_info = self.frequency_bands[current_band]
        frequency_hz = float(current_band_info['frequency'])  # Fix: convert string to float
        
        # Use the improved physics calculator for enhanced throughput
        self.physics_calculator.bandwidth = band_bandwidth_hz
        self.physics_calculator.current_oam_modes = int(self.current_oam_modes[agent_id])
        self.physics_calculator.current_distance_m = distance_m  # Pass distance for outdoor constraints
        
        throughput_result = self.physics_calculator.calculate_enhanced_throughput(
            sinr_dB=sinr_db,
            frequency=frequency_hz,
            modulation_scheme="adaptive",
            coding_rate=0.8,
            distance_m=distance_m,
            oam_modes=self.current_oam_modes[agent_id]
        )
        
        throughput_bps = throughput_result.get('practical_throughput', 0.0)
        throughput_gbps = throughput_bps / 1e9

        propagation_latency_ms = (distance_m / 3e8) * 1000                  
        processing_latency_ms = 0.03 + (self.current_oam_modes[agent_id] - 1) * 0.005
        if self.beam_tracking_enabled_flags[agent_id]:
            processing_latency_ms *= 0.9
        if getattr(self.mobility_models[agent_id], 'beam_prediction_enabled', False):
            processing_latency_ms *= 0.9
        switching_latency_ms = 0.0
        if self.step_count == self._last_band_switch_step[agent_id]:
            switching_latency_ms = 0.01
        total_latency_ms = propagation_latency_ms + processing_latency_ms + switching_latency_ms

        doppler_info = self.mobility_models[agent_id].calculate_doppler_shift(
            frequency_ghz=frequency_ghz,
            base_station_position=self.base_station_position
        )

        base_energy_w = 0.5
        frequency_energy_w = frequency_ghz / 300.0
        oam_energy_w = self.current_oam_modes[agent_id] * 0.05
        mobility_energy_w = current_state.speed_kmh / 500.0 * 0.5
        total_energy_w = base_energy_w + frequency_energy_w + oam_energy_w + mobility_energy_w
        
        if self._is_link_stable(agent_id):
            total_energy_w *= 0.9
        if self._is_using_optimal_band(agent_id, distance_m):
            total_energy_w *= 0.9
        if self.beam_tracking_enabled_flags[agent_id]:
            total_energy_w *= 0.95
        total_energy_w = max(total_energy_w, 0.1)

        if sinr_db > 20:
            reliability_score = 0.9999999
        elif sinr_db > 10:
            reliability_score = 0.999999
        elif sinr_db > 0:
            reliability_score = 0.99999
        else:
            reliability_score = 0.999
        
        reliability_score = max(0.0, reliability_score)

        path_loss_db = self.physics_calculator.calculate_path_loss(distance_m, frequency_hz)

        interference_db = 0.0 # This will be updated by channel sim

        return {
            'sinr_db': sinr_db,
            'throughput_gbps': throughput_gbps,
            'latency_ms': total_latency_ms,
            'distance_m': distance_m,
            'mobility_kmh': current_state.speed_kmh,
            'energy_consumption_w': total_energy_w,
            'reliability_score': reliability_score,
            'atmospheric_loss_db': atmospheric_losses['total_mean_db'],
            'beam_alignment_error_deg': beam_alignment_error_deg,
            'doppler_shift_hz': doppler_info['doppler_shift_hz'],
            'frequency_ghz': frequency_ghz,
            'bandwidth_mhz': bandwidth_hz / 1e6,
            'oam_crosstalk_db': -max(0, (self.current_oam_modes[agent_id] - 1) * 2.0),
            'path_loss_db': path_loss_db,
            'interference_db': interference_db,
            'in_atmospheric_window': self.atmospheric_models.is_atmospheric_window(frequency_ghz),
            'using_optimal_band': self._is_using_optimal_band(agent_id, distance_m),
            'link_stable': self._is_link_stable(agent_id),
            'error_rate': max(1e-9, 1.0 / (10**(sinr_db/10.0))),
            'handover_success_rate': self._calculate_handover_success_rate(agent_id),
            'mobility_prediction_accuracy': beam_angles['confidence'],
            'doppler_compensated': True,
            'beam_tracking_active': True,
            'multi_band_active': False,
            'oam_mode_efficiency': max(0.0, 1.0 - (self.current_oam_modes[agent_id] - 1) * 0.1)
        }
    
    def _get_observation(self, requesting_agent_id: str, all_performance_metrics: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Get current observation vector for a specific agent, including other agent info"""
        if not self.mobility_models[requesting_agent_id].mobility_history:
            num_obs_dims = self.observation_space[requesting_agent_id].shape[0]
            return np.zeros(num_obs_dims, dtype=np.float32)
        
        current_state = self.mobility_models[requesting_agent_id].current_state
        performance_metrics = all_performance_metrics[requesting_agent_id]
        
        # Base 19 + 4 QoS dimensions
        obs = np.array([
            performance_metrics['sinr_db'],
            performance_metrics['throughput_gbps'],
            performance_metrics['latency_ms'],
            performance_metrics['distance_m'],
            current_state.velocity_x,
            current_state.velocity_y,
            current_state.velocity_z,
            self.band_names.index(self.current_bands[requesting_agent_id]),
            self.current_oam_modes[requesting_agent_id],
            performance_metrics['doppler_shift_hz'],
            performance_metrics['atmospheric_loss_db'],
            performance_metrics['energy_consumption_w'],
            performance_metrics['reliability_score'],
            performance_metrics['interference_db'],
            performance_metrics['mobility_prediction_accuracy'],
            self.beam_alignment_errors_deg[requesting_agent_id], # Use per-agent beam error
            1, # Placeholder for Connection Count
            current_state.acceleration_magnitude,
            self._get_weather_factor(),
        ], dtype=np.float32)
        
        # Append QoS targets
        qos_obs = np.array([
            self.current_qos_targets[requesting_agent_id].get('min_throughput_gbps', 0.0),
            self.current_qos_targets[requesting_agent_id].get('max_latency_ms', 0.0),
            self.current_qos_targets[requesting_agent_id].get('min_reliability', 0.0),
            self.current_qos_targets[requesting_agent_id].get('max_mobility_kmh', 0.0)
        ], dtype=np.float32)
        obs = np.concatenate((obs, qos_obs))
        
        # Append other agents' information
        if self.num_users > 1:
            other_agents_info = []
            for other_agent_id in self.agent_ids:
                if other_agent_id != requesting_agent_id:
                    other_agent_state = self.mobility_models[other_agent_id].current_state
                    other_agent_band_idx = self.band_names.index(self.current_bands[other_agent_id])
                    other_agent_oam_mode = self.current_oam_modes[other_agent_id]
                    other_agents_info.extend([
                        other_agent_state.position_x,
                        other_agent_state.position_y,
                        other_agent_state.position_z,
                        float(other_agent_band_idx),
                        float(other_agent_oam_mode)
                    ])
            obs = np.concatenate((obs, np.array(other_agents_info, dtype=np.float32)))
        
        return obs
    
    def _is_using_optimal_band(self, agent_id: str, distance_m: float) -> bool:
        """Check if using optimal frequency band for current distance for a given agent"""
        optimal_band = self._get_optimal_band_for_distance(agent_id, distance_m)
        return self.current_bands[agent_id] == optimal_band
    
    def _get_optimal_band_for_distance(self, agent_id: str, distance_m: float) -> str:
        """Get optimal frequency band for given distance for a given agent"""
        distance_km = distance_m / 1000.0
        
        for band_name in reversed(self.band_names):                                
            if distance_km <= self.frequency_bands[band_name]['max_range_km']:
                return band_name
        
        return self.band_names[0]                                
    
    def _is_link_stable(self, agent_id: str) -> bool:
        """Check if link is stable based on recent performance for a given agent"""
        if len(self.throughput_histories[agent_id]) < 10:
            return True                                           
        
        recent_throughput = list(self.throughput_histories[agent_id])[-10:]
        throughput_std = np.std(recent_throughput)
        throughput_mean = np.mean(recent_throughput)
        
        if throughput_mean > 0:
            cv = throughput_std / throughput_mean
            return cv < 0.2                                          
        
        return True
    
    def _calculate_handover_success_rate(self, agent_id: str) -> float:
        """Calculate recent handover success rate for a given agent"""
        if not self.handover_histories[agent_id]:
            return 1.0                    
        
        recent_handovers = list(self.handover_histories[agent_id])[-10:]                     
        if not recent_handovers:
            return 1.0
        
        successful = sum(1 for h in recent_handovers if h['successful'])
        return successful / len(recent_handovers)
    
    def _get_weather_factor(self) -> float:
        """Get weather impact factor (0=bad, 1=good) - shared for all agents"""
        if self.atmospheric_params.condition == AtmosphericCondition.CLEAR:
            return 1.0
        elif self.atmospheric_params.condition in [AtmosphericCondition.LIGHT_RAIN, AtmosphericCondition.FOG_LIGHT]:
            return 0.8
        elif self.atmospheric_params.condition in [AtmosphericCondition.MODERATE_RAIN, AtmosphericCondition.FOG_DENSE]:
            return 0.5
        elif self.atmospheric_params.condition == AtmosphericCondition.HEAVY_RAIN:
            return 0.2
        else:
            return 0.7
    
    def _update_episode_stats(self, agent_id: str, performance: Dict[str, float], reward: float, compliance: Dict[str, float]):
        """Update episode-level statistics for a given agent"""
        self.episode_stats[agent_id]['total_reward'] += reward
        self.episode_stats[agent_id]['peak_throughput_gbps'] = max(
            self.episode_stats[agent_id]['peak_throughput_gbps'],
            performance['throughput_gbps']
        )
        self.episode_stats[agent_id]['min_latency_ms'] = min(
            self.episode_stats[agent_id]['min_latency_ms'],
            performance['latency_ms']
        )
        self.episode_stats[agent_id]['compliance_score'] = compliance.get('overall_current', 0.0)
    
    def _check_termination(self, agent_id: str, performance: Dict[str, float]) -> Tuple[bool, bool]:
        """Check if episode should terminate for a given agent"""
        terminated = False
        truncated = False
        
        if self.step_count >= self.max_steps:
            truncated = True
        
        # More lenient termination conditions to allow learning
        if performance['sinr_db'] < -30.0:  # Allow lower SINR                     
            terminated = True
        elif performance['throughput_gbps'] < 0.001 and self.step_count > 50:  # Only terminate on very low throughput after some steps                           
            terminated = True
        elif performance['distance_m'] > 15000.0:  # Increase distance limit                             
            terminated = True
        
        return terminated, truncated
    
    def _get_termination_reason(self, terminated: bool, truncated: bool, performance: Dict[str, float]) -> str:
        """Get human-readable termination reason"""
        if terminated:
            if performance['sinr_db'] < -20.0:
                return "Signal quality too poor"
            elif performance['throughput_gbps'] < 0.01:
                return "Insufficient throughput"
            elif performance['distance_m'] > 10000.0:
                return "Out of coverage range"
            else:
                return "System failure"
        elif truncated:
            return "Maximum steps reached"
        else:
            return "Episode ongoing"
    
    def _get_action_name(self, action: int) -> str:
        """Get human-readable action name"""
        action_names = [
            'STAY', 'OAM_UP', 'OAM_DOWN', 'BAND_UP', 'BAND_DOWN',
            'BEAM_TRACK', 'PREDICT', 'OPTIMIZE_BAND'
        ]
        return action_names[action] if 0 <= action < len(action_names) else 'UNKNOWN'
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n DOCOMO 6G Environment - Step {self.step_count} (Multi-Agent)")
            for agent_id in self.agent_ids:
                current_kpis = self.kpi_trackers[agent_id].get_current_kpis()
                compliance = self.kpi_trackers[agent_id].get_compliance_score()
                
                print(f"  Agent {agent_id}: ")
                print(f"    Band: {self.current_bands[agent_id]} | OAM Mode: {self.current_oam_modes[agent_id]}")
                print(f"    Throughput: {current_kpis.get('current_throughput_gbps', 0):.2f} Gbps")
                print(f"    Latency: {current_kpis.get('current_latency_ms', 0):.3f} ms") 
                print(f"    Speed: {current_kpis.get('current_mobility_kmh', 0):.1f} km/h")
                print(f"    Compliance: {compliance.get('overall_current', 0)*100:.1f}%")
            
        return None
    
    def close(self):
        """Clean up environment"""
        pass
    
    def get_docomo_report(self, agent_id: str) -> Dict[str, Any]:
        """Generate comprehensive DOCOMO compliance report for a given agent"""
        return self.kpi_trackers[agent_id].get_docomo_report()
    
    def save_docomo_report(self, agent_id: str, filepath: str):
        """Save DOCOMO compliance report to file for a given agent"""
        self.kpi_trackers[agent_id].save_report(filepath)

    def _apply_band_to_simulator(self, agent_id: str) -> None:
        """Apply current band's frequency/bandwidth/power to simulator and physics calculator for a given agent."""
        band_spec = self.frequency_bands[self.current_bands[agent_id]]
        system_cfg = {
            'frequency': float(band_spec.get('frequency', 28.0e9)),
            'bandwidth': float(band_spec.get('bandwidth', 400e6)),
            'tx_power_dBm': float(band_spec.get('tx_power_dbm', getattr(self.channel_simulator, 'tx_power_dBm', 30.0)))
        }
        
        antenna_gain = band_spec.get('antenna_gain_dbi', None)
        if antenna_gain is not None:
            system_cfg['tx_antenna_gain_dBi'] = float(antenna_gain)
            system_cfg['rx_antenna_gain_dBi'] = float(antenna_gain)

        oam_modes = band_spec.get('oam_modes', [1, 8])
        min_mode = min(oam_modes) if isinstance(oam_modes, list) and oam_modes else 1
        max_mode = max(oam_modes) if isinstance(oam_modes, list) and oam_modes else 8

        cfg = {
            'system': system_cfg,
            'oam': {
                'min_mode': int(min_mode),
                'max_mode': int(max_mode)
            }
        }
        
        try:
            self.channel_simulator._update_config(cfg)
            
            self.channel_simulator.wavelength = 3e8 / self.channel_simulator.frequency
            self.channel_simulator.k = 2 * np.pi / self.channel_simulator.wavelength
            self.channel_simulator.num_modes = self.channel_simulator.max_mode - self.channel_simulator.min_mode + 1
            self.channel_simulator.H = np.eye(self.channel_simulator.num_modes, dtype=complex)
        except Exception:
            
            pass

        try:
            self.physics_calculator.bandwidth = float(system_cfg['bandwidth'])
            if hasattr(self.physics_calculator, 'reset_cache'):
                self.physics_calculator.reset_cache()
        except Exception:
            pass
