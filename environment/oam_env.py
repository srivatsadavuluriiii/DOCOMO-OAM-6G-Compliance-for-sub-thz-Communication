import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List, Optional, Protocol
import math
import os
import sys

# Ensure project root is on sys.path before importing project modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()
from utils.exception_handler import safe_calculation, graceful_degradation, get_exception_handler

from simulator.channel_simulator import ChannelSimulator
from environment.physics_calculator import PhysicsCalculator
from environment.mobility_model import MobilityModel
from environment.reward_calculator import RewardCalculator


class SimulatorInterface(Protocol):
    """
    Interface for channel simulators.
    
    This defines the contract that any simulator must implement,
    enabling dependency injection and easier testing.
    """
    
    def run_step(self, position: np.ndarray, current_mode: int) -> Tuple[np.ndarray, float]:
        """
        Run a simulation step.
        
        Args:
            position: Current position [x, y, z]
            current_mode: Current OAM mode
            
        Returns:
            Tuple of (channel_matrix, sinr_dB)
        """
        ...


class OAM_Env(gym.Env):
    """
    Gymnasium environment for OAM mode handover decisions.
    
    This environment simulates a user moving in a wireless network with OAM-based
    transmission. The agent must decide when to switch OAM modes to maximize throughput
    while minimizing unnecessary handovers.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, simulator: Optional[SimulatorInterface] = None):
        """
        Initialize the OAM environment.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary containing environment parameters.
                If None, uses default values. Expected keys:
                - 'system': System parameters (frequency, bandwidth, etc.)
                - 'environment': Environment parameters (humidity, temperature, etc.)
                - 'oam': OAM parameters (min_mode, max_mode, beam_width)
            simulator (Optional[SimulatorInterface]): Channel simulator instance for dependency injection.
                If None, creates default ChannelSimulator.
                
        Returns:
            OAM_Env: Initialized environment instance.
            
        Example:
            >>> env = OAM_Env()  # Default configuration
            >>> config = {'oam': {'min_mode': 1, 'max_mode': 6}}
            >>> env = OAM_Env(config)  # Custom configuration
        """
        super(OAM_Env, self).__init__()
        
        # Default parameters
        self.distance_min = 50.0  # meters
        self.distance_max = 300.0  # meters
        self.velocity_min = 1.0  # m/s
        self.velocity_max = 5.0  # m/s
        self.area_size = np.array([500.0, 500.0])  # meters [x, y]
        self.pause_time_max = 5.0  # seconds
        self.min_mode = None  # Will be set from config
        self.max_mode = None  # Will be set from config
        
        # Time step in seconds
        self.dt = 0.1
        
        # Update with provided config
        if config:
            self._update_config(config)
        
        # Validate basic environment parameters
        self._validate_environment_parameters()
        
        # Initialize the channel simulator (dependency injection)
        if simulator is not None:
            self.simulator = simulator
        else:
            # Default to ChannelSimulator if no simulator provided
            self.simulator = ChannelSimulator(config)
        
        # Initialize separated components
        self.physics_calculator = PhysicsCalculator(
            bandwidth=getattr(self.simulator, 'bandwidth', 400e6)
        )
        self.mobility_model = MobilityModel(config)
        self.reward_calculator = RewardCalculator(config)
        
        # OPTIMIZED: Add throughput caching for training efficiency
        self._throughput_cache = {}
        self._last_sinr = None
        self._last_throughput = None
        
        # Validate separated components
        self._validate_separated_components()
        
        # Initialize current_mode to prevent None errors
        if self.min_mode is not None and self.max_mode is not None:
            self.current_mode = (self.min_mode + self.max_mode) // 2
        else:
            self.current_mode = 1  # Default fallback
        
        # Action space: 0 = STAY, 1 = UP, 2 = DOWN
        self.action_space = spaces.Discrete(3)
        
        # State space: [SINR, distance, velocity_x, velocity_y, velocity_z, current_mode, min_mode, max_mode]
        low = np.array(
            [-30.0,                 # Minimum SINR in dB
             self.distance_min,     # Minimum distance
             -self.velocity_max,    # Minimum velocity in x
             -self.velocity_max,    # Minimum velocity in y
             -self.velocity_max,    # Minimum velocity in z
             self.min_mode,         # Minimum OAM mode
             self.min_mode,         # Minimum possible mode
             self.min_mode],        # Minimum of max mode (doesn't change)
            dtype=np.float32
        )
        
        high = np.array(
            [50.0,                  # Maximum SINR in dB
             self.distance_max,     # Maximum distance
             self.velocity_max,     # Maximum velocity in x
             self.velocity_max,     # Maximum velocity in y
             self.velocity_max,     # Maximum velocity in z
             self.max_mode,         # Maximum OAM mode
             self.max_mode,         # Maximum of min mode (doesn't change)
             self.max_mode],        # Maximum possible mode
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # Initialize state variables
        self.position = None
        self.velocity = None
        self.current_mode = None
        self.current_sinr = None
        self.target_position = None
        self.pause_time = 0
        
        # Episode tracking
        self.steps = 0
        self.max_steps = 1000
        self.episode_throughput = 0.0
        self.episode_handovers = 0
        # Hysteresis and stickiness
        self.last_switch_step = -9999
        # Handover optimization (can be overridden via config)
        self.min_steps_between_switches = 8
        self.time_to_trigger_steps = 8
        self.sinr_hysteresis_db = 2.0
        self._pending_switch_dir: Optional[int] = None  # +1 for UP, -1 for DOWN
        self._pending_switch_counter: int = 0
        self.stickiness_window = 10
        self.stickiness_counter = 0
        self.last_mode_seen = None
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update environment parameters from configuration.
        
        Args:
            config: Dictionary containing environment parameters
        """
        if 'environment' in config:
            env_config = config['environment']
            self.distance_min = env_config.get('distance_min', self.distance_min)
            self.distance_max = env_config.get('distance_max', self.distance_max)
        
        if 'mobility' in config:
            mob_config = config['mobility']
            self.velocity_min = mob_config.get('velocity_min', self.velocity_min)
            self.velocity_max = mob_config.get('velocity_max', self.velocity_max)
            self.area_size = np.array(mob_config.get('area_size', self.area_size))
            self.pause_time_max = mob_config.get('pause_time_max', self.pause_time_max)
        
        if 'oam' in config:
            oam_config = config['oam']
            self.min_mode = oam_config.get('min_mode', self.min_mode)
            self.max_mode = oam_config.get('max_mode', self.max_mode)
        
        # Set default OAM parameters if not provided in config
        if self.min_mode is None:
            self.min_mode = 1
        if self.max_mode is None:
            self.max_mode = 8
        
        # Load training parameters
        if 'training' in config:
            training_config = config['training']
            # Ensure max_steps is initialized before using it as default
            if hasattr(self, 'max_steps'):
                self.max_steps = training_config.get('max_steps_per_episode', self.max_steps)
            else:
                self.max_steps = training_config.get('max_steps_per_episode', 1000)
        
        # Optional: handover optimization parameters
        if 'handover_optimization' in config:
            ho = config['handover_optimization']
            self.min_steps_between_switches = int(ho.get('min_handover_interval', self.min_steps_between_switches))
            self.time_to_trigger_steps = int(ho.get('time_to_trigger_steps', self.time_to_trigger_steps))
            self.sinr_hysteresis_db = float(ho.get('handover_hysteresis_db', self.sinr_hysteresis_db))

        # Reward parameters are now handled by the reward calculator
        # No need to update them here since the reward calculator is initialized with the config
    
    def _validate_environment_parameters(self):
        """Validate environment parameters are within reasonable ranges."""
        
        # Distance validation
        if not (1.0 <= self.distance_min < self.distance_max <= 100000):
            raise ValueError(f"Distance range [{self.distance_min}, {self.distance_max}] is invalid")
        
        # Velocity validation
        if not (0.1 <= self.velocity_min < self.velocity_max <= 100):
            raise ValueError(f"Velocity range [{self.velocity_min}, {self.velocity_max}] m/s is invalid")
        
        # Area validation
        if not all(10.0 <= size <= 100000 for size in self.area_size):
            raise ValueError(f"Area size {self.area_size} is outside reasonable range")
        
        # Time step validation
        if not (0.01 <= self.dt <= 10.0):
            raise ValueError(f"Time step {self.dt} s is outside reasonable range (0.01-10.0 s)")
        
        # Reward parameters are validated by the reward calculator
        # No need to validate them here
        
        print("✅ Environment parameters validated successfully")
    
    def _validate_separated_components(self):
        """Validate separated components."""
        # Validate mobility model
        mob_valid, mob_error = self.mobility_model.validate_mobility_parameters()
        if not mob_valid:
            raise ValueError(f"Mobility model validation failed: {mob_error}")
        
        # Validate reward calculator
        reward_valid, reward_error = self.reward_calculator.validate_reward_parameters()
        if not reward_valid:
            raise ValueError(f"Reward calculator validation failed: {reward_error}")
        
        # Validate physics calculator
        physics_valid, physics_error = self.physics_calculator.validate_physics_parameters(
            sinr_dB=0.0, throughput=0.0
        )
        if not physics_valid:
            raise ValueError(f"Physics calculator validation failed: {physics_error}")
        
        print("✅ Separated components validated successfully")

    def _generate_random_position(self) -> np.ndarray:
        """
        Generate a random position using the mobility model.
        
        Returns:
            3D position array [x, y, z]
        """
        return self.mobility_model.generate_random_position()
    
    def _generate_random_velocity(self) -> np.ndarray:
        """
        Generate a random velocity using the mobility model.
        
        Returns:
            3D velocity vector [vx, vy, vz]
        """
        return self.mobility_model.generate_random_velocity()
    
    def _update_position(self) -> None:
        """
        Update user position using the mobility model.
        """
        self.mobility_model.update_position()
        self.position = self.mobility_model.get_position()
        self.velocity = self.mobility_model.get_velocity()
    
    @safe_calculation("throughput_calculation", fallback_value=0.0)
    def _calculate_throughput(self, sinr_dB: float) -> float:
        """
        Calculate throughput using the physics calculator with caching.
        
        Args:
            sinr_dB: Signal-to-Interference-plus-Noise Ratio in dB
            
        Returns:
            Throughput in bits per second
        """
        # OPTIMIZED: Check if SINR hasn't changed (common in training)
        if self._last_sinr == sinr_dB:
            return self._last_throughput
        
        # OPTIMIZED: Check environment cache first
        sinr_rounded = round(sinr_dB, 1)
        if sinr_rounded in self._throughput_cache:
            self._last_sinr = sinr_dB
            self._last_throughput = self._throughput_cache[sinr_rounded]
            return self._last_throughput
        
        # Calculate throughput using physics calculator (which has its own caching)
        throughput = self.physics_calculator.calculate_throughput(sinr_dB)
        
        # OPTIMIZED: Cache the result
        self._throughput_cache[sinr_rounded] = throughput
        self._last_sinr = sinr_dB
        self._last_throughput = throughput
        
        return throughput
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed (Optional[int]): Random seed for reproducible initialization. Default: None.
            options (Optional[Dict[str, Any]]): Additional reset options. Default: None.
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (initial_state, info_dict)
                - initial_state: 8-dimensional state vector [SINR, distance, vx, vy, vz, current_mode, min_mode, max_mode]
                - info_dict: Additional information including position, velocity, throughput, handovers
                
        Example:
            >>> obs, info = env.reset(seed=42)
            >>> print(f"Initial SINR: {obs[0]:.2f} dB")
            >>> print(f"Initial position: {info['position']}")
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.steps = 0
        self.episode_throughput = 0.0
        self.episode_handovers = 0
        self.last_switch_step = -9999
        self._pending_switch_dir = None
        self._pending_switch_counter = 0
        self.stickiness_counter = 0
        self.last_mode_seen = None
        
        # Reset separated components
        self.mobility_model.reset()
        self.reward_calculator.reset_episode()
        
        # Initialize user position and velocity
        self.position = self._generate_random_position()
        self.velocity = self._generate_random_velocity()
        
        # Initialize OAM mode (start in the middle of the range)
        self.current_mode = (self.min_mode + self.max_mode) // 2
        
        # Run simulator to get initial channel state
        _, self.current_sinr = self.simulator.run_step(self.position, self.current_mode)
        
        # Construct the state vector
        state = np.array([
            self.current_sinr,
            np.linalg.norm(self.position),  # distance
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
            self.current_mode,
            self.min_mode,
            self.max_mode
        ], dtype=np.float32)
        
        info = {
            'position': self.position,
            'velocity': self.velocity,
            'throughput': self._calculate_throughput(self.current_sinr),
            'handovers': 0,
            'episode_handovers': self.episode_handovers,
            'episode_throughput': self.episode_throughput,
            'episode_reward': 0.0,
            'episode_steps': self.steps
        }
        
        return state, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take (0: STAY, 1: UP, 2: DOWN)
                - 0: Keep current OAM mode
                - 1: Increase OAM mode (if possible)
                - 2: Decrease OAM mode (if possible)
            
        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: (next_state, reward, terminated, truncated, info)
                - next_state: Updated state vector
                - reward: Reward value for the action
                - terminated: True if episode ended naturally
                - truncated: True if episode was artificially terminated
                - info: Additional information including SINR, mode, throughput, handovers
                
        Example:
            >>> action = env.action_space.sample()  # Random action
            >>> next_obs, reward, terminated, truncated, info = env.step(action)
            >>> print(f"Reward: {reward:.3f}")
            >>> print(f"Current mode: {info['mode']}")
        """
        self.steps += 1
        
        # Track the previous mode for handover detection
        prev_mode = self.current_mode
        
        # Update OAM mode based on action with TTT/hysteresis gating
        ignored_switch = False
        executed_switch = False
        requested_dir: Optional[int] = None
        sinr_ttt_pass = False
        sinr_hyst_pass = False
        candidate_sinr_db = None
        current_sinr_db_pre = None
        if action == 0:  # STAY
            # Reset pending TTT if agent chooses to stay
            self._pending_switch_dir = None
            self._pending_switch_counter = 0
        elif action in (1, 2):  # UP/DOWN
            requested_dir = +1 if action == 1 else -1
            # Enforce minimum steps between executed switches
            if (self.steps - self.last_switch_step) < self.min_steps_between_switches:
                ignored_switch = True
            else:
                # TTT logic: require consecutive identical requests
                if self._pending_switch_dir == requested_dir:
                    self._pending_switch_counter += 1
                else:
                    self._pending_switch_dir = requested_dir
                    self._pending_switch_counter = 1
                # Execute switch only if TTT satisfied
                if self._pending_switch_counter >= self.time_to_trigger_steps:
                    sinr_ttt_pass = True
                    # SINR-based hysteresis gating: probe candidate vs current at current position
                    try:
                        # Measure current SINR at current mode (pre-mobility update)
                        _, current_measure_db = self.simulator.run_step(self.position, self.current_mode)
                        current_sinr_db_pre = current_measure_db
                        # Determine candidate mode
                        candidate_mode = self.current_mode + (1 if requested_dir == +1 else -1)
                        candidate_mode = max(self.min_mode, min(self.max_mode, candidate_mode))
                        # Probe candidate SINR at same position
                        _, cand_db = self.simulator.run_step(self.position, candidate_mode)
                        candidate_sinr_db = cand_db
                        # Hysteresis check in dB
                        if (candidate_sinr_db - current_measure_db) >= self.sinr_hysteresis_db:
                            sinr_hyst_pass = True
                        else:
                            sinr_hyst_pass = False
                    except Exception:
                        # If probing fails, default to not passing hysteresis
                        sinr_hyst_pass = False
                    # Switch only if hysteresis passes
                    if sinr_hyst_pass:
                        if requested_dir == +1:
                            self.current_mode = min(self.current_mode + 1, self.max_mode)
                        else:
                            self.current_mode = max(self.current_mode - 1, self.min_mode)
                        self._pending_switch_dir = None
                        self._pending_switch_counter = 0
                        executed_switch = True
                        self.last_switch_step = self.steps
                    else:
                        ignored_switch = True
        
        # Detect if a handover occurred with hysteresis
        handover_occurred = (prev_mode != self.current_mode)
        if handover_occurred:
            # Enforce minimum steps between switches
            if (self.steps - self.last_switch_step) < self.min_steps_between_switches:
                # Penalize via extra handover flag; RewardCalculator will charge once per step
                # We tag this in info later via handovers count increase
                pass
            else:
                self.last_switch_step = self.steps
            self.episode_handovers += 1
        
        # Update user position using mobility model
        self._update_position()
        
        # Run simulator to get new channel state
        _, self.current_sinr = self.simulator.run_step(self.position, self.current_mode)
        
        # Calculate throughput using physics calculator
        throughput = self._calculate_throughput(self.current_sinr)
        self.episode_throughput += float(throughput)
        
        # Calculate reward using reward calculator
        reward = self.reward_calculator.calculate_reward(throughput, self.current_sinr, handover_occurred)
        
        # Stickiness bonus for maintaining same mode when stable
        if self.last_mode_seen is None or self.last_mode_seen == self.current_mode:
            self.stickiness_counter += 1
        else:
            self.stickiness_counter = 0
        self.last_mode_seen = self.current_mode
        
        if self.stickiness_counter >= self.stickiness_window and self.current_sinr > 5.0:
            # Small positive bonus for stability under decent SINR
            reward += 0.2
        
        # Construct the next state vector
        next_state = np.array([
            self.current_sinr,
            np.linalg.norm(self.position),  # distance
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
            self.current_mode,
            self.min_mode,
            self.max_mode
        ], dtype=np.float32)
        
        # Check if episode is done
        done = False
        truncated = (self.steps >= self.max_steps)
        
        # Prepare info dictionary
        episode_stats = self.reward_calculator.get_episode_stats()
        info = {
            'position': self.position,
            'velocity': self.velocity,
            'throughput': throughput,
            'handovers': episode_stats['episode_handovers'],
            'min_steps_between_switches': self.min_steps_between_switches,
            'time_to_trigger_steps': self.time_to_trigger_steps,
            'pending_switch_counter': self._pending_switch_counter,
            'pending_switch_dir': self._pending_switch_dir,
            'ignored_switch': ignored_switch,
            'executed_switch': executed_switch,
            'sinr_ttt_pass': sinr_ttt_pass,
            'sinr_hysteresis_pass': sinr_hyst_pass,
            'candidate_sinr_db': candidate_sinr_db,
            'current_sinr_db_pre': current_sinr_db_pre,
            'episode_handovers': self.episode_handovers,
            'episode_throughput': self.episode_throughput,
            'episode_reward': float(episode_stats.get('episode_reward', 0.0)),
            'episode_steps': self.steps,
            'sinr': self.current_sinr,
            'mode': self.current_mode
        }
        
        return next_state, reward, done, truncated, info
    
    def render(self, mode='human'):
        """Render the environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
    
    def clear_throughput_cache(self):
        """Clear the throughput cache to free memory."""
        self._throughput_cache.clear()
        self._last_sinr = None
        self._last_throughput = None
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring performance."""
        physics_stats = self.physics_calculator.get_cache_stats()
        return {
            'environment_cache_size': len(self._throughput_cache),
            'last_sinr': self._last_sinr,
            'last_throughput': self._last_throughput,
            'physics_calculator_stats': physics_stats
        } 