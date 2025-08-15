#!/usr/bin/env python3
"""
Ultra-High Mobility Model for DOCOMO 6G
Supports mobility up to 500 km/h with predictive beam tracking
Advanced Doppler compensation and handover prediction
"""

import numpy as np
import scipy.signal
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import warnings
from enum import Enum

class MobilityProfile(Enum):
    """Mobility profile types"""
    STATIONARY = "stationary"                
    PEDESTRIAN = "pedestrian"                   
    VEHICULAR_SLOW = "vehicular_slow"               
    VEHICULAR_FAST = "vehicular_fast"              
    HIGH_SPEED_RAIL = "high_speed_rail"               
    EXTREME_MOBILITY = "extreme_mobility"               
    AERIAL = "aerial"                                           

@dataclass
class MobilityState:
    """3D mobility state representation"""
    position_x: float = 0.0                  
    position_y: float = 0.0                    
    position_z: float = 0.0                  
    velocity_x: float = 0.0               
    velocity_y: float = 0.0               
    velocity_z: float = 0.0               
    acceleration_x: float = 0.0            
    acceleration_y: float = 0.0            
    acceleration_z: float = 0.0            
    
    @property
    def speed_ms(self) -> float:
        """Current speed in m/s"""
        return np.sqrt(self.velocity_x**2 + self.velocity_y**2 + self.velocity_z**2)
    
    @property
    def speed_kmh(self) -> float:
        """Current speed in km/h"""
        return self.speed_ms * 3.6
    
    @property
    def acceleration_magnitude(self) -> float:
        """Current acceleration magnitude in m/s²"""
        return np.sqrt(self.acceleration_x**2 + self.acceleration_y**2 + self.acceleration_z**2)

@dataclass
class BeamTrackingState:
    """Beam tracking state"""
    azimuth_deg: float = 0.0                             
    elevation_deg: float = 0.0                             
    beam_width_deg: float = 1.0                  
    tracking_error_deg: float = 0.0                          
    prediction_confidence: float = 1.0                              
    last_update_time: float = 0.0                           

class UltraHighMobilityModel:
    """
    Ultra-high mobility model supporting 500 km/h DOCOMO target
    Features predictive tracking, Doppler compensation, and handover optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ultra-high mobility model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        mobility_config = config.get('docomo_6g_system', {}).get('mobility', {})
        
        # General mobility settings
        self.max_speed_kmh = mobility_config.get('max_speed_kmh', 500.0)
        self.max_acceleration_ms2 = mobility_config.get('acceleration_max_ms2', 10.0)
        self.prediction_horizon_ms = mobility_config.get('prediction_horizon_ms', 50.0)
        self.beam_prediction_enabled = mobility_config.get('beam_prediction_enabled', True)
        self.doppler_compensation = mobility_config.get('doppler_compensation', True)

        # Handover thresholds - make handovers much less frequent
        handover_cfg = mobility_config.get('handover', {})
        self.handover_benefit_threshold = handover_cfg.get('benefit_threshold', 0.8)  # Very high threshold
        self.handover_confidence_threshold = handover_cfg.get('confidence_threshold', 0.95)  # Very high confidence needed

        # Path model settings - check both main mobility config and nested path config
        main_mobility = mobility_config
        path_cfg = mobility_config.get('path', {})
        
        # Support both nested and direct mobility config formats
        self.path_model = main_mobility.get('path_type', path_cfg.get('model', 'random_walk'))
        self.max_speed_kmh = main_mobility.get('speed_kmh', self.max_speed_kmh)  # Override with config speed
        self.path_circular_radius = main_mobility.get('radius_m', path_cfg.get('radius_m', 2000))
        
        # Initial position from config
        init_pos = main_mobility.get('initial_position', path_cfg.get('target_position_m', [5000, 5000, 1.5]))
        self.path_target_position = np.array(init_pos)
        self.path_circular_center = np.array(path_cfg.get('center_m', [0, 0, 1.5]))
        self.path_circular_direction = path_cfg.get('direction', 'clockwise')
        self.path_angular_velocity = 0.0 # rad/s, for circular path
        
                             
        self.beam_tracking_accuracy_deg = 0.05                              
        self.handover_prediction_time_ms = 20.0                    
        self.mobility_state_estimation = True                      
        
                        
        # Initialize mobility state with configured position
        self.current_state = MobilityState(
            position_x=self.path_target_position[0],
            position_y=self.path_target_position[1], 
            position_z=self.path_target_position[2]
        )
        self.predicted_states = deque(maxlen=100)                           
        self.mobility_history = deque(maxlen=1000)                    
        
                       
        self.beam_state = BeamTrackingState()
        self.beam_prediction_buffer = deque(maxlen=50)
        
                                            
        self.kalman_filter = self._initialize_kalman_filter()
        
                          
        self.doppler_history = deque(maxlen=100)
        self.doppler_compensation_active = False
        
                             
        self.handover_predictor = HandoverPredictor(self.prediction_horizon_ms)
        
                    
        self.stats = {
            'max_speed_achieved_kmh': 0.0,
            'max_acceleration_achieved_ms2': 0.0,
            'tracking_errors_deg': deque(maxlen=1000),
            'handover_predictions': 0,
            'successful_predictions': 0,
            'doppler_corrections': 0
        }
        
    def _initialize_kalman_filter(self):
        """Initialize Kalman filter for mobility state estimation"""
                                                                  
        dt = 0.001                  
        
                                                               
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt                                           
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt**2                                    
        F[3:6, 6:9] = np.eye(3) * dt                                               
        
                                                            
        H = np.zeros((6, 9))
        H[0:3, 0:3] = np.eye(3)                    
        H[3:6, 3:6] = np.eye(3)                    
        
                                  
        Q = np.eye(9) * 0.1
        Q[6:9, 6:9] *= 10.0                                      
        
                                        
        R = np.eye(6) * 0.01
        R[0:3, 0:3] *= 1.0                                
        R[3:6, 3:6] *= 0.1                                
        
                                  
        P = np.eye(9) * 1.0
        
        return {
            'F': F, 'H': H, 'Q': Q, 'R': R, 'P': P,
            'x': np.zeros(9)                 
        }
    
    def update_mobility_state(self, 
                            position: Tuple[float, float, float],
                            velocity: Optional[Tuple[float, float, float]] = None,
                            timestamp: float = 0.0) -> MobilityState:
        """
        Update mobility state with new measurements
        
        Args:
            position: 3D position (x, y, z) in meters
            velocity: 3D velocity (vx, vy, vz) in m/s (optional)
            timestamp: Measurement timestamp
            
        Returns:
            Updated mobility state
        """
                                    
        if velocity is not None:
            z = np.array([position[0], position[1], position[2],
                         velocity[0], velocity[1], velocity[2]])
        else:
                                                        
            if self.mobility_history:
                dt = max(timestamp - self.mobility_history[-1]['timestamp'], 0.001)
                prev_pos = self.mobility_history[-1]['position']
                est_velocity = [(position[i] - prev_pos[i]) / dt for i in range(3)]
            else:
                est_velocity = [0.0, 0.0, 0.0]
            
            z = np.array([position[0], position[1], position[2],
                         est_velocity[0], est_velocity[1], est_velocity[2]])
        
                              
        self._kalman_predict()
        self._kalman_update(z)
        
                                          
        state = self.kalman_filter['x']
        
                              
        self.current_state = MobilityState(
            position_x=state[0],
            position_y=state[1], 
            position_z=state[2],
            velocity_x=state[3],
            velocity_y=state[4],
            velocity_z=state[5],
            acceleration_x=state[6],
            acceleration_y=state[7],
            acceleration_z=state[8]
        )
        
                           
        self.stats['max_speed_achieved_kmh'] = max(
            self.stats['max_speed_achieved_kmh'],
            self.current_state.speed_kmh
        )
        self.stats['max_acceleration_achieved_ms2'] = max(
            self.stats['max_acceleration_achieved_ms2'],
            self.current_state.acceleration_magnitude
        )
        
                          
        self.mobility_history.append({
            'timestamp': timestamp,
            'position': position,
            'state': self.current_state,
            'speed_kmh': self.current_state.speed_kmh
        })

        # For circular path, initialize angular velocity
        if self.path_model == 'circular':
            speed_ms = self.current_state.speed_ms
            if self.path_circular_radius > 0:
                self.path_angular_velocity = speed_ms / self.path_circular_radius
        
        return self.current_state
    
    def _get_target_acceleration(self, dt: float) -> np.ndarray:
        """Calculates the acceleration needed to follow a predefined path."""
        if self.path_model == 'linear':
            direction_vector = self.path_target_position - np.array([self.current_state.position_x, self.current_state.position_y, self.current_state.position_z])
            distance_to_target = np.linalg.norm(direction_vector)
            
            if distance_to_target < 10.0: # Reached target
                return np.zeros(3)

            desired_velocity = (direction_vector / distance_to_target) * (self.max_speed_kmh / 3.6)
            required_accel = (desired_velocity - np.array([self.current_state.velocity_x, self.current_state.velocity_y, self.current_state.velocity_z])) / dt
            return required_accel

        elif self.path_model == 'circular':
            current_pos_relative = np.array([self.current_state.position_x, self.current_state.position_y, self.current_state.position_z]) - self.path_circular_center
            # Centripetal acceleration: a = v^2 / r, directed towards the center
            centripetal_accel = -current_pos_relative * (self.current_state.speed_ms**2) / (self.path_circular_radius**2)
            
            # Adjust tangential velocity
            current_angle = np.arctan2(current_pos_relative[1], current_pos_relative[0])
            if self.path_circular_direction == 'clockwise':
                new_angle = current_angle - self.path_angular_velocity * dt
            else: # counter_clockwise
                new_angle = current_angle + self.path_angular_velocity * dt
            
            target_pos = self.path_circular_center + self.path_circular_radius * np.array([np.cos(new_angle), np.sin(new_angle), 0])
            desired_velocity = (target_pos - np.array([self.current_state.position_x, self.current_state.position_y, self.current_state.position_z])) / dt
            required_accel = (desired_velocity - np.array([self.current_state.velocity_x, self.current_state.velocity_y, self.current_state.velocity_z])) / dt
            return centripetal_accel + required_accel

        else: # random_walk
            return np.random.normal(0, 0.5, 3)

    def step_mobility(self, dt: float = 0.1):
        """Advances the mobility model by one time step."""
        
        # Get target acceleration based on path model
        target_accel = self._get_target_acceleration(dt)

        # Apply noise and update state (similar to original random walk logic)
        new_accel = target_accel + np.random.normal(0, 0.1, 3) # Less noise for path models
        
        accel_mag = np.linalg.norm(new_accel)
        if accel_mag > self.max_acceleration_ms2:
            new_accel = new_accel / accel_mag * self.max_acceleration_ms2

        new_velocity = np.array([self.current_state.velocity_x, self.current_state.velocity_y, self.current_state.velocity_z]) + new_accel * dt
        
        speed = np.linalg.norm(new_velocity)
        max_speed_ms = self.max_speed_kmh / 3.6
        if speed > max_speed_ms:
            new_velocity = new_velocity / speed * max_speed_ms

        new_position = np.array([self.current_state.position_x, self.current_state.position_y, self.current_state.position_z]) + new_velocity * dt

        self.update_mobility_state(
            position=tuple(new_position),
            velocity=tuple(new_velocity)
        )

    def _kalman_predict(self):
        """Kalman filter prediction step"""
        F = self.kalman_filter['F']
        Q = self.kalman_filter['Q']
        
                       
        self.kalman_filter['x'] = F @ self.kalman_filter['x']
        
                            
        self.kalman_filter['P'] = F @ self.kalman_filter['P'] @ F.T + Q
    
    def _kalman_update(self, measurement: np.ndarray):
        """Kalman filter update step"""
        H = self.kalman_filter['H']
        R = self.kalman_filter['R']
        P = self.kalman_filter['P']
        x = self.kalman_filter['x']
        
                    
        y = measurement - H @ x
        
                               
        S = H @ P @ H.T + R
        
                     
        K = P @ H.T @ np.linalg.inv(S)
        
                                     
        self.kalman_filter['x'] = x + K @ y
        I = np.eye(len(x))
        self.kalman_filter['P'] = (I - K @ H) @ P
    
    def predict_future_position(self, 
                              prediction_time_ms: float) -> Tuple[MobilityState, float]:
        """
        Predict future mobility state
        
        Args:
            prediction_time_ms: Prediction time horizon in milliseconds
            
        Returns:
            Tuple of (predicted_state, confidence)
        """
        if not self.mobility_history:
            return self.current_state, 0.0
        
        dt = prediction_time_ms / 1000.0                      
        
                                          
        current = self.current_state
        
                                                
        predicted_state = MobilityState(
            position_x=current.position_x + current.velocity_x * dt + 0.5 * current.acceleration_x * dt**2,
            position_y=current.position_y + current.velocity_y * dt + 0.5 * current.acceleration_y * dt**2,
            position_z=current.position_z + current.velocity_z * dt + 0.5 * current.acceleration_z * dt**2,
            velocity_x=current.velocity_x + current.acceleration_x * dt,
            velocity_y=current.velocity_y + current.acceleration_y * dt,
            velocity_z=current.velocity_z + current.acceleration_z * dt,
            acceleration_x=current.acceleration_x,                                
            acceleration_y=current.acceleration_y,
            acceleration_z=current.acceleration_z
        )
        
                                                                    
        confidence = self._calculate_prediction_confidence(prediction_time_ms)
        
                          
        self.predicted_states.append({
            'prediction_time_ms': prediction_time_ms,
            'predicted_state': predicted_state,
            'confidence': confidence,
            'timestamp': self.mobility_history[-1]['timestamp'] if self.mobility_history else 0.0
        })
        
        return predicted_state, confidence
    
    def _calculate_prediction_confidence(self, prediction_time_ms: float) -> float:
        """Calculate prediction confidence based on motion patterns"""
        if len(self.mobility_history) < 10:
            return 0.5                                            
        
                                      
        recent_velocities = [h['state'].speed_ms for h in list(self.mobility_history)[-10:]]
        velocity_std = np.std(recent_velocities)
        velocity_mean = np.mean(recent_velocities)
        
        if velocity_mean > 0:
            velocity_consistency = max(0.0, 1.0 - velocity_std / velocity_mean)
        else:
            velocity_consistency = 1.0
        
                                     
        time_decay = max(0.0, 1.0 - prediction_time_ms / 100.0)                    
        
                                                                     
        speed_factor = max(0.1, 1.0 - self.current_state.speed_kmh / 500.0)
        
                             
        confidence = velocity_consistency * time_decay * speed_factor
        
        return min(1.0, max(0.0, confidence))
    
    def calculate_doppler_shift(self, 
                              frequency_ghz: float,
                              base_station_position: Tuple[float, float, float]) -> Dict[str, float]:
        """
        Calculate Doppler shift for current mobility state
        
        Args:
            frequency_ghz: Carrier frequency in GHz
            base_station_position: BS position (x, y, z) in meters
            
        Returns:
            Doppler shift information
        """
        if not self.mobility_history:
            return {'doppler_shift_hz': 0.0, 'doppler_rate_hz_s': 0.0}
        
                                       
        current_pos = np.array([self.current_state.position_x,
                               self.current_state.position_y,
                               self.current_state.position_z])
        current_vel = np.array([self.current_state.velocity_x,
                               self.current_state.velocity_y,
                               self.current_state.velocity_z])
        
        bs_pos = np.array(base_station_position)
        
                                  
        r_vector = current_pos - bs_pos
        r_distance = np.linalg.norm(r_vector)
        
        if r_distance < 1e-6:                          
            return {'doppler_shift_hz': 0.0, 'doppler_rate_hz_s': 0.0}
        
                                    
        r_unit = r_vector / r_distance
        
                                                          
        v_radial = np.dot(current_vel, r_unit)
        
                                   
        c = 3e8                        
        frequency_hz = frequency_ghz * 1e9
        doppler_shift_hz = -(frequency_hz * v_radial) / c                            
        
                                            
        current_acc = np.array([self.current_state.acceleration_x,
                               self.current_state.acceleration_y,
                               self.current_state.acceleration_z])
        a_radial = np.dot(current_acc, r_unit)
        doppler_rate_hz_s = -(frequency_hz * a_radial) / c
        
                               
        doppler_info = {
            'doppler_shift_hz': doppler_shift_hz,
            'doppler_rate_hz_s': doppler_rate_hz_s,
            'radial_velocity_ms': v_radial,
            'radial_acceleration_ms2': a_radial,
            'distance_m': r_distance,
            'frequency_ghz': frequency_ghz
        }
        
        self.doppler_history.append(doppler_info)
        
        return doppler_info
    
    def predict_beam_angles(self,
                          base_station_position: Tuple[float, float, float],
                          prediction_time_ms: float) -> Dict[str, float]:
        """
        Predict required beam angles for future position
        
        Args:
            base_station_position: BS position (x, y, z)
            prediction_time_ms: Prediction horizon in milliseconds
            
        Returns:
            Predicted beam angles and tracking information
        """
                                
        predicted_state, confidence = self.predict_future_position(prediction_time_ms)
        
                                                
        bs_pos = np.array(base_station_position)
        predicted_pos = np.array([predicted_state.position_x,
                                 predicted_state.position_y,
                                 predicted_state.position_z])
        
                                                     
        r_vector = predicted_pos - bs_pos
        r_distance = np.linalg.norm(r_vector)
        
        if r_distance < 1e-6:
            return {
                'azimuth_deg': 0.0,
                'elevation_deg': 0.0,
                'distance_m': 0.0,
                'confidence': 0.0,
                'tracking_error_deg': 0.0
            }
        
                                         
        azimuth_rad = np.arctan2(r_vector[1], r_vector[0])
        elevation_rad = np.arcsin(r_vector[2] / r_distance)
        
                            
        azimuth_deg = np.degrees(azimuth_rad)
        elevation_deg = np.degrees(elevation_rad)
        
                                                                              
        velocity_magnitude = predicted_state.speed_ms
        angular_velocity_rad_s = velocity_magnitude / r_distance               
        
                                 
        prediction_time_s = prediction_time_ms / 1000.0
        angular_uncertainty_rad = angular_velocity_rad_s * prediction_time_s * (1.0 - confidence)
        tracking_error_deg = np.degrees(angular_uncertainty_rad)
        
                           
        self.beam_state.azimuth_deg = azimuth_deg
        self.beam_state.elevation_deg = elevation_deg
        self.beam_state.tracking_error_deg = tracking_error_deg
        self.beam_state.prediction_confidence = confidence
        
                                    
        self.beam_prediction_buffer.append({
            'prediction_time_ms': prediction_time_ms,
            'azimuth_deg': azimuth_deg,
            'elevation_deg': elevation_deg,
            'confidence': confidence,
            'tracking_error_deg': tracking_error_deg
        })
        
        return {
            'azimuth_deg': azimuth_deg,
            'elevation_deg': elevation_deg,  
            'distance_m': r_distance,
            'confidence': confidence,
            'tracking_error_deg': tracking_error_deg,
            'angular_velocity_deg_s': np.degrees(angular_velocity_rad_s),
            'required_beam_width_deg': max(1.0, tracking_error_deg * 2.0)
        }
    
    def should_trigger_handover(self,
                              current_bs_position: Tuple[float, float, float],
                              candidate_bs_positions: List[Tuple[float, float, float]],
                              prediction_time_ms: float = None) -> Dict[str, Any]:
        """
        Determine if handover should be triggered based on mobility prediction
        
        Args:
            current_bs_position: Current base station position
            candidate_bs_positions: List of candidate BS positions
            prediction_time_ms: Prediction horizon (default from config)
            
        Returns:
            Handover decision and analysis
        """
        if prediction_time_ms is None:
            prediction_time_ms = self.handover_prediction_time_ms
        
                                
        predicted_state, confidence = self.predict_future_position(prediction_time_ms)
        predicted_pos = np.array([predicted_state.position_x,
                                 predicted_state.position_y,
                                 predicted_state.position_z])
        
                                                  
        current_bs_pos = np.array(current_bs_position)
        current_distance = np.linalg.norm(predicted_pos - current_bs_pos)
        
        best_candidate = None
        best_distance = current_distance
        
        for i, candidate_pos in enumerate(candidate_bs_positions):
            candidate_pos = np.array(candidate_pos)
            candidate_distance = np.linalg.norm(predicted_pos - candidate_pos)
            
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_candidate = i
        
                                 
        should_handover = False
        handover_benefit = 0.0
        
        if best_candidate is not None:
            handover_benefit = (current_distance - best_distance) / current_distance
            
            if handover_benefit > self.handover_benefit_threshold and confidence > self.handover_confidence_threshold:
                should_handover = True
        
                            
        velocity_factor = min(1.0, self.current_state.speed_kmh / 100.0)                         
        urgency = velocity_factor * handover_benefit * confidence
        
        handover_decision = {
            'should_handover': should_handover,
            'best_candidate_index': best_candidate,
            'handover_benefit': handover_benefit,
            'prediction_confidence': confidence,
            'urgency': urgency,
            'current_distance_m': current_distance,
            'best_candidate_distance_m': best_distance,
            'prediction_time_ms': prediction_time_ms,
            'recommendation': self._get_handover_recommendation(
                should_handover, handover_benefit, confidence, urgency
            )
        }
        
                           
        if should_handover:
            self.stats['handover_predictions'] += 1
        
        return handover_decision
    
    def _get_handover_recommendation(self, 
                                   should_handover: bool,
                                   benefit: float,
                                   confidence: float,
                                   urgency: float) -> str:
        """Get handover recommendation string"""
        if should_handover:
            if urgency > 0.8:
                return "Immediate handover recommended"
            elif urgency > 0.5:
                return "Handover recommended within 10ms"
            else:
                return "Prepare for handover"
        else:
            if benefit > 0.1:
                return "Monitor for handover opportunity" 
            else:
                return "Maintain current connection"
    
    def get_mobility_profile(self) -> MobilityProfile:
        """Classify current mobility profile"""
        speed_kmh = self.current_state.speed_kmh
        
        if speed_kmh < 1.0:
            return MobilityProfile.STATIONARY
        elif speed_kmh <= 10.0:
            return MobilityProfile.PEDESTRIAN
        elif speed_kmh <= 60.0:
            return MobilityProfile.VEHICULAR_SLOW
        elif speed_kmh <= 120.0:
            return MobilityProfile.VEHICULAR_FAST
        elif speed_kmh <= 350.0:
            return MobilityProfile.HIGH_SPEED_RAIL
        elif speed_kmh <= 500.0:
            return MobilityProfile.EXTREME_MOBILITY
        else:
            return MobilityProfile.AERIAL
    
    def get_mobility_statistics(self) -> Dict[str, Any]:
        """Get comprehensive mobility statistics"""
        return {
            'current_speed_kmh': self.current_state.speed_kmh,
            'current_acceleration_ms2': self.current_state.acceleration_magnitude,
            'mobility_profile': self.get_mobility_profile().value,
            'max_speed_achieved_kmh': self.stats['max_speed_achieved_kmh'],
            'max_acceleration_achieved_ms2': self.stats['max_acceleration_achieved_ms2'],
            'tracking_error_avg_deg': np.mean(self.stats['tracking_errors_deg']) if self.stats['tracking_errors_deg'] else 0.0,
            'handover_predictions': self.stats['handover_predictions'],
            'successful_predictions': self.stats['successful_predictions'],
            'prediction_accuracy': (self.stats['successful_predictions'] / 
                                  max(1, self.stats['handover_predictions'])),
            'doppler_corrections': self.stats['doppler_corrections'],
            'history_length': len(self.mobility_history)
        }

class HandoverPredictor:
    """Specialized handover prediction component"""
    
    def __init__(self, prediction_horizon_ms: float):
        self.prediction_horizon_ms = prediction_horizon_ms
        self.handover_history = deque(maxlen=100)
        
    def predict_handover_timing(self, mobility_state: MobilityState, 
                              bs_positions: List[Tuple[float, float, float]]) -> float:
        """Predict optimal handover timing in milliseconds"""
                                                           
        speed_factor = mobility_state.speed_kmh / 100.0             
        
                                                
        optimal_timing_ms = self.prediction_horizon_ms * (1.0 + speed_factor)
        
        return min(optimal_timing_ms, 100.0)                   
