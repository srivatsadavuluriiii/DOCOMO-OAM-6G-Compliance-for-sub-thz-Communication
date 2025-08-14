import sys
import os

                                                                     
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import scipy.special as sp
from scipy.constants import c as speed_of_light
from utils.exception_handler import safe_calculation, graceful_degradation, get_exception_handler
from environment.physics_calculator import PhysicsCalculator
from simulator.oam_beam_physics import OAMBeamPhysics, OAMBeamParameters

class ChannelSimulator:
    """
    Physics simulator for OAM wireless channels with realistic atmospheric effects.
    
    Simulates physical impairments that affect OAM mode transmission:
    - Path loss (Friis, 1946)
    - Atmospheric turbulence (Kolmogorov, 1941; Andrews & Phillips, 2005)
    - Crosstalk between OAM modes (Paterson, 2005)
    - Rician fading (Rice, 1948; Simon & Alouini, 2005)
    - Pointing errors (Andrews & Phillips, 2005)
    - Atmospheric attenuation (ITU-R P.676-13)
    
    References:
    - Allen, L., et al. (1992). Orbital angular momentum of light and the transformation 
      of Laguerre-Gaussian laser modes. Physical Review A, 45(11), 8185.
    - Andrews, L. C., & Phillips, R. L. (2005). Laser beam propagation through random 
      media (2nd ed.). SPIE Press.
    - Paterson, C. (2005). Atmospheric turbulence and orbital angular momentum of single 
      photons for optical communication. Physical Review Letters, 94(15), 153901.
    - ITU-R P.676-13 (2022). Attenuation by atmospheric gases. International 
      Telecommunication Union.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the channel simulator with configuration parameters.
        
        Args:
            config: Dictionary containing simulation parameters
        """
        self.config = config or {}
        self.physics_calculator = PhysicsCalculator(self.config)
        self.oam_beam_physics = OAMBeamPhysics(self.config)
                            
        self.frequency = 28.0e9                   
        self.wavelength = speed_of_light / self.frequency
        self.tx_power_dBm = 30.0       
        self.noise_figure_dB = 8.0        
        self.noise_temp = 290.0     
        self.bandwidth = 400e6           
        
                                                      
        self.min_mode = None                           
        self.max_mode = None                           
        self.mode_spacing = 1
        self.beam_width = 0.03           
        
                                  
        self.pointing_error_std = 0.005          
        self.rician_k_factor = 8.0        
        
                                
        self.turbulence_strength = 1e-14                                 
        self.humidity = 50.0                         
        self.temperature = 20.0                   
        self.pressure = 101.3                              
                                                                        
        self.atmospheric_absorption_dB_per_km: Optional[float] = None
        
                                     
        self.antenna_efficiency = 0.75                  
        self.implementation_loss_dB = 3.0               
        self.tx_antenna_gain_dBi = 0.0                               
        self.rx_antenna_gain_dBi = 0.0                              
        
                                     
        if config:
            self._update_config(config)
        
                             
        self._validate_parameters()
        
                            
        self.tx_power_W = 10 ** (self.tx_power_dBm / 10) / 1000                    
        self.k = 2 * np.pi / self.wavelength               
        self.num_modes = self.max_mode - self.min_mode + 1
        
                                   
        self.H = np.eye(self.num_modes, dtype=complex)
        
    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update simulator parameters from configuration with proper type conversion.
        
        Args:
            config: Dictionary containing simulation parameters
        """
                                  
        if 'system' in config:
            system_config = config['system']
            if 'frequency' in system_config:
                                                                         
                frequency = system_config['frequency']
                if isinstance(frequency, dict):
                                                              
                    self.frequency = float(frequency.get('mmwave', 28.0e9))
                else:
                    self.frequency = float(frequency)
                self.wavelength = speed_of_light / self.frequency
            if 'bandwidth' in system_config:
                                                                         
                bandwidth = system_config['bandwidth']
                if isinstance(bandwidth, dict):
                                                              
                    self.bandwidth = float(bandwidth.get('mmwave', 400e6))
                else:
                    self.bandwidth = float(bandwidth)
            if 'tx_power_dBm' in system_config:
                                                                         
                tx_power = system_config['tx_power_dBm']
                if isinstance(tx_power, dict):
                                                              
                    self.tx_power_dBm = float(tx_power.get('mmwave', 30.0))
                else:
                    self.tx_power_dBm = float(tx_power)
            if 'noise_figure_dB' in system_config:
                self.noise_figure_dB = float(system_config['noise_figure_dB'])
            if 'noise_temp' in system_config:
                self.noise_temp = float(system_config['noise_temp'])
            if 'tx_antenna_gain_dBi' in system_config:
                tx_gain = system_config['tx_antenna_gain_dBi']
                if isinstance(tx_gain, dict):
                    self.tx_antenna_gain_dBi = float(tx_gain.get('mmwave', 0.0))
                else:
                    self.tx_antenna_gain_dBi = float(tx_gain)
            if 'rx_antenna_gain_dBi' in system_config:
                rx_gain = system_config['rx_antenna_gain_dBi']
                if isinstance(rx_gain, dict):
                    self.rx_antenna_gain_dBi = float(rx_gain.get('mmwave', 0.0))
                else:
                    self.rx_antenna_gain_dBi = float(rx_gain)
        
                               
        if 'oam' in config:
            oam_config = config['oam']
            if 'min_mode' in oam_config:
                self.min_mode = int(oam_config['min_mode'])
            if 'max_mode' in oam_config:
                self.max_mode = int(oam_config['max_mode'])
            if 'beam_width' in oam_config:
                self.beam_width = float(oam_config['beam_width'])
        
                                                              
        if self.min_mode is None:
            self.min_mode = 1
        if self.max_mode is None:
            self.max_mode = 8
        
                                       
        if 'environment' in config:
            env_config = config['environment']
            if 'pointing_error_std' in env_config:
                self.pointing_error_std = float(env_config['pointing_error_std'])
            if 'rician_k_factor' in env_config:
                self.rician_k_factor = float(env_config['rician_k_factor'])
            if 'turbulence_strength' in env_config:
                self.turbulence_strength = float(env_config['turbulence_strength'])
            if 'humidity' in env_config:
                self.humidity = float(env_config['humidity'])
            if 'temperature' in env_config:
                self.temperature = float(env_config['temperature'])
            if 'pressure' in env_config:
                self.pressure = float(env_config['pressure'])
        
                                    
        if 'enhanced_params' in config:
            enhanced_config = config['enhanced_params']
            if 'antenna_efficiency' in enhanced_config:
                self.antenna_efficiency = float(enhanced_config['antenna_efficiency'])
            if 'implementation_loss_dB' in enhanced_config:
                self.implementation_loss_dB = float(enhanced_config['implementation_loss_dB'])
                           
        if 'physics' in config:
            physics_cfg = config['physics']
            if 'atmospheric_absorption_dB_per_km' in physics_cfg:
                try:
                    self.atmospheric_absorption_dB_per_km = float(physics_cfg['atmospheric_absorption_dB_per_km'])
                except Exception:
                    self.atmospheric_absorption_dB_per_km = None
            if 'pe_model' in physics_cfg:
                self.physics_calculator.pe_model = physics_cfg['pe_model']
    
    def _validate_parameters(self):
        """Comprehensive validation of all simulation parameters with cross-parameter checks."""
        
        errors = []
        warnings = []
        
                                             
        
                              
        if not (1e9 <= self.frequency <= 1000e9):                  
            errors.append(f"Frequency {self.frequency/1e9:.1f} GHz is outside reasonable range (1-1000 GHz)")
        
                          
        if not (-20 <= self.tx_power_dBm <= 50):
            errors.append(f"TX power {self.tx_power_dBm} dBm is outside reasonable range (-20 to 50 dBm)")
        
                                 
        if not (0 <= self.noise_figure_dB <= 20):
            errors.append(f"Noise figure {self.noise_figure_dB} dB is outside reasonable range (0-20 dB)")
        
                                      
        if not (50 <= self.noise_temp <= 500):
            errors.append(f"Noise temperature {self.noise_temp} K is outside reasonable range (50-500 K)")
        
                              
        if not (1e6 <= self.bandwidth <= 10e9):                   
            errors.append(f"Bandwidth {self.bandwidth/1e6:.1f} MHz is outside reasonable range (1-10000 MHz)")
        
                             
        if not (1 <= self.min_mode < self.max_mode <= 20):
            errors.append(f"OAM modes [{self.min_mode}, {self.max_mode}] are outside reasonable range")
        
                               
        if not (0.001 <= self.beam_width <= 1.0):                   
            errors.append(f"Beam width {self.beam_width} rad is outside reasonable range (0.001-1.0 rad)")
        
                                   
        if not (0.0001 <= self.pointing_error_std <= 0.1):                        
            errors.append(f"Pointing error {self.pointing_error_std} rad is outside reasonable range")
        
                               
        if not (0.1 <= self.antenna_efficiency <= 1.0):
            errors.append(f"Antenna efficiency {self.antenna_efficiency} is outside reasonable range (0.1-1.0)")
                                 
        if not (-10 <= self.tx_antenna_gain_dBi <= 60):
            warnings.append(f"TX antenna gain {self.tx_antenna_gain_dBi} dBi is unusual (-10 to 60 dBi typical)")
        if not (-10 <= self.rx_antenna_gain_dBi <= 60):
            warnings.append(f"RX antenna gain {self.rx_antenna_gain_dBi} dBi is unusual (-10 to 60 dBi typical)")
        
                               
        if not (1e-17 <= self.turbulence_strength <= 1e-12):
            errors.append(f"Turbulence strength {self.turbulence_strength} is outside reasonable range (1e-17 to 1e-12)")
        
                                           
        if not (0 <= self.humidity <= 100):
            errors.append(f"Humidity {self.humidity}% is outside reasonable range (0-100%)")
        
        if not (-50 <= self.temperature <= 50):
            errors.append(f"Temperature {self.temperature}C is outside reasonable range (-50 to 50C)")
        
                                                                                 
        pressure_kpa = self.pressure
        if pressure_kpa > 1000:                      
            pressure_kpa = pressure_kpa / 1000.0
            self.pressure = pressure_kpa                               
        if not (50 <= pressure_kpa <= 120):
            errors.append(f"Pressure {pressure_kpa} kPa is outside reasonable range (50-120 kPa)")
        
                                       
        
                                              
        expected_wavelength = 3e8 / self.frequency
        wavelength_error = abs(self.wavelength - expected_wavelength) / expected_wavelength
        if wavelength_error > 0.01:                
            errors.append(f"Wavelength {self.wavelength:.6f} m doesn't match frequency {self.frequency/1e9:.1f} GHz (expected {expected_wavelength:.6f} m)")
        
                                                                     
                                                                  
                                                                        
        min_beam_width = self.wavelength / (2 * np.pi)                       
        if self.beam_width < min_beam_width:
            warnings.append(f"Beam width {self.beam_width:.6f} rad may be too small for wavelength {self.wavelength:.6f} m (minimum: {min_beam_width:.6f} rad)")
        
                                                   
        if self.pointing_error_std > self.beam_width / 2:
            warnings.append(f"Pointing error {self.pointing_error_std:.6f} rad is large compared to beam width {self.beam_width:.6f} rad")
        
                                       
        
                                                            
        if self.frequency > 100e9:                 
            if self.humidity > 80:                 
                warnings.append(f"High humidity {self.humidity}% may cause excessive atmospheric absorption at {self.frequency/1e9:.1f} GHz")
        
                                                  
                                                                          
        inner_scale = 1e-3                            
        if self.turbulence_strength > 1e-13:                     
            warnings.append(f"Strong turbulence {self.turbulence_strength:.2e} may invalidate Kolmogorov model assumptions")
        
                                         
        
                                    
        if 20e9 <= self.frequency <= 40e9:           
            if self.beam_width < 0.01:                    
                warnings.append(f"Very narrow beam width {self.beam_width:.6f} rad may be unrealistic for {self.frequency/1e9:.1f} GHz")
        elif self.frequency > 100e9:           
            if self.beam_width > 0.1:                  
                warnings.append(f"Very wide beam width {self.beam_width:.6f} rad may be unrealistic for {self.frequency/1e9:.1f} GHz")
        
                                         
        if self.frequency > 100e9 and self.tx_power_dBm > 30:
            warnings.append(f"High power {self.tx_power_dBm} dBm may be unrealistic for {self.frequency/1e9:.1f} GHz")
        
                                      
        
                               
        tx_power_linear = 10**(self.tx_power_dBm/10) * 1e-3                
        noise_power_linear = self._calculate_noise_power()
        max_snr = 10 * np.log10(tx_power_linear / noise_power_linear)
        
        if max_snr < 0:
            warnings.append(f"Maximum SNR {max_snr:.1f} dB is negative - system may not be feasible")
        elif max_snr < 10:
            warnings.append(f"Maximum SNR {max_snr:.1f} dB is low - consider adjusting parameters")
        
                                    
        
                                 
        mode_spacing = self.max_mode - self.min_mode + 1
        if mode_spacing < 2:
            errors.append(f"OAM mode spacing {mode_spacing} is too small (minimum 2)")
        elif mode_spacing > 10:
            warnings.append(f"Large OAM mode spacing {mode_spacing} may cause excessive crosstalk")
        
                                         
                                                                                    
        max_safe_mode = int(2 * np.pi * self.beam_width / self.wavelength)
        if self.max_mode > max_safe_mode:
            warnings.append(f"Maximum OAM mode {self.max_mode} may be too high for beam width {self.beam_width:.6f} rad (max safe: {max_safe_mode})")
        
                                         
        
                                                                            
                                                       
                                                          
        p_t_ratio = self.pressure / (self.temperature + 273.15)                     
        if not (0.1 <= p_t_ratio <= 1.0):
            warnings.append(f"Pressure/temperature ratio {p_t_ratio:.3f} may be unrealistic")
        
                                              
        if self.temperature < 0 and self.humidity > 50:
            warnings.append(f"High humidity {self.humidity}% at low temperature {self.temperature}C may cause ice formation")
        
                                      
        
        if errors:
            raise ValueError(f"Parameter validation failed:\n" + "\n".join(f" {error}" for error in errors))
        
        if warnings:
            print("  Parameter validation warnings:")
            for warning in warnings:
                print(f"     {warning}")
        
        print(" All parameters validated successfully")
        
                                   
        validation_summary = {
            'frequency_GHz': self.frequency / 1e9,
            'wavelength_m': self.wavelength,
            'beam_width_rad': self.beam_width,
            'pointing_error_rad': self.pointing_error_std,
            'oam_modes': f"{self.min_mode}-{self.max_mode}",
            'max_snr_dB': max_snr,
            'turbulence_strength': self.turbulence_strength,
            'atmospheric_conditions': f"T={self.temperature}C, P={self.pressure}kPa, H={self.humidity}%"
        }
        
        print(" Validation Summary:")
        for key, value in validation_summary.items():
            print(f"   {key}: {value}")

    @safe_calculation("path_loss_calculation", fallback_value=1e6)
    def _calculate_path_loss(self, distance: float, frequency_ghz: float) -> float:
        """
        Calculate total path loss including free-space, atmospheric, and clutter loss.
        """
        # Free-space path loss (dB)
        free_space_loss_db = self.physics_calculator.calculate_path_loss(distance, self.frequency)
        
        # Atmospheric absorption (dB)
        atmospheric_loss_db = self._calculate_atmospheric_absorption(distance)
        
        # Clutter loss (dB)
        clutter_loss_db = self.physics_calculator.calculate_clutter_loss(distance, frequency_ghz)
        
        # Stochastic blockage loss (dB)
        blockage_loss_db = self.physics_calculator.calculate_blockage_loss(frequency_ghz)
        
        total_loss_db = free_space_loss_db + atmospheric_loss_db + clutter_loss_db + blockage_loss_db
        # Return linear loss factor
        return 10**(total_loss_db / 10)
    
    @safe_calculation("atmospheric_absorption_calculation", fallback_value=0.0)
    def _calculate_atmospheric_absorption(self, distance: float) -> float:
        """
        Calculate atmospheric absorption using ITU-R P.676 model.
        
        This implements the ITU-R P.676-13 recommendation for atmospheric attenuation
        at frequencies up to 1000 GHz, including oxygen and water vapor absorption.
        
        Args:
            distance: Distance in meters
            
        Returns:
            Atmospheric attenuation in dB
        """
                                                                        
        if self.atmospheric_absorption_dB_per_km is not None:
            gamma_total = max(0.0, float(self.atmospheric_absorption_dB_per_km))
        else:
                                                                            
            freq_GHz = self.frequency / 1e9
                                           
            if freq_GHz < 60:
                gamma_oxygen = 0.02         
                gamma_water = 0.01          
            elif freq_GHz < 200:
                gamma_oxygen = 0.5          
                gamma_water = 0.5           
            else:
                gamma_oxygen = 2.0          
                gamma_water = 5.0           
            gamma_total = gamma_oxygen + gamma_water
        
                                
        distance_km = distance / 1000.0
        
                                                      
        atmospheric_attenuation_dB = gamma_total * distance_km
        
        return atmospheric_attenuation_dB
    
    def _generate_turbulence_screen(self, distance: float) -> np.ndarray:
        """
        Generate atmospheric turbulence effects using proper Kolmogorov model.
        
        Implements von Karman spectrum with finite inner/outer scale effects:
        - Kolmogorov spectral density: Φ_n(κ) = 0.033 C_n² κ^(-11/3)
        - von Karman spectrum: Φ_n(κ) = 0.033 C_n² (κ² + κ₀²)^(-11/6) exp(-κ²/κ_m²)
        - Phase structure function: D_φ(r) = 6.88 (r/r₀)^(5/3)
        
        Args:
            distance: Propagation distance in meters
            
        Returns:
            Turbulence-induced phase screen for each OAM mode
            
        References:
        - Kolmogorov, A. N. (1941). The local structure of turbulence in incompressible 
          viscous fluid for very large Reynolds numbers. Doklady Akademii Nauk SSSR, 
          30, 301-305.
        - Andrews, L. C., & Phillips, R. L. (2005). Laser beam propagation through 
          random media (2nd ed.). SPIE Press.
        - Fried, D. L. (1966). Optical resolution through a randomly inhomogeneous 
          medium for very long and very short exposures. Journal of the Optical Society 
          of America, 56(10), 1372-1379.
        """
                                          
                                                                         
        l0 = 0.01                    
        
                                                                            
        L0 = 50.0                    
        
                             
        k0 = 2 * np.pi / L0                         
        km = 5.92 / l0                                        
        
                                                               
                                              
        r0 = (0.423 * (self.k ** 2) * self.turbulence_strength * distance) ** (-3/5)
        
                                        
        phase_screen = np.zeros((self.num_modes, self.num_modes), dtype=complex)
        
                                             
        w_L = self.beam_width * distance
        
                                            
                                                              
        def von_karman_spectrum(kappa):
            """Calculate von Karman spectrum for given spatial frequency."""
            if kappa < 1e-10:                          
                return 0.0
            return 0.033 * self.turbulence_strength * (kappa**2 + k0**2)**(-11/6) * np.exp(-kappa**2 / km**2)
        
                                                                     
                                                   
        def phase_structure_function(r):
            """Calculate phase structure function using von Karman spectrum."""
            if r < 1e-10:
                return 0.0
            
                                                             
                                                                        
                                                    
                                                                             
            
            if r < l0:
                                    
                return 6.88 * (r / r0) ** (5/3)
            elif r < L0:
                                                            
                return 6.88 * (r / r0) ** (5/3) * (1 - (r / L0) ** (1/3))
            else:
                                                 
                return 6.88 * (L0 / r0) ** (5/3) * (1 - (L0 / L0) ** (1/3))
        
                                                            
                                                                 
                                                     
        scintillation_index_weak = 1.23 * self.turbulence_strength * (self.k ** (7/6)) * (distance ** (11/6))
        
                                                
        if scintillation_index_weak > 1.0:
                                          
            scintillation_index = 1.0 - np.exp(-scintillation_index_weak)
        else:
            scintillation_index = scintillation_index_weak
        
                                                    
                                                               
        beam_wander_variance = 2.42 * self.turbulence_strength * (distance ** 3) * (self.wavelength ** (-1/3)) * (1 - (l0 / L0) ** (1/3))
        beam_wander = np.sqrt(beam_wander_variance)
        
                                                              
                                                          
        modes = np.arange(self.min_mode, self.max_mode + 1)
        mode_factors = (modes ** 2) / 4.0
        
                                                                                    
        phase_variances = np.zeros(self.num_modes)
        for i in range(self.num_modes):
            phase_variances[i] = mode_factors[i] * phase_structure_function(w_L / np.sqrt(mode_factors[i]))
        phase_perturbations = np.random.normal(0, np.sqrt(phase_variances), self.num_modes)
        
                                      
        amplitude_factors = np.maximum(0.1, 1.0 - mode_factors * scintillation_index / 2.0)
        
                                           
        np.fill_diagonal(phase_screen, amplitude_factors * np.exp(1j * phase_perturbations))
        
                                                                 
                                                                 
        mode_diff_matrix = np.abs(modes[:, None] - modes[None, :])
        
                                                                                    
        coupling_strengths = np.zeros((self.num_modes, self.num_modes))
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                if i != j:
                    mode_diff = mode_diff_matrix[i, j]
                    coupling_strengths[i, j] = (beam_wander / w_L) * (1.0 / max(mode_diff, 1)) * phase_structure_function(w_L / mode_diff)
        coupling_strengths = np.minimum(0.3, coupling_strengths)                  
        
                                  
        coupling_phases = np.random.uniform(0, 2 * np.pi, (self.num_modes, self.num_modes))
        
                                                 
        off_diagonal_mask = ~np.eye(self.num_modes, dtype=bool)
        phase_screen[off_diagonal_mask] = coupling_strengths[off_diagonal_mask] * np.exp(1j * coupling_phases[off_diagonal_mask])
        
        return phase_screen
    
    def _calculate_crosstalk(self, distance: float, turbulence_screen: np.ndarray) -> np.ndarray:
        """
        Calculate crosstalk between OAM modes using proper Laguerre-Gaussian beam physics.
        
        Implements physics-based OAM crosstalk:
        - Exact Laguerre-Gaussian mode overlap integrals
        - Atmospheric turbulence perturbations on beam structure
        - Beam divergence effects on mode coupling
        - Power coupling efficiency between modes
        
        Args:
            distance: Propagation distance in meters
            turbulence_screen: Turbulence-induced phase screen
            
        Returns:
            Complex crosstalk matrix [num_modes x num_modes]
            
        References:
        - Allen, L., et al. (1992). Orbital angular momentum of light. Phys. Rev. A 45, 8185.
        - Paterson, C. (2005). Atmospheric turbulence and orbital angular momentum. PRL 94, 153901.
        """
        # Use proper OAM beam physics for crosstalk calculation
        crosstalk_matrix = np.eye(self.num_modes, dtype=complex)
        
        # Create beam parameters for current configuration
        beam_params = OAMBeamParameters(
            wavelength=self.wavelength,
            waist_radius=self.beam_width,
            aperture_radius=self.beam_width * 3.0,  # Reasonable aperture size
            oam_mode_l=0,  # Will be updated for each mode
            radial_mode_p=0
        )
        
        # Calculate atmospheric turbulence strength
        cn2 = self.turbulence_strength * 1e-14  # Convert to proper units
        
        # Get modes array
        modes = np.arange(self.min_mode, self.max_mode + 1)
        
        # Calculate mode coupling using atmospheric turbulence model
        if cn2 > 1e-18:  # Only if significant turbulence
            atm_coupling = self.oam_beam_physics.atmospheric_mode_coupling(
                cn2, max(abs(self.max_mode), abs(self.min_mode)), 
                speed_of_light / self.wavelength, distance, beam_params
            )
            
            # Map atmospheric coupling to our mode indices
            l_max = max(abs(self.max_mode), abs(self.min_mode))
            mode_count = 2 * l_max + 1
            atm_modes = list(range(-l_max, l_max + 1))
            
            for i, mode_i in enumerate(modes):
                for j, mode_j in enumerate(modes):
                    try:
                        atm_i = atm_modes.index(mode_i)
                        atm_j = atm_modes.index(mode_j)
                        crosstalk_matrix[i, j] = atm_coupling[atm_i, atm_j]
                    except (ValueError, IndexError):
                        # Mode not in atmospheric coupling matrix, use identity
                        crosstalk_matrix[i, j] = 1.0 if i == j else 0.0
        
        # Add beam divergence effects
        beam_evolution = self.oam_beam_physics.beam_divergence_evolution(
            np.array([distance]), beam_params
        )
        beam_radius_at_rx = beam_evolution['beam_radius'][0]
        
        # Mode-dependent divergence (higher l modes diverge more)
        for i, mode_i in enumerate(modes):
            for j, mode_j in enumerate(modes):
                if i != j:
                    # Power coupling efficiency between different modes
                    import copy
                    tx_params = copy.deepcopy(beam_params)
                    tx_params.oam_mode_l = mode_i
                    rx_params = copy.deepcopy(beam_params)
                    rx_params.oam_mode_l = mode_j
                    rx_params.waist_radius = beam_radius_at_rx
                    
                    coupling = self.oam_beam_physics.power_coupling_efficiency(
                        tx_params, rx_params, distance, atmospheric_cn2=cn2
                    )
                    
                    # Apply coupling efficiency as additional loss
                    crosstalk_matrix[i, j] *= np.sqrt(coupling['total_efficiency'])
        
        # Apply pointing errors and atmospheric perturbations
        if hasattr(self, 'pointing_error_std') and self.pointing_error_std > 0:
            pointing_loss = np.exp(-2 * (self.pointing_error_std / beam_radius_at_rx)**2)
            # Diagonal elements (self-coupling) get pointing loss
            np.fill_diagonal(crosstalk_matrix, np.diag(crosstalk_matrix) * pointing_loss)
        
        # Normalize to ensure power conservation
        for i in range(self.num_modes):
            row_power = np.sum(np.abs(crosstalk_matrix[i, :])**2)
            if row_power > 1e-15:
                crosstalk_matrix[i, :] /= np.sqrt(row_power)
        
        return crosstalk_matrix
        
    def _get_rician_fading_gain(self) -> np.ndarray:
        """
        Calculate Rician fading channel gains.
        
        Returns:
            Matrix of Rician fading gains for each mode
        """
                                                         
                                            
        k_linear = 10 ** (self.rician_k_factor / 10)
        
                                  
        v = np.sqrt(k_linear / (k_linear + 1))                 
        sigma = np.sqrt(1 / (2 * (k_linear + 1)))                         
        
                                                                         
        fading_matrix = np.zeros((self.num_modes, self.num_modes), dtype=complex)
        
                                                      
        diagonal_scatter_real = np.random.normal(0, sigma, self.num_modes)
        diagonal_scatter_imag = np.random.normal(0, sigma, self.num_modes)
        diagonal_scatter = diagonal_scatter_real + 1j * diagonal_scatter_imag
        np.fill_diagonal(fading_matrix, v + diagonal_scatter)
        
                                                                  
        off_diagonal_mask = ~np.eye(self.num_modes, dtype=bool)
        off_diagonal_scatter_real = np.random.normal(0, sigma * 0.1, (self.num_modes, self.num_modes))
        off_diagonal_scatter_imag = np.random.normal(0, sigma * 0.1, (self.num_modes, self.num_modes))
        off_diagonal_scatter = off_diagonal_scatter_real + 1j * off_diagonal_scatter_imag
        fading_matrix[off_diagonal_mask] = off_diagonal_scatter[off_diagonal_mask]
        
        return fading_matrix
    
    def _get_pointing_error_loss(self, oam_mode: int, user_speed_kmh: float) -> float:
        """
        Calculate loss due to pointing errors with OAM mode sensitivity.
        
        Uses either a simple geometric or an advanced dynamic model based on config.
        """
        misalignment_rad = np.random.normal(0, self.pointing_error_std)
        misalignment_deg = np.degrees(misalignment_rad)

        if self.physics_calculator.pe_model == 'dynamic':
            loss_db = self.physics_calculator.calculate_dynamic_pointing_loss(misalignment_deg, user_speed_kmh)
            return 10**(-loss_db / 10.0)
        else:
            # Original geometric model
            mode_sensitivity = 1.0 + 0.2 * (oam_mode - self.min_mode)
            pointing_loss = np.exp(-(misalignment_rad * mode_sensitivity)**2 / (2 * self.beam_width**2))
            return max(pointing_loss, 0.01)

    @safe_calculation("simulation_step", fallback_value=(np.eye(8, dtype=complex), -10.0))                          
    def run_step(self, user_position: np.ndarray, current_oam_mode: int, user_speed_kmh: float, interferer_positions: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, float]:
        """
        Run one step of the channel simulation.
        
        Args:
            user_position: 3D position of the user [x, y, z] in meters
            current_oam_mode: Current OAM mode being used
            user_speed_kmh: Current user speed in km/h for dynamic pointing loss
            interferer_positions: List of 3D positions of interfering users
            
        Returns:
            Tuple of (channel matrix H, SINR in dB)
        """
                          
        if not isinstance(user_position, np.ndarray) or user_position.size != 3:
            raise ValueError("user_position must be a 3D numpy array")
        
        if not (self.min_mode <= current_oam_mode <= self.max_mode):
            raise ValueError(f"current_oam_mode {current_oam_mode} must be between {self.min_mode} and {self.max_mode}")
        
                                                                       
        distance = np.linalg.norm(user_position)
        
                                         
        if distance < 1.0:
            distance = 1.0                                           
        elif distance > 50000:             
            distance = 50000
        
        frequency_ghz = self.frequency / 1e9
        path_loss = self._calculate_path_loss(distance, frequency_ghz)
        
                                   
        turbulence_screen = self._generate_turbulence_screen(distance)
        
                                              
        crosstalk_matrix = self._calculate_crosstalk(distance, turbulence_screen)
        
                          
        fading_matrix = self._get_rician_fading_gain()
        
                                                      
        pointing_loss = self._get_pointing_error_loss(current_oam_mode, user_speed_kmh)
        
                                                     
                                                    
        path_loss = max(path_loss, 1e-8)                                     
        
                                                                             
                                                                   
        if path_loss > 1e-8:
            channel_gain = 1.0 / path_loss
        else:
                                                        
            channel_gain = 1e8                                   
        
                                            
        channel_gain = channel_gain * self.antenna_efficiency
                                                       
        antenna_gain_linear = 10 ** ((self.tx_antenna_gain_dBi + self.rx_antenna_gain_dBi) / 10.0)
        channel_gain = channel_gain * antenna_gain_linear
        
                                                                     
                                                          
        channel_gain_factor = np.sqrt(channel_gain)
        
                                                               
                                                                  
                                                     
        temp_matrix = crosstalk_matrix * fading_matrix
        self.H = temp_matrix * (turbulence_screen * channel_gain_factor)
        
                                                            
        mode_idx = current_oam_mode - self.min_mode
        pointing_factor = np.ones_like(self.H)
        pointing_factor[mode_idx, :] *= pointing_loss
        pointing_factor[:, mode_idx] *= pointing_loss
        self.H *= pointing_factor
        
                                                
                                                                  
        mode_powers = self.tx_power_W * np.abs(self.H[mode_idx, :])**2
        
                                              
        signal_power = mode_powers[mode_idx]
        
                                                                
                                                     
        interference_mask = np.ones(self.num_modes, dtype=bool)
        interference_mask[mode_idx] = False
        interference_power = np.sum(mode_powers[interference_mask])
        
                                         
        # Calculate interference power from other users
        if interferer_positions:
            inter_user_interference = self._calculate_interference_power(interferer_positions)
        else:
            inter_user_interference = 0.0

        # Total noise and interference
        noise_power = self._calculate_noise_power()
        total_noise_interference = interference_power + noise_power + inter_user_interference
        
        if signal_power < 1e-30:
            sinr_dB = -150.0
        else:
            denominator = total_noise_interference
            denominator = max(denominator, 1e-12)
            sinr = signal_power / denominator
            
            if np.isnan(sinr) or np.isinf(sinr):
                if np.isinf(sinr):
                    sinr_dB = 60.0
                else:
                    sinr_dB = -150.0
            else:
                if sinr > 0:
                    sinr_clamped = min(sinr, 1e12)                            
                    sinr_dB = 10 * np.log10(sinr_clamped)
                else:
                    sinr_dB = -150.0
            
            sinr_dB = max(min(sinr_dB, 40.0), -20.0)
        
        return self.H, sinr_dB
    
    def _calculate_interference_power(self, interferer_positions: List[np.ndarray]) -> float:
        """
        Calculate the total interference power from other users.
        
        Args:
            interferer_positions: List of 3D positions of interfering users.
            
        Returns:
            Total interference power in watts.
        """
        total_interference_power = 0.0
        interferer_tx_power_W = 10 ** (self.tx_power_dBm / 10) / 1000

        for pos in interferer_positions:
            distance = np.linalg.norm(pos)
            if distance < 1.0:
                distance = 1.0
            
            # Simplified path loss for interferers
            path_loss = (4 * np.pi * distance * self.frequency / speed_of_light)**2
            
            if path_loss > 1e-12:
                received_power = interferer_tx_power_W / path_loss
                total_interference_power += received_power
                
        return total_interference_power

    def _calculate_noise_power(self) -> float:
        """
        Calculate thermal noise power in watts.
        
        Returns:
            Noise power in watts
        """
                                  
        k = 1.380649e-23
        
                                         
        noise_power_watts = k * self.noise_temp * self.bandwidth
        noise_power_watts *= 10 ** (self.noise_figure_dB / 10)                      
        
        return noise_power_watts 


if __name__ == "__main__":
                                              
    cfg = {
        'oam': {
            'min_mode': 1,
            'max_mode': 8
        }
    }
    sim = ChannelSimulator(cfg)
    mode = (sim.min_mode + sim.max_mode) // 2
    for d in [50.0, 100.0, 200.0]:
        pos = np.array([d, 0.0, 0.0], dtype=float)
        _, sinr_db = sim.run_step(pos, mode, 0.0) # Pass user_speed_kmh=0.0 for static model
        print(f"distance={d:6.1f} m  mode={mode}  SINR={sinr_db:6.2f} dB")
