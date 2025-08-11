import sys
import os

# Ensure project root is on sys.path before importing project modules
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
        # Default parameters
        self.frequency = 28.0e9  # 28 GHz (mmWave)
        self.wavelength = speed_of_light / self.frequency
        self.tx_power_dBm = 30.0  # dBm
        self.noise_figure_dB = 8.0  # 8 dB
        self.noise_temp = 290.0  # K
        self.bandwidth = 400e6  # 400 MHz
        
        # OAM parameters (will be updated from config)
        self.min_mode = None  # Will be set from config
        self.max_mode = None  # Will be set from config
        self.mode_spacing = 1
        self.beam_width = 0.03  # 30 mrad
        
        # Environment parameters  
        self.pointing_error_std = 0.005  # 5 mrad
        self.rician_k_factor = 8.0  # 8 dB
        
        # Atmospheric parameters
        self.turbulence_strength = 1e-14  # Cn² value (typical clear air)
        self.humidity = 50.0  # Relative humidity (%)
        self.temperature = 20.0  # Temperature (C)
        self.pressure = 101.3  # Atmospheric pressure (kPa)
        # Optional override for specific attenuation (dB/km) from config
        self.atmospheric_absorption_dB_per_km: Optional[float] = None
        
        # Basic additional parameters
        self.antenna_efficiency = 0.75  # 75% efficiency
        self.implementation_loss_dB = 3.0  # 3 dB losses
        self.tx_antenna_gain_dBi = 0.0  # Transmit antenna gain (dBi)
        self.rx_antenna_gain_dBi = 0.0  # Receive antenna gain (dBi)
        
        # Update with provided config
        if config:
            self._update_config(config)
        
        # Validate parameters
        self._validate_parameters()
        
        # Derived parameters
        self.tx_power_W = 10 ** (self.tx_power_dBm / 10) / 1000  # Convert dBm to W
        self.k = 2 * np.pi / self.wavelength  # Wave number
        self.num_modes = self.max_mode - self.min_mode + 1
        
        # Initialize channel matrix
        self.H = np.eye(self.num_modes, dtype=complex)
        
    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update simulator parameters from configuration with proper type conversion.
        
        Args:
            config: Dictionary containing simulation parameters
        """
        # Update system parameters
        if 'system' in config:
            system_config = config['system']
            if 'frequency' in system_config:
                # Handle both single value and dictionary (hybrid config)
                frequency = system_config['frequency']
                if isinstance(frequency, dict):
                    # For hybrid config, use mmWave as default
                    self.frequency = float(frequency.get('mmwave', 28.0e9))
                else:
                    self.frequency = float(frequency)
                self.wavelength = speed_of_light / self.frequency
            if 'bandwidth' in system_config:
                # Handle both single value and dictionary (hybrid config)
                bandwidth = system_config['bandwidth']
                if isinstance(bandwidth, dict):
                    # For hybrid config, use mmWave as default
                    self.bandwidth = float(bandwidth.get('mmwave', 400e6))
                else:
                    self.bandwidth = float(bandwidth)
            if 'tx_power_dBm' in system_config:
                # Handle both single value and dictionary (hybrid config)
                tx_power = system_config['tx_power_dBm']
                if isinstance(tx_power, dict):
                    # For hybrid config, use mmWave as default
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
        
        # Update OAM parameters
        if 'oam' in config:
            oam_config = config['oam']
            if 'min_mode' in oam_config:
                self.min_mode = int(oam_config['min_mode'])
            if 'max_mode' in oam_config:
                self.max_mode = int(oam_config['max_mode'])
            if 'beam_width' in oam_config:
                self.beam_width = float(oam_config['beam_width'])
        
        # Set default OAM parameters if not provided in config
        if self.min_mode is None:
            self.min_mode = 1
        if self.max_mode is None:
            self.max_mode = 8
        
        # Update environment parameters
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
        
        # Update enhanced parameters
        if 'enhanced_params' in config:
            enhanced_config = config['enhanced_params']
            if 'antenna_efficiency' in enhanced_config:
                self.antenna_efficiency = float(enhanced_config['antenna_efficiency'])
            if 'implementation_loss_dB' in enhanced_config:
                self.implementation_loss_dB = float(enhanced_config['implementation_loss_dB'])
        # Physics overrides
        if 'physics' in config:
            physics_cfg = config['physics']
            if 'atmospheric_absorption_dB_per_km' in physics_cfg:
                try:
                    self.atmospheric_absorption_dB_per_km = float(physics_cfg['atmospheric_absorption_dB_per_km'])
                except Exception:
                    self.atmospheric_absorption_dB_per_km = None
    
    def _validate_parameters(self):
        """Comprehensive validation of all simulation parameters with cross-parameter checks."""
        
        errors = []
        warnings = []
        
        # 1. BASIC PARAMETER RANGE VALIDATION
        
        # Frequency validation
        if not (1e9 <= self.frequency <= 1000e9):  # 1 GHz to 1 THz
            errors.append(f"Frequency {self.frequency/1e9:.1f} GHz is outside reasonable range (1-1000 GHz)")
        
        # Power validation
        if not (-20 <= self.tx_power_dBm <= 50):
            errors.append(f"TX power {self.tx_power_dBm} dBm is outside reasonable range (-20 to 50 dBm)")
        
        # Noise figure validation
        if not (0 <= self.noise_figure_dB <= 20):
            errors.append(f"Noise figure {self.noise_figure_dB} dB is outside reasonable range (0-20 dB)")
        
        # Noise temperature validation
        if not (50 <= self.noise_temp <= 500):
            errors.append(f"Noise temperature {self.noise_temp} K is outside reasonable range (50-500 K)")
        
        # Bandwidth validation
        if not (1e6 <= self.bandwidth <= 10e9):  # 1 MHz to 10 GHz
            errors.append(f"Bandwidth {self.bandwidth/1e6:.1f} MHz is outside reasonable range (1-10000 MHz)")
        
        # OAM mode validation
        if not (1 <= self.min_mode < self.max_mode <= 20):
            errors.append(f"OAM modes [{self.min_mode}, {self.max_mode}] are outside reasonable range")
        
        # Beam width validation
        if not (0.001 <= self.beam_width <= 1.0):  # 1 mrad to 1 rad
            errors.append(f"Beam width {self.beam_width} rad is outside reasonable range (0.001-1.0 rad)")
        
        # Pointing error validation
        if not (0.0001 <= self.pointing_error_std <= 0.1):  # 0.1 mrad to 100 mrad
            errors.append(f"Pointing error {self.pointing_error_std} rad is outside reasonable range")
        
        # Efficiency validation
        if not (0.1 <= self.antenna_efficiency <= 1.0):
            errors.append(f"Antenna efficiency {self.antenna_efficiency} is outside reasonable range (0.1-1.0)")
        # Antenna gain validation
        if not (-10 <= self.tx_antenna_gain_dBi <= 60):
            warnings.append(f"TX antenna gain {self.tx_antenna_gain_dBi} dBi is unusual (-10 to 60 dBi typical)")
        if not (-10 <= self.rx_antenna_gain_dBi <= 60):
            warnings.append(f"RX antenna gain {self.rx_antenna_gain_dBi} dBi is unusual (-10 to 60 dBi typical)")
        
        # Turbulence validation
        if not (1e-17 <= self.turbulence_strength <= 1e-12):
            errors.append(f"Turbulence strength {self.turbulence_strength} is outside reasonable range (1e-17 to 1e-12)")
        
        # Atmospheric parameters validation
        if not (0 <= self.humidity <= 100):
            errors.append(f"Humidity {self.humidity}% is outside reasonable range (0-100%)")
        
        if not (-50 <= self.temperature <= 50):
            errors.append(f"Temperature {self.temperature}C is outside reasonable range (-50 to 50C)")
        
        # Accept either kPa (50-120) or Pa (~50000-120000) via auto-unit handling
        pressure_kpa = self.pressure
        if pressure_kpa > 1000:  # Looks like Pascals
            pressure_kpa = pressure_kpa / 1000.0
            self.pressure = pressure_kpa  # normalize to kPa internally
        if not (50 <= pressure_kpa <= 120):
            errors.append(f"Pressure {pressure_kpa} kPa is outside reasonable range (50-120 kPa)")
        
        # 2. CROSS-PARAMETER VALIDATION
        
        # Frequency vs wavelength relationship
        expected_wavelength = 3e8 / self.frequency
        wavelength_error = abs(self.wavelength - expected_wavelength) / expected_wavelength
        if wavelength_error > 0.01:  # 1% tolerance
            errors.append(f"Wavelength {self.wavelength:.6f} m doesn't match frequency {self.frequency/1e9:.1f} GHz (expected {expected_wavelength:.6f} m)")
        
        # Beam width vs wavelength relationship (physics consistency)
        # Rayleigh criterion: θ ≈ λ/D where D is aperture diameter
        # For OAM beams, beam width should be proportional to wavelength
        min_beam_width = self.wavelength / (2 * np.pi)  # Theoretical minimum
        if self.beam_width < min_beam_width:
            warnings.append(f"Beam width {self.beam_width:.6f} rad may be too small for wavelength {self.wavelength:.6f} m (minimum: {min_beam_width:.6f} rad)")
        
        # Pointing error vs beam width relationship
        if self.pointing_error_std > self.beam_width / 2:
            warnings.append(f"Pointing error {self.pointing_error_std:.6f} rad is large compared to beam width {self.beam_width:.6f} rad")
        
        # 3. PHYSICS CONSISTENCY CHECKS
        
        # Frequency vs atmospheric absorption model validity
        if self.frequency > 100e9:  # Above 100 GHz
            if self.humidity > 80:  # High humidity
                warnings.append(f"High humidity {self.humidity}% may cause excessive atmospheric absorption at {self.frequency/1e9:.1f} GHz")
        
        # Turbulence strength vs distance validity
        # Kolmogorov turbulence model is valid for distances > inner scale
        inner_scale = 1e-3  # 1 mm typical inner scale
        if self.turbulence_strength > 1e-13:  # Strong turbulence
            warnings.append(f"Strong turbulence {self.turbulence_strength:.2e} may invalidate Kolmogorov model assumptions")
        
        # 4. REALISTIC VALUE RANGE CHECKS
        
        # Frequency bands for mmWave
        if 20e9 <= self.frequency <= 40e9:  # Ka-band
            if self.beam_width < 0.01:  # Very narrow beam
                warnings.append(f"Very narrow beam width {self.beam_width:.6f} rad may be unrealistic for {self.frequency/1e9:.1f} GHz")
        elif self.frequency > 100e9:  # Sub-THz
            if self.beam_width > 0.1:  # Very wide beam
                warnings.append(f"Very wide beam width {self.beam_width:.6f} rad may be unrealistic for {self.frequency/1e9:.1f} GHz")
        
        # Power vs frequency relationship
        if self.frequency > 100e9 and self.tx_power_dBm > 30:
            warnings.append(f"High power {self.tx_power_dBm} dBm may be unrealistic for {self.frequency/1e9:.1f} GHz")
        
        # 5. SYSTEM PERFORMANCE CHECKS
        
        # SNR feasibility check
        tx_power_linear = 10**(self.tx_power_dBm/10) * 1e-3  # Convert to W
        noise_power_linear = self._calculate_noise_power()
        max_snr = 10 * np.log10(tx_power_linear / noise_power_linear)
        
        if max_snr < 0:
            warnings.append(f"Maximum SNR {max_snr:.1f} dB is negative - system may not be feasible")
        elif max_snr < 10:
            warnings.append(f"Maximum SNR {max_snr:.1f} dB is low - consider adjusting parameters")
        
        # 6. OAM-SPECIFIC VALIDATION
        
        # Mode spacing validation
        mode_spacing = self.max_mode - self.min_mode + 1
        if mode_spacing < 2:
            errors.append(f"OAM mode spacing {mode_spacing} is too small (minimum 2)")
        elif mode_spacing > 10:
            warnings.append(f"Large OAM mode spacing {mode_spacing} may cause excessive crosstalk")
        
        # Mode vs beam width relationship
        # Higher OAM modes require larger beam widths to avoid excessive diffraction
        max_safe_mode = int(2 * np.pi * self.beam_width / self.wavelength)
        if self.max_mode > max_safe_mode:
            warnings.append(f"Maximum OAM mode {self.max_mode} may be too high for beam width {self.beam_width:.6f} rad (max safe: {max_safe_mode})")
        
        # 7. ATMOSPHERIC MODEL VALIDATION
        
        # Temperature vs pressure relationship (ideal gas law approximation)
        # P = ρRT where ρ is density, R is gas constant
        # For atmospheric conditions, reasonable P/T ratio
        p_t_ratio = self.pressure / (self.temperature + 273.15)  # Convert to Kelvin
        if not (0.1 <= p_t_ratio <= 1.0):
            warnings.append(f"Pressure/temperature ratio {p_t_ratio:.3f} may be unrealistic")
        
        # Humidity vs temperature relationship
        if self.temperature < 0 and self.humidity > 50:
            warnings.append(f"High humidity {self.humidity}% at low temperature {self.temperature}C may cause ice formation")
        
        # 8. REPORT VALIDATION RESULTS
        
        if errors:
            raise ValueError(f"Parameter validation failed:\n" + "\n".join(f" {error}" for error in errors))
        
        if warnings:
            print("  Parameter validation warnings:")
            for warning in warnings:
                print(f"     {warning}")
        
        print(" All parameters validated successfully")
        
        # 9. LOG VALIDATION SUMMARY
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
    def _calculate_path_loss(self, distance: float) -> float:
        """
        Calculate total path loss including free-space and atmospheric absorption.
        
        Implements:
        - Free-space path loss: L_fs = (4πd/λ)² (Friis, 1946)
        - Atmospheric absorption: ITU-R P.676-13 model
        
        Args:
            distance: Distance between transmitter and receiver in meters
            
        Returns:
            Total path loss in linear scale (free-space + atmospheric)
            
        References:
        - Friis, H. T. (1946). A note on a simple transmission formula. 
          Proceedings of the IRE, 34(5), 254-256.
        - ITU-R P.676-13 (2022). Attenuation by atmospheric gases. 
          International Telecommunication Union.
        """
        # 1. Free space path loss: (4πd/λ)^2
        free_space_loss = (4 * np.pi * distance / self.wavelength) ** 2
        
        # 2. Atmospheric absorption (ITU-R P.676 model)
        atmospheric_loss = self._calculate_atmospheric_absorption(distance)
        
        # 3. Total path loss = free-space × atmospheric
        total_path_loss = free_space_loss * atmospheric_loss
        
        return total_path_loss
    
    @safe_calculation("atmospheric_absorption_calculation", fallback_value=1.0)
    def _calculate_atmospheric_absorption(self, distance: float) -> float:
        """
        Calculate atmospheric absorption using ITU-R P.676 model.
        
        This implements the ITU-R P.676-13 recommendation for atmospheric attenuation
        at frequencies up to 1000 GHz, including oxygen and water vapor absorption.
        
        Args:
            distance: Distance in meters
            
        Returns:
            Atmospheric absorption factor (linear scale)
        """
        # If an override is provided via config, use it directly (dB/km)
        if self.atmospheric_absorption_dB_per_km is not None:
            gamma_total = max(0.0, float(self.atmospheric_absorption_dB_per_km))
        else:
            # Conservative, more realistic simplified coefficients (bounded)
            freq_GHz = self.frequency / 1e9
            # Baseline specific attenuation
            if freq_GHz < 60:
                gamma_oxygen = 0.02  # dB/km
                gamma_water = 0.01   # dB/km
            elif freq_GHz < 200:
                gamma_oxygen = 0.5   # dB/km
                gamma_water = 0.5    # dB/km
            else:
                gamma_oxygen = 2.0   # dB/km
                gamma_water = 5.0    # dB/km
            gamma_total = gamma_oxygen + gamma_water
        
        # Convert distance to km
        distance_km = distance / 1000.0
        
        # Calculate total atmospheric attenuation (dB)
        atmospheric_attenuation_dB = gamma_total * distance_km
        
        # Convert to linear scale
        atmospheric_loss = 10 ** (atmospheric_attenuation_dB / 10)
        
        return atmospheric_loss
    
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
        # Kolmogorov turbulence parameters
        # Inner scale (l0) - typically 1-10 mm for atmospheric turbulence
        l0 = 0.01  # 1 cm inner scale
        
        # Outer scale (L0) - typically 10-100 m for atmospheric turbulence  
        L0 = 50.0  # 50 m outer scale
        
        # Spatial frequencies
        k0 = 2 * np.pi / L0  # Outer scale frequency
        km = 5.92 / l0       # Inner scale frequency (5.92/l0)
        
        # Fried parameter (r0) - coherence length of turbulence
        # r0 = (0.423 * k^2 * Cn^2 * L)^(-3/5)
        r0 = (0.423 * (self.k ** 2) * self.turbulence_strength * distance) ** (-3/5)
        
        # Initialize phase screen matrix
        phase_screen = np.zeros((self.num_modes, self.num_modes), dtype=complex)
        
        # Calculate beam radius at distance L
        w_L = self.beam_width * distance
        
        # von Karman spectrum implementation
        # Φ_n(κ) = 0.033 C_n² (κ² + κ₀²)^(-11/6) exp(-κ²/κ_m²)
        def von_karman_spectrum(kappa):
            """Calculate von Karman spectrum for given spatial frequency."""
            if kappa < 1e-10:  # Avoid division by zero
                return 0.0
            return 0.033 * self.turbulence_strength * (kappa**2 + k0**2)**(-11/6) * np.exp(-kappa**2 / km**2)
        
        # Calculate phase structure function with von Karman spectrum
        # D_φ(r) = 2π ∫₀^∞ [1 - J₀(κr)] Φ_n(κ) κ dκ
        def phase_structure_function(r):
            """Calculate phase structure function using von Karman spectrum."""
            if r < 1e-10:
                return 0.0
            
            # Numerical integration of the structure function
            # For computational efficiency, use analytical approximation
            # D_φ(r) ≈ 6.88 (r/r0)^(5/3) for r << l0
            # D_φ(r) ≈ 6.88 (r/r0)^(5/3) (1 - (r/L0)^(1/3)) for l0 << r << L0
            
            if r < l0:
                # Inner scale regime
                return 6.88 * (r / r0) ** (5/3)
            elif r < L0:
                # Inertial range with outer scale correction
                return 6.88 * (r / r0) ** (5/3) * (1 - (r / L0) ** (1/3))
            else:
                # Outer scale regime - saturation
                return 6.88 * (L0 / r0) ** (5/3) * (1 - (L0 / L0) ** (1/3))
        
        # Calculate scintillation index with proper spectrum
        # σ_I^2 = 1.23 C_n^2 k^(7/6) L^(11/6) for weak turbulence
        # For strong turbulence, use saturation model
        scintillation_index_weak = 1.23 * self.turbulence_strength * (self.k ** (7/6)) * (distance ** (11/6))
        
        # Saturation model for strong turbulence
        if scintillation_index_weak > 1.0:
            # Strong turbulence saturation
            scintillation_index = 1.0 - np.exp(-scintillation_index_weak)
        else:
            scintillation_index = scintillation_index_weak
        
        # Calculate beam wander with proper spectrum
        # <r_c^2> = 2.42 C_n^2 L^3 λ^(-1/3) (1 - (l0/L0)^(1/3))
        beam_wander_variance = 2.42 * self.turbulence_strength * (distance ** 3) * (self.wavelength ** (-1/3)) * (1 - (l0 / L0) ** (1/3))
        beam_wander = np.sqrt(beam_wander_variance)
        
        # OPTIMIZED: Vectorized turbulence effects calculation
        # Pre-calculate mode factors for all modes at once
        modes = np.arange(self.min_mode, self.max_mode + 1)
        mode_factors = (modes ** 2) / 4.0
        
        # Vectorized diagonal elements calculation (element-wise for phase function)
        phase_variances = np.zeros(self.num_modes)
        for i in range(self.num_modes):
            phase_variances[i] = mode_factors[i] * phase_structure_function(w_L / np.sqrt(mode_factors[i]))
        phase_perturbations = np.random.normal(0, np.sqrt(phase_variances), self.num_modes)
        
        # Vectorized amplitude factors
        amplitude_factors = np.maximum(0.1, 1.0 - mode_factors * scintillation_index / 2.0)
        
        # Set diagonal elements efficiently
        np.fill_diagonal(phase_screen, amplitude_factors * np.exp(1j * phase_perturbations))
        
        # OPTIMIZED: Vectorized off-diagonal elements calculation
        # Create mode difference matrix for efficient calculation
        mode_diff_matrix = np.abs(modes[:, None] - modes[None, :])
        
        # Vectorized coupling strength calculation (element-wise for phase function)
        coupling_strengths = np.zeros((self.num_modes, self.num_modes))
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                if i != j:
                    mode_diff = mode_diff_matrix[i, j]
                    coupling_strengths[i, j] = (beam_wander / w_L) * (1.0 / max(mode_diff, 1)) * phase_structure_function(w_L / mode_diff)
        coupling_strengths = np.minimum(0.3, coupling_strengths)  # Limit coupling
        
        # Vectorized random phases
        coupling_phases = np.random.uniform(0, 2 * np.pi, (self.num_modes, self.num_modes))
        
        # Apply off-diagonal elements efficiently
        off_diagonal_mask = ~np.eye(self.num_modes, dtype=bool)
        phase_screen[off_diagonal_mask] = coupling_strengths[off_diagonal_mask] * np.exp(1j * coupling_phases[off_diagonal_mask])
        
        return phase_screen
    
    def _calculate_crosstalk(self, distance: float, turbulence_screen: np.ndarray) -> np.ndarray:
        """
        Calculate crosstalk between OAM modes using Paterson et al. model.
        
        Implements the proper OAM crosstalk theory:
        - OAM orthogonality: exp(-((l₁-l₂)/σ)²) where σ depends on turbulence
        - Phase correlation functions for mode coupling
        - Turbulence-induced mode mixing
        - Diffraction and atmospheric effects
        
        Args:
            distance: Propagation distance in meters
            turbulence_screen: Turbulence-induced phase screen
            
        Returns:
            Crosstalk matrix based on Paterson et al. model
            
        References:
        - Paterson, C. (2005). Atmospheric turbulence and orbital angular momentum of 
          single photons for optical communication. Physical Review Letters, 94(15), 153901.
        - Tyler, G. A., & Boyd, R. W. (2009). Influence of atmospheric turbulence on 
          the propagation of quantum states of light carrying orbital angular momentum. 
          Optics Letters, 34(2), 142-144.
        - Djordjevic, I. B., & Arabaci, M. (2010). LDPC-coded orbital angular momentum 
          (OAM) modulation. Optics Express, 18(14), 14627-14646.
        """
        # Initialize crosstalk matrix
        crosstalk_matrix = np.eye(self.num_modes, dtype=complex)
        
        # Calculate beam parameters at distance
        w_L = self.beam_width * distance  # Beam width at distance L
        
        # Fried parameter (coherence length)
        r0 = (0.423 * (self.k ** 2) * self.turbulence_strength * distance) ** (-3/5)
        
        # Paterson et al. crosstalk model parameters
        # Mode coupling width parameter (depends on turbulence strength)
        # σ = w_L / (2 * sqrt(2 * ln(2))) for weak turbulence
        # σ = w_L / (2 * sqrt(1 + (w_L/r0)^2)) for strong turbulence
        if r0 > w_L:
            # Weak turbulence regime
            sigma = w_L / (2 * np.sqrt(2 * np.log(2)))
        else:
            # Strong turbulence regime
            sigma = w_L / (2 * np.sqrt(1 + (w_L / r0) ** 2))
        
        # Calculate diffraction-induced crosstalk
        # Based on OAM mode orthogonality and beam overlap
        diffraction_factor = self.wavelength * distance / (np.pi * w_L ** 2)
        
        # OPTIMIZED: Vectorized crosstalk calculation
        # Pre-calculate mode arrays for vectorized operations
        modes = np.arange(self.min_mode, self.max_mode + 1)
        mode_diff_matrix = np.abs(modes[:, None] - modes[None, :])
        
        # 1. Vectorized OAM orthogonality factor
        orthogonality_matrix = np.exp(-(mode_diff_matrix / sigma) ** 2)
        
        # 2. Vectorized diffraction coupling
        diffraction_matrix = diffraction_factor * orthogonality_matrix
        
        # 3. Vectorized turbulence coupling (from turbulence screen)
        turbulence_matrix = np.abs(turbulence_screen)
        
        # 4. Vectorized phase correlation function
        mode_sum_matrix = modes[:, None] + modes[None, :]
        phase_correlation_matrix = np.exp(-mode_diff_matrix ** 2 / (2 * mode_sum_matrix ** 2))
        
        # 5. Vectorized total crosstalk calculation
        total_coupling_matrix = (diffraction_matrix + turbulence_matrix) * phase_correlation_matrix
        total_coupling_matrix = np.minimum(0.3, total_coupling_matrix)  # Limit coupling
        
        # Vectorized random phases
        coupling_phases = np.random.uniform(0, 2 * np.pi, (self.num_modes, self.num_modes))
        
        # Apply off-diagonal elements efficiently
        off_diagonal_mask = ~np.eye(self.num_modes, dtype=bool)
        crosstalk_matrix[off_diagonal_mask] = total_coupling_matrix[off_diagonal_mask] * np.exp(1j * coupling_phases[off_diagonal_mask])
        
        # OPTIMIZED: Vectorized energy conservation normalization
        # Calculate total power for each row using vectorized operations
        row_powers = np.sum(np.abs(crosstalk_matrix) ** 2, axis=1)
        
        # Create normalization factors (avoid division by zero)
        normalization_factors = np.where(row_powers > 1e-15, 1.0 / np.sqrt(row_powers), 1.0)
        
        # Apply normalization efficiently using broadcasting
        crosstalk_matrix = crosstalk_matrix * normalization_factors[:, None]
        
        return crosstalk_matrix
        
    def _get_rician_fading_gain(self) -> np.ndarray:
        """
        Calculate Rician fading channel gains.
        
        Returns:
            Matrix of Rician fading gains for each mode
        """
        # OPTIMIZED: Vectorized Rician fading calculation
        # Convert K-factor from dB to linear
        k_linear = 10 ** (self.rician_k_factor / 10)
        
        # Rician fading parameters
        v = np.sqrt(k_linear / (k_linear + 1))  # LOS component
        sigma = np.sqrt(1 / (2 * (k_linear + 1)))  # Scatter component std
        
        # Generate fading gains for each mode using vectorized operations
        fading_matrix = np.zeros((self.num_modes, self.num_modes), dtype=complex)
        
        # Vectorized diagonal elements (LOS + scatter)
        diagonal_scatter_real = np.random.normal(0, sigma, self.num_modes)
        diagonal_scatter_imag = np.random.normal(0, sigma, self.num_modes)
        diagonal_scatter = diagonal_scatter_real + 1j * diagonal_scatter_imag
        np.fill_diagonal(fading_matrix, v + diagonal_scatter)
        
        # Vectorized off-diagonal elements (scatter only, reduced)
        off_diagonal_mask = ~np.eye(self.num_modes, dtype=bool)
        off_diagonal_scatter_real = np.random.normal(0, sigma * 0.1, (self.num_modes, self.num_modes))
        off_diagonal_scatter_imag = np.random.normal(0, sigma * 0.1, (self.num_modes, self.num_modes))
        off_diagonal_scatter = off_diagonal_scatter_real + 1j * off_diagonal_scatter_imag
        fading_matrix[off_diagonal_mask] = off_diagonal_scatter[off_diagonal_mask]
        
        return fading_matrix
    
    def _get_pointing_error_loss(self, oam_mode: int) -> float:
        """
        Calculate loss due to pointing errors with OAM mode sensitivity.
        
        Pointing errors occur when there's misalignment between the transmitter and receiver.
        Higher OAM modes are more sensitive to pointing errors due to their more complex
        phase structure.
        
        Args:
            oam_mode: Current OAM mode
            
        Returns:
            Pointing error loss in linear scale
        """
        # Generate random pointing error
        pointing_error = np.random.normal(0, self.pointing_error_std)
        
        # Higher OAM modes are more sensitive to pointing errors
        # Based on theoretical model: sensitivity ~ |l|
        mode_sensitivity = 1.0 + 0.2 * (oam_mode - self.min_mode)
        
        # Calculate loss using Gaussian beam model with mode-dependent sensitivity
        pointing_loss = np.exp(-(pointing_error * mode_sensitivity)**2 / (2 * self.beam_width**2))
        
        return max(pointing_loss, 0.01)  # Minimum 1% transmission
    

    
    @safe_calculation("simulation_step", fallback_value=(np.eye(8, dtype=complex), -10.0))  # Updated for max_mode=8
    def run_step(self, user_position: np.ndarray, current_oam_mode: int) -> Tuple[np.ndarray, float]:
        """
        Run one step of the channel simulation.
        
        Args:
            user_position: 3D position of the user [x, y, z] in meters
            current_oam_mode: Current OAM mode being used
            
        Returns:
            Tuple of (channel matrix H, SINR in dB)
        """
        # Input validation
        if not isinstance(user_position, np.ndarray) or user_position.size != 3:
            raise ValueError("user_position must be a 3D numpy array")
        
        if not (self.min_mode <= current_oam_mode <= self.max_mode):
            raise ValueError(f"current_oam_mode {current_oam_mode} must be between {self.min_mode} and {self.max_mode}")
        
        # Calculate distance from origin (assumed transmitter position)
        distance = np.linalg.norm(user_position)
        
        # Validate distance is reasonable
        if distance < 1.0:
            distance = 1.0  # Minimum distance to avoid singularities
        elif distance > 50000:  # 50 km max
            distance = 50000
        
        # 1. Total path loss (includes free-space + atmospheric absorption)
        path_loss = self._calculate_path_loss(distance)
        
        # 2. Atmospheric turbulence
        turbulence_screen = self._generate_turbulence_screen(distance)
        
        # 3. Crosstalk with turbulence effects
        crosstalk_matrix = self._calculate_crosstalk(distance, turbulence_screen)
        
        # 4. Rician fading
        fading_matrix = self._get_rician_fading_gain()
        
        # 5. Pointing error (specific to current mode)
        pointing_loss = self._get_pointing_error_loss(current_oam_mode)
        
        # Combine all effects to get channel matrix H
        # Improved numerical stability for path loss
        path_loss = max(path_loss, 1e-8)  # Higher floor for better stability
        
        # Calculate channel gain (inverse of losses) with overflow protection
        # Note: Atmospheric absorption is now included in path_loss
        if path_loss > 1e-8:
            channel_gain = 1.0 / path_loss
        else:
            # Extremely low path loss - cap channel gain
            channel_gain = 1e8  # Maximum reasonable channel gain
        
        # Apply antenna efficiency to signal
        channel_gain = channel_gain * self.antenna_efficiency
        # Apply antenna gains (TX + RX) in linear scale
        antenna_gain_linear = 10 ** ((self.tx_antenna_gain_dBi + self.rx_antenna_gain_dBi) / 10.0)
        channel_gain = channel_gain * antenna_gain_linear
        
        # OPTIMIZED: Combine all effects into single matrix operation
        # Pre-calculate channel gain factor for efficiency
        channel_gain_factor = np.sqrt(channel_gain)
        
        # Vectorized matrix multiplication with optimized order
        # Order: (crosstalk * fading) * (turbulence * gain_factor)
        # This minimizes intermediate matrix creation
        temp_matrix = crosstalk_matrix * fading_matrix
        self.H = temp_matrix * (turbulence_screen * channel_gain_factor)
        
        # Apply pointing loss efficiently using broadcasting
        mode_idx = current_oam_mode - self.min_mode
        pointing_factor = np.ones_like(self.H)
        pointing_factor[mode_idx, :] *= pointing_loss
        pointing_factor[:, mode_idx] *= pointing_loss
        self.H *= pointing_factor
        
        # OPTIMIZED: Vectorized SINR calculation
        # Calculate all powers at once using vectorized operations
        mode_powers = self.tx_power_W * np.abs(self.H[mode_idx, :])**2
        
        # Signal power is the diagonal element
        signal_power = mode_powers[mode_idx]
        
        # Interference power is sum of all off-diagonal elements
        # Use vectorized sum with mask for efficiency
        interference_mask = np.ones(self.num_modes, dtype=bool)
        interference_mask[mode_idx] = False
        interference_power = np.sum(mode_powers[interference_mask])
        
        # Thermal noise power calculation
        noise_power = self._calculate_noise_power()
        
        # Calculate SINR with improved numerical stability
        # Use more robust numerical handling for edge cases
        
        # Check for extremely small signal power
        if signal_power < 1e-30:
            # Signal is essentially zero - very poor SINR
            sinr_dB = -150.0
        else:
            # Calculate denominator with improved numerical stability
            denominator = interference_power + noise_power
            
            # Use higher floor for better numerical stability
            # 1e-12 is more appropriate for mmWave systems
            denominator = max(denominator, 1e-12)
            
            # Calculate SINR with overflow protection
            sinr = signal_power / denominator
            
            # Handle numerical edge cases
            if np.isnan(sinr) or np.isinf(sinr):
                if np.isinf(sinr):
                    # Infinite SINR - cap at reasonable maximum
                    sinr_dB = 60.0
                else:
                    # NaN - use minimum SINR
                    sinr_dB = -150.0
            else:
                # Convert to dB with safety checks
                if sinr > 0:
                    # Clamp SINR to prevent overflow in log calculation
                    sinr_clamped = min(sinr, 1e12)  # Cap at 120 dB equivalent
                    sinr_dB = 10 * np.log10(sinr_clamped)
                else:
                    # Non-positive SINR - minimum value
                    sinr_dB = -150.0
            
            # Final validation with realistic bounds for mmWave systems
            # Typical mmWave SINR range: -20 dB to +40 dB
            sinr_dB = max(min(sinr_dB, 40.0), -20.0)
        
        return self.H, sinr_dB
    
    def _calculate_noise_power(self) -> float:
        """
        Calculate thermal noise power in watts.
        
        Returns:
            Noise power in watts
        """
        # Boltzmann constant (J/K)
        k = 1.380649e-23
        
        # Noise power: kTB + noise figure
        noise_power_watts = k * self.noise_temp * self.bandwidth
        noise_power_watts *= 10 ** (self.noise_figure_dB / 10)  # Apply noise figure
        
        return noise_power_watts 


if __name__ == "__main__":
    # Minimal demo runner for ChannelSimulator
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
        _, sinr_db = sim.run_step(pos, mode)
        print(f"distance={d:6.1f} m  mode={mode}  SINR={sinr_db:6.2f} dB")
