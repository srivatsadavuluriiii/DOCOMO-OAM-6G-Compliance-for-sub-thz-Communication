#!/usr/bin/env python3
"""
Advanced Physics Models for 6G THz Communications
Implements missing critical physics effects for ultra-high frequency communications
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.constants import c as speed_of_light, k as boltzmann_constant
import warnings

@dataclass
class DopplerParameters:
    """Doppler effect parameters for mobile communications"""
    velocity_kmh: float  # User velocity in km/h
    frequency_hz: float  # Carrier frequency in Hz
    angle_degrees: float = 0.0  # Angle between motion and LoS (0° = direct approach)
    acceleration_ms2: float = 0.0  # Acceleration in m/s²

@dataclass
class PhaseNoiseParameters:
    """Phase noise characteristics for THz oscillators"""
    frequency_hz: float  # Carrier frequency
    reference_offset_hz: float = 1e6  # Reference offset (1 MHz)
    phase_noise_dbchz: float = -90  # Phase noise at reference offset
    flicker_corner_hz: float = 1e4  # 1/f noise corner frequency
    oscillator_type: str = "crystal"  # "crystal", "cavity", "pll"

@dataclass 
class AntennaParameters:
    """Antenna array parameters for mutual coupling analysis"""
    num_elements: int  # Number of antenna elements
    element_spacing_wavelengths: float  # Spacing in wavelengths
    array_type: str = "linear"  # "linear", "planar", "conformal"
    frequency_hz: float = 300e9  # Operating frequency
    substrate_permittivity: float = 3.5  # Substrate relative permittivity

class AdvancedPhysicsModels:
    """
    Advanced physics models for 6G THz communications
    
    Implements critical missing physics effects:
    1. Doppler frequency shift for extreme mobility (500 km/h)
    2. Phase noise models for THz oscillators
    3. Antenna mutual coupling at high frequencies
    4. Beam squint effects in wideband THz systems
    5. Non-linear propagation effects at high power
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced physics models"""
        self.config = config or {}
        
        # Physical constants
        self.c = speed_of_light
        self.kb = boltzmann_constant
        
        # Cache for expensive calculations
        self._doppler_cache = {}
        self._coupling_cache = {}
        
    def calculate_doppler_shift(self, params: DopplerParameters) -> Dict[str, float]:
        """
        Calculate Doppler frequency shift for high-speed mobility
        
        Critical for 500 km/h DOCOMO target at THz frequencies
        
        Args:
            params: Doppler parameters including velocity and frequency
            
        Returns:
            Dictionary with Doppler analysis
        """
        try:
            # Convert velocity to m/s
            velocity_ms = params.velocity_kmh / 3.6
            
            # Doppler shift: f_d = f_c * (v/c) * cos(θ)
            angle_rad = math.radians(params.angle_degrees)
            doppler_shift_hz = params.frequency_hz * (velocity_ms / self.c) * math.cos(angle_rad)
            
            # Relative frequency shift
            relative_shift = doppler_shift_hz / params.frequency_hz
            
            # Maximum Doppler spread (for isotropic scattering)
            max_doppler_hz = params.frequency_hz * velocity_ms / self.c
            
            # Doppler bandwidth (twice maximum for bidirectional)
            doppler_bandwidth_hz = 2 * max_doppler_hz
            
            # Coherence time (inversely related to Doppler spread)
            coherence_time_ms = 0.423 / max_doppler_hz * 1000 if max_doppler_hz > 0 else float('inf')
            
            # Acceleration-induced chirp rate (df/dt)
            if params.acceleration_ms2 != 0:
                chirp_rate_hz_s = params.frequency_hz * params.acceleration_ms2 / self.c * math.cos(angle_rad)
            else:
                chirp_rate_hz_s = 0.0
            
            # Critical for THz: phase noise due to Doppler
            phase_noise_variance = (2 * math.pi * doppler_shift_hz)**2 * (coherence_time_ms / 1000)**2
            
            return {
                'doppler_shift_hz': doppler_shift_hz,
                'relative_shift': relative_shift,
                'max_doppler_hz': max_doppler_hz,
                'doppler_bandwidth_hz': doppler_bandwidth_hz,
                'coherence_time_ms': coherence_time_ms,
                'chirp_rate_hz_s': chirp_rate_hz_s,
                'phase_noise_variance': phase_noise_variance,
                'doppler_severity': self._classify_doppler_severity(max_doppler_hz, params.frequency_hz)
            }
            
        except (ValueError, TypeError, ZeroDivisionError):
            return {
                'doppler_shift_hz': 0.0,
                'relative_shift': 0.0,
                'max_doppler_hz': 0.0,
                'doppler_bandwidth_hz': 0.0,
                'coherence_time_ms': float('inf'),
                'chirp_rate_hz_s': 0.0,
                'phase_noise_variance': 0.0,
                'doppler_severity': 'negligible'
            }
    
    def calculate_phase_noise_impact(self, params: PhaseNoiseParameters, 
                                   symbol_rate_hz: float) -> Dict[str, float]:
        """
        Calculate phase noise impact for THz oscillators
        
        Critical for THz communications where phase noise dominates
        
        Args:
            params: Phase noise parameters
            symbol_rate_hz: Symbol rate for analysis
            
        Returns:
            Dictionary with phase noise analysis
        """
        try:
            # Phase noise spectral density at different offsets
            reference_offset = params.reference_offset_hz
            
            # 1/f³ region (close-in)
            close_in_offset = reference_offset / 100
            close_in_noise = params.phase_noise_dbchz + 30 * math.log10(reference_offset / close_in_offset)
            
            # 1/f² region (flicker)
            flicker_noise = params.phase_noise_dbchz + 20 * math.log10(reference_offset / params.flicker_corner_hz)
            
            # Flat region (thermal)
            thermal_noise = params.phase_noise_dbchz
            
            # Integrated phase noise over symbol bandwidth
            integration_bandwidth = symbol_rate_hz / 2
            
            # Phase noise power (integrated)
            if integration_bandwidth < params.flicker_corner_hz:
                # Dominated by flicker noise
                integrated_noise_db = flicker_noise + 10 * math.log10(integration_bandwidth)
            else:
                # Mixed flicker and thermal
                flicker_power = 10**(flicker_noise/10) * params.flicker_corner_hz
                thermal_power = 10**(thermal_noise/10) * (integration_bandwidth - params.flicker_corner_hz)
                total_power = flicker_power + thermal_power
                integrated_noise_db = 10 * math.log10(total_power)
            
            # RMS phase error (radians)
            phase_error_rms_rad = math.sqrt(10**(integrated_noise_db/10))
            
            # Phase error in degrees
            phase_error_rms_deg = math.degrees(phase_error_rms_rad)
            
            # SNR degradation due to phase noise
            snr_degradation_db = -10 * math.log10(math.exp(-phase_error_rms_rad**2/2))
            
            # Frequency-dependent scaling for THz
            thz_scaling_factor = (params.frequency_hz / 100e9)**1.5  # Empirical scaling
            
            return {
                'phase_noise_dbchz': params.phase_noise_dbchz,
                'integrated_noise_db': integrated_noise_db,
                'phase_error_rms_rad': phase_error_rms_rad,
                'phase_error_rms_deg': phase_error_rms_deg,
                'snr_degradation_db': snr_degradation_db * thz_scaling_factor,
                'close_in_noise_dbchz': close_in_noise,
                'flicker_noise_dbchz': flicker_noise,
                'thermal_noise_dbchz': thermal_noise,
                'thz_scaling_factor': thz_scaling_factor,
                'oscillator_quality': self._assess_oscillator_quality(phase_error_rms_deg)
            }
            
        except (ValueError, TypeError, ZeroDivisionError):
            return {
                'phase_noise_dbchz': -90.0,
                'integrated_noise_db': -80.0,
                'phase_error_rms_rad': 0.1,
                'phase_error_rms_deg': 5.7,
                'snr_degradation_db': 0.5,
                'close_in_noise_dbchz': -60.0,
                'flicker_noise_dbchz': -80.0,
                'thermal_noise_dbchz': -90.0,
                'thz_scaling_factor': 1.0,
                'oscillator_quality': 'acceptable'
            }
    
    def calculate_antenna_mutual_coupling(self, params: AntennaParameters) -> Dict[str, Any]:
        """
        Calculate antenna mutual coupling at high frequencies
        
        Critical for THz antenna arrays where coupling is severe
        
        Args:
            params: Antenna array parameters
            
        Returns:
            Dictionary with coupling analysis
        """
        try:
            # Wavelength
            wavelength = self.c / params.frequency_hz
            
            # Element spacing in meters
            spacing_m = params.element_spacing_wavelengths * wavelength
            
            # Mutual coupling coefficients between adjacent elements
            # Simplified model based on spacing and frequency
            
            if params.element_spacing_wavelengths < 0.25:
                # Very close spacing - strong coupling
                coupling_db = -5 - 20 * math.log10(params.element_spacing_wavelengths / 0.1)
            elif params.element_spacing_wavelengths < 0.5:
                # Close spacing - moderate coupling
                coupling_db = -10 - 15 * math.log10(params.element_spacing_wavelengths / 0.25)
            else:
                # Wide spacing - weak coupling
                coupling_db = -20 - 10 * math.log10(params.element_spacing_wavelengths / 0.5)
            
            # Frequency-dependent effects (higher frequency = stronger coupling)
            frequency_factor = max((params.frequency_hz / 100e9)**0.5, 1.0)
            if frequency_factor > 1.0:
                coupling_db += 3 * math.log10(frequency_factor)  # 3 dB per decade above 100 GHz
            
            # Substrate effects
            substrate_factor = math.sqrt(params.substrate_permittivity)
            coupling_db += 2 * math.log10(substrate_factor)  # Substrate increases coupling
            
            # Coupling matrix (simplified for adjacent elements)
            coupling_matrix = np.eye(params.num_elements, dtype=complex)
            coupling_linear = 10**(coupling_db / 20)
            
            for i in range(params.num_elements - 1):
                coupling_matrix[i, i+1] = coupling_linear
                coupling_matrix[i+1, i] = coupling_linear
            
            # Efficiency reduction due to coupling
            total_coupling_power = (params.num_elements - 1) * coupling_linear**2
            efficiency_reduction_db = 10 * math.log10(1 - total_coupling_power)
            
            # Beam pattern distortion
            pattern_distortion_db = abs(coupling_db) / 4  # Empirical relation
            
            return {
                'coupling_coefficient_db': coupling_db,
                'coupling_matrix': coupling_matrix,
                'efficiency_reduction_db': efficiency_reduction_db,
                'pattern_distortion_db': pattern_distortion_db,
                'frequency_factor': frequency_factor,
                'substrate_factor': substrate_factor,
                'element_spacing_wavelengths': params.element_spacing_wavelengths,
                'coupling_severity': self._classify_coupling_severity(coupling_db)
            }
            
        except (ValueError, TypeError, ZeroDivisionError):
            return {
                'coupling_coefficient_db': -20.0,
                'coupling_matrix': np.eye(params.num_elements, dtype=complex),
                'efficiency_reduction_db': -0.1,
                'pattern_distortion_db': 1.0,
                'frequency_factor': 1.0,
                'substrate_factor': 1.0,
                'element_spacing_wavelengths': 0.5,
                'coupling_severity': 'moderate'
            }
    
    def calculate_beam_squint(self, center_frequency_hz: float, bandwidth_hz: float,
                            scan_angle_deg: float = 0.0) -> Dict[str, float]:
        """
        Calculate beam squint effects in wideband THz systems
        
        Beam squint occurs when wideband signals cause frequency-dependent beam steering
        
        Args:
            center_frequency_hz: Center frequency
            bandwidth_hz: Signal bandwidth
            scan_angle_deg: Beam scan angle from broadside
            
        Returns:
            Dictionary with beam squint analysis
        """
        try:
            # Fractional bandwidth
            fractional_bandwidth = bandwidth_hz / center_frequency_hz
            
            # Beam squint angle (frequency-dependent steering)
            # For a linear array: Δθ ≈ (Δf/f) * tan(θ_0)
            scan_angle_rad = math.radians(scan_angle_deg)
            squint_angle_rad = fractional_bandwidth * math.tan(scan_angle_rad)
            squint_angle_deg = math.degrees(squint_angle_rad)
            
            # Beam broadening due to squint
            # Additional beamwidth increase
            broadening_factor = 1 / math.cos(squint_angle_rad) if abs(squint_angle_rad) < math.pi/2 else 2.0
            
            # Gain loss due to beam squint
            # Sinc function approximation for main lobe
            if abs(squint_angle_rad) > 1e-6:
                normalized_squint = squint_angle_rad / (1.22 / math.sqrt(8))  # Normalized to 3dB beamwidth
                gain_loss_db = 20 * math.log10(abs(math.sin(math.pi * normalized_squint) / (math.pi * normalized_squint)))
            else:
                gain_loss_db = 0.0
            
            # Frequency-dependent pointing error
            pointing_error_deg = squint_angle_deg / 2  # RMS pointing error
            
            # Critical frequency where squint becomes severe (>1° squint)
            critical_bandwidth_hz = center_frequency_hz / math.tan(math.radians(1.0))
            
            return {
                'squint_angle_deg': squint_angle_deg,
                'fractional_bandwidth': fractional_bandwidth,
                'broadening_factor': broadening_factor,
                'gain_loss_db': max(gain_loss_db, -20.0),  # Cap at -20 dB
                'pointing_error_deg': pointing_error_deg,
                'critical_bandwidth_hz': critical_bandwidth_hz,
                'squint_severity': self._classify_squint_severity(abs(squint_angle_deg))
            }
            
        except (ValueError, TypeError, ZeroDivisionError):
            return {
                'squint_angle_deg': 0.0,
                'fractional_bandwidth': 0.01,
                'broadening_factor': 1.0,
                'gain_loss_db': 0.0,
                'pointing_error_deg': 0.0,
                'critical_bandwidth_hz': center_frequency_hz,
                'squint_severity': 'negligible'
            }
    
    def calculate_nonlinear_propagation(self, power_dbm: float, frequency_hz: float,
                                      distance_m: float, humidity_percent: float = 50.0) -> Dict[str, float]:
        """
        Calculate non-linear propagation effects at high power
        
        Includes atmospheric non-linearities and power-dependent effects
        
        Args:
            power_dbm: Transmit power in dBm
            frequency_hz: Frequency in Hz
            distance_m: Propagation distance in meters
            humidity_percent: Atmospheric humidity
            
        Returns:
            Dictionary with non-linear effects analysis
        """
        try:
            # Convert power to Watts
            power_w = 10**((power_dbm - 30) / 10)
            
            # Power density (W/m²) - assuming focused beam
            beam_area_m2 = (self.c / frequency_hz)**2  # Diffraction-limited spot
            power_density_wm2 = power_w / beam_area_m2
            
            # Non-linear refractive index coefficient for air
            # n2 ≈ 4 × 10⁻²³ m²/W (typical for air at STP)
            n2_air = 4e-23  # m²/W
            
            # Humidity correction factor
            humidity_factor = 1 + (humidity_percent / 100) * 0.2  # 20% increase at 100% humidity
            n2_eff = n2_air * humidity_factor
            
            # Non-linear phase shift
            # Δφ = 2π * n2 * I * L / λ
            wavelength = self.c / frequency_hz
            nonlinear_phase_rad = 2 * math.pi * n2_eff * power_density_wm2 * distance_m / wavelength
            
            # Self-focusing critical power
            # P_critical = λ² / (2π * n2)
            critical_power_w = wavelength**2 / (2 * math.pi * n2_eff)
            critical_power_dbm = 10 * math.log10(critical_power_w * 1000)
            
            # Beam distortion factor
            if power_w > critical_power_w:
                distortion_factor = power_w / critical_power_w
            else:
                distortion_factor = 1.0
            
            # Harmonic generation efficiency (very weak in air)
            # Second harmonic generation
            harmonic_efficiency = (nonlinear_phase_rad / (2 * math.pi))**2 * 1e-12  # Very small for air
            
            # Stimulated scattering effects (Raman, Brillouin)
            # Threshold powers are very high in air
            raman_threshold_dbm = 60.0  # Typical for air
            brillouin_threshold_dbm = 65.0
            
            return {
                'nonlinear_phase_rad': nonlinear_phase_rad,
                'critical_power_dbm': critical_power_dbm,
                'power_density_wm2': power_density_wm2,
                'distortion_factor': distortion_factor,
                'harmonic_efficiency': harmonic_efficiency,
                'raman_threshold_dbm': raman_threshold_dbm,
                'brillouin_threshold_dbm': brillouin_threshold_dbm,
                'nonlinearity_severity': self._classify_nonlinearity_severity(power_dbm, critical_power_dbm)
            }
            
        except (ValueError, TypeError, ZeroDivisionError):
            return {
                'nonlinear_phase_rad': 0.0,
                'critical_power_dbm': 80.0,
                'power_density_wm2': 1.0,
                'distortion_factor': 1.0,
                'harmonic_efficiency': 1e-15,
                'raman_threshold_dbm': 60.0,
                'brillouin_threshold_dbm': 65.0,
                'nonlinearity_severity': 'negligible'
            }
    
    def _classify_doppler_severity(self, max_doppler_hz: float, frequency_hz: float) -> str:
        """Classify Doppler effect severity"""
        relative_doppler = max_doppler_hz / frequency_hz
        
        if relative_doppler > 1e-6:
            return 'severe'
        elif relative_doppler > 1e-7:
            return 'moderate'
        elif relative_doppler > 1e-8:
            return 'minor'
        else:
            return 'negligible'
    
    def _assess_oscillator_quality(self, phase_error_deg: float) -> str:
        """Assess oscillator quality based on phase error"""
        if phase_error_deg < 1.0:
            return 'excellent'
        elif phase_error_deg < 3.0:
            return 'good'
        elif phase_error_deg < 10.0:
            return 'acceptable'
        else:
            return 'poor'
    
    def _classify_coupling_severity(self, coupling_db: float) -> str:
        """Classify antenna coupling severity"""
        if coupling_db > -10:
            return 'severe'
        elif coupling_db > -20:
            return 'moderate'
        elif coupling_db > -30:
            return 'minor'
        else:
            return 'negligible'
    
    def _classify_squint_severity(self, squint_angle_deg: float) -> str:
        """Classify beam squint severity"""
        if squint_angle_deg > 5.0:
            return 'severe'
        elif squint_angle_deg > 2.0:
            return 'moderate'
        elif squint_angle_deg > 0.5:
            return 'minor'
        else:
            return 'negligible'
    
    def _classify_nonlinearity_severity(self, power_dbm: float, critical_power_dbm: float) -> str:
        """Classify non-linearity severity"""
        power_ratio_db = power_dbm - critical_power_dbm
        
        if power_ratio_db > 10:
            return 'severe'
        elif power_ratio_db > 0:
            return 'moderate'
        elif power_ratio_db > -10:
            return 'minor'
        else:
            return 'negligible'
