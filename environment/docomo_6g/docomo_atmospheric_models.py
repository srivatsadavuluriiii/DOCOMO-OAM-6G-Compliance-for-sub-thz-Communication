#!/usr/bin/env python3
"""
DOCOMO Enhanced Atmospheric Models
Advanced atmospheric propagation models for 6G sub-THz frequencies
Implements ITU-R P.676-13, molecular absorption, and weather effects
"""

import numpy as np
import scipy.special
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from enum import Enum

class AtmosphericCondition(Enum):
    """Atmospheric condition types"""
    CLEAR = "clear"
    LIGHT_RAIN = "light_rain"  # 1-5 mm/h
    MODERATE_RAIN = "moderate_rain"  # 5-15 mm/h
    HEAVY_RAIN = "heavy_rain"  # 15-50 mm/h
    FOG_LIGHT = "fog_light"  # Visibility 500-1000m
    FOG_DENSE = "fog_dense"  # Visibility < 500m
    SNOW = "snow"
    URBAN_HAZE = "urban_haze"

@dataclass
class AtmosphericParameters:
    """Atmospheric propagation parameters"""
    temperature_c: float = 20.0        # °C
    pressure_hpa: float = 1013.25      # hPa (sea level)
    humidity_percent: float = 50.0     # % relative humidity
    rain_rate_mm_h: float = 0.0       # mm/h
    visibility_m: float = 10000.0     # meters
    wind_speed_ms: float = 5.0        # m/s
    condition: AtmosphericCondition = AtmosphericCondition.CLEAR
    
    # Urban environment parameters
    urban_factor: float = 1.0         # Urban excess attenuation
    foliage_factor: float = 0.0       # Vegetation attenuation
    building_density: float = 0.0     # 0-1, building obstruction

@dataclass
class MolecularAbsorptionLine:
    """Molecular absorption line parameters"""
    frequency_ghz: float
    intensity_db_km: float
    width_ghz: float
    molecule: str  # H2O, O2, N2, etc.

class DOCOMOAtmosphericModels:
    """
    DOCOMO-aligned atmospheric propagation models
    Supporting frequencies up to 1 THz with molecular-level accuracy
    """
    
    def __init__(self):
        """Initialize atmospheric models with molecular data"""
        
        # ITU-R P.676-13 water vapor absorption lines (major lines only)
        self.water_vapor_lines = [
            MolecularAbsorptionLine(22.235, 0.1079, 2.144, "H2O"),
            MolecularAbsorptionLine(67.803, 0.0011, 1.468, "H2O"), 
            MolecularAbsorptionLine(119.995, 0.0007, 1.349, "H2O"),
            MolecularAbsorptionLine(183.310, 2.273, 2.200, "H2O"),
            MolecularAbsorptionLine(321.226, 0.0470, 1.540, "H2O"),
            MolecularAbsorptionLine(325.153, 1.514, 1.760, "H2O"),
            MolecularAbsorptionLine(336.228, 0.0010, 1.890, "H2O"),
            MolecularAbsorptionLine(380.197, 11.67, 1.640, "H2O"),
            MolecularAbsorptionLine(390.134, 0.0045, 1.920, "H2O"),
            MolecularAbsorptionLine(437.347, 0.0632, 1.930, "H2O"),
            MolecularAbsorptionLine(443.018, 0.9098, 1.920, "H2O"),
            MolecularAbsorptionLine(448.001, 0.192, 1.930, "H2O"),
            MolecularAbsorptionLine(470.889, 0.3254, 1.920, "H2O"),
            MolecularAbsorptionLine(474.689, 1.260, 1.920, "H2O"),
            MolecularAbsorptionLine(488.491, 0.2529, 1.920, "H2O"),
            MolecularAbsorptionLine(503.568, 0.0390, 1.930, "H2O"),
            MolecularAbsorptionLine(504.482, 0.0130, 1.920, "H2O"),
            MolecularAbsorptionLine(547.676, 0.9785, 1.920, "H2O"),
            MolecularAbsorptionLine(552.021, 9.009, 1.930, "H2O"),
            MolecularAbsorptionLine(556.936, 13.21, 1.920, "H2O"),
            MolecularAbsorptionLine(620.701, 0.2024, 1.930, "H2O"),
            MolecularAbsorptionLine(645.866, 0.2654, 1.920, "H2O"),
            MolecularAbsorptionLine(658.006, 0.0631, 1.920, "H2O"),
            MolecularAbsorptionLine(752.033, 9.002, 1.920, "H2O"),
            MolecularAbsorptionLine(841.053, 0.0079, 1.920, "H2O"),
            MolecularAbsorptionLine(859.865, 0.0900, 1.920, "H2O"),
            MolecularAbsorptionLine(899.407, 0.0134, 1.920, "H2O"),
        ]
        
        # Oxygen absorption lines (60 GHz complex)
        self.oxygen_lines = [
            MolecularAbsorptionLine(50.474, 0.975, 0.207, "O2"),
            MolecularAbsorptionLine(50.987, 2.529, 0.207, "O2"),
            MolecularAbsorptionLine(51.503, 6.193, 0.207, "O2"),
            MolecularAbsorptionLine(52.021, 14.32, 0.207, "O2"),
            MolecularAbsorptionLine(52.542, 31.24, 0.207, "O2"),
            MolecularAbsorptionLine(53.066, 64.29, 0.207, "O2"),
            MolecularAbsorptionLine(53.596, 124.6, 0.207, "O2"),
            MolecularAbsorptionLine(54.130, 228.0, 0.207, "O2"),
            MolecularAbsorptionLine(54.671, 391.8, 0.207, "O2"),
            MolecularAbsorptionLine(55.221, 631.9, 0.207, "O2"),
            MolecularAbsorptionLine(55.783, 953.5, 0.207, "O2"),
            MolecularAbsorptionLine(56.264, 548.9, 0.207, "O2"),
            MolecularAbsorptionLine(56.363, 1344.0, 0.207, "O2"),
            MolecularAbsorptionLine(56.968, 1763.0, 0.207, "O2"),
            MolecularAbsorptionLine(57.612, 2141.0, 0.207, "O2"),
            MolecularAbsorptionLine(58.323, 2386.0, 0.207, "O2"),
            MolecularAbsorptionLine(58.446, 1457.0, 0.207, "O2"),
            MolecularAbsorptionLine(59.164, 2404.0, 0.207, "O2"),
            MolecularAbsorptionLine(59.590, 2112.0, 0.207, "O2"),
            MolecularAbsorptionLine(60.306, 2124.0, 0.207, "O2"),
            MolecularAbsorptionLine(60.434, 2461.0, 0.207, "O2"),
            MolecularAbsorptionLine(61.150, 2504.0, 0.207, "O2"),
            MolecularAbsorptionLine(61.800, 2298.0, 0.207, "O2"),
            MolecularAbsorptionLine(62.411, 1933.0, 0.207, "O2"),
            MolecularAbsorptionLine(62.486, 1517.0, 0.207, "O2"),
            MolecularAbsorptionLine(62.997, 1503.0, 0.207, "O2"),
            MolecularAbsorptionLine(63.568, 1087.0, 0.207, "O2"),
            MolecularAbsorptionLine(64.127, 733.5, 0.207, "O2"),
            MolecularAbsorptionLine(64.678, 463.5, 0.207, "O2"),
            MolecularAbsorptionLine(65.224, 274.0, 0.207, "O2"),
            MolecularAbsorptionLine(65.764, 153.0, 0.207, "O2"),
            MolecularAbsorptionLine(66.302, 80.40, 0.207, "O2"),
            MolecularAbsorptionLine(66.836, 39.80, 0.207, "O2"),
            MolecularAbsorptionLine(67.369, 18.56, 0.207, "O2"),
            MolecularAbsorptionLine(67.900, 8.172, 0.207, "O2"),
            MolecularAbsorptionLine(68.431, 3.397, 0.207, "O2"),
            MolecularAbsorptionLine(68.960, 1.334, 0.207, "O2")
        ]
        
        # Atmospheric windows (low absorption regions)
        self.atmospheric_windows = [
            (94, 96),    # 95 GHz window
            (134, 146),  # 140 GHz window  
            (217, 223),  # 220 GHz window
            (285, 315),  # 300 GHz window
            (410, 450),  # 430 GHz window
            (850, 1000)  # 900+ GHz window
        ]
        
    def calculate_total_atmospheric_loss(self,
                                       frequency_ghz: float,
                                       distance_km: float,
                                       params: AtmosphericParameters) -> Dict[str, float]:
        """
        Calculate total atmospheric loss with all effects
        
        Args:
            frequency_ghz: Frequency in GHz
            distance_km: Propagation distance in km
            params: Atmospheric parameters
            
        Returns:
            Dictionary with loss components in dB
        """
        losses = {}
        
        # Molecular absorption losses
        losses['water_vapor_db'] = self.calculate_water_vapor_absorption(
            frequency_ghz, distance_km, params
        )
        losses['oxygen_db'] = self.calculate_oxygen_absorption(
            frequency_ghz, distance_km, params
        )
        losses['nitrogen_db'] = self.calculate_nitrogen_absorption(
            frequency_ghz, distance_km, params
        )
        
        # Weather-related losses
        losses['rain_db'] = self.calculate_rain_attenuation(
            frequency_ghz, distance_km, params.rain_rate_mm_h
        )
        losses['fog_db'] = self.calculate_fog_attenuation(
            frequency_ghz, distance_km, params.visibility_m
        )
        losses['snow_db'] = self.calculate_snow_attenuation(
            frequency_ghz, distance_km, params
        )
        
        # Urban environment effects
        losses['urban_excess_db'] = self.calculate_urban_excess_loss(
            frequency_ghz, distance_km, params
        )
        
        # Scintillation (amplitude fluctuation)
        losses['scintillation_std_db'] = self.calculate_scintillation_variance(
            frequency_ghz, distance_km, params
        )
        
        # Total atmospheric loss
        losses['molecular_total_db'] = (
            losses['water_vapor_db'] + 
            losses['oxygen_db'] + 
            losses['nitrogen_db']
        )
        
        losses['weather_total_db'] = (
            losses['rain_db'] + 
            losses['fog_db'] + 
            losses['snow_db']
        )
        
        losses['total_mean_db'] = (
            losses['molecular_total_db'] + 
            losses['weather_total_db'] + 
            losses['urban_excess_db']
        )
        
        return losses
    
    def calculate_water_vapor_absorption(self,
                                       frequency_ghz: float,
                                       distance_km: float,
                                       params: AtmosphericParameters) -> float:
        """
        Calculate water vapor absorption using ITU-R P.676-13
        
        Args:
            frequency_ghz: Frequency in GHz
            distance_km: Distance in km
            params: Atmospheric parameters
            
        Returns:
            Water vapor absorption in dB
        """
        # Convert to absolute humidity
        rho_wv = self._relative_to_absolute_humidity(
            params.humidity_percent, 
            params.temperature_c, 
            params.pressure_hpa
        )
        
        # Total water vapor absorption
        gamma_w = 0.0  # dB/km
        
        # Line absorption
        for line in self.water_vapor_lines:
            # Frequency-dependent line strength
            line_strength = line.intensity_db_km * rho_wv * self._line_shape_factor(
                frequency_ghz, line.frequency_ghz, line.width_ghz, 
                params.temperature_c, params.pressure_hpa
            )
            gamma_w += line_strength
        
        # Non-resonant absorption (continuum)
        gamma_w += self._water_vapor_continuum(
            frequency_ghz, rho_wv, params.temperature_c, params.pressure_hpa
        )
        
        return gamma_w * distance_km
    
    def calculate_oxygen_absorption(self,
                                  frequency_ghz: float,
                                  distance_km: float,
                                  params: AtmosphericParameters) -> float:
        """
        Calculate oxygen absorption using ITU-R P.676-13
        """
        # Dry air density factor
        rho_da = params.pressure_hpa / 1013.25  # Normalized to sea level
        
        gamma_o = 0.0  # dB/km
        
        # 60 GHz oxygen complex
        if 50 <= frequency_ghz <= 70:
            for line in self.oxygen_lines:
                line_strength = line.intensity_db_km * rho_da * self._line_shape_factor(
                    frequency_ghz, line.frequency_ghz, line.width_ghz,
                    params.temperature_c, params.pressure_hpa
                )
                gamma_o += line_strength
        
        # 118.75 GHz line
        if frequency_ghz > 100:
            line_118 = 0.049 * rho_da * self._line_shape_factor(
                frequency_ghz, 118.75, 0.08,
                params.temperature_c, params.pressure_hpa
            )
            gamma_o += line_118
            
        # Non-resonant oxygen absorption
        gamma_o += self._oxygen_continuum(
            frequency_ghz, rho_da, params.temperature_c, params.pressure_hpa
        )
        
        return gamma_o * distance_km
    
    def calculate_nitrogen_absorption(self,
                                    frequency_ghz: float,
                                    distance_km: float,
                                    params: AtmosphericParameters) -> float:
        """
        Calculate nitrogen continuum absorption (pressure broadening)
        """
        # Simplified nitrogen continuum model
        rho_da = params.pressure_hpa / 1013.25
        
        # Temperature factor
        temp_factor = (288.15 / (params.temperature_c + 273.15))**2
        
        # Frequency-dependent absorption coefficient
        gamma_n2 = 4.64e-5 * frequency_ghz**2 * rho_da * temp_factor  # dB/km
        
        return gamma_n2 * distance_km
    
    def calculate_rain_attenuation(self,
                                 frequency_ghz: float,
                                 distance_km: float,
                                 rain_rate_mm_h: float) -> float:
        """
        Calculate rain attenuation using ITU-R P.838-3
        """
        if rain_rate_mm_h <= 0:
            return 0.0
        
        # ITU-R P.838-3 coefficients
        if frequency_ghz <= 35:
            k_h = 0.0000387 * frequency_ghz**0.912
            k_v = k_h
            alpha_h = 0.784 * frequency_ghz**0.158  
            alpha_v = alpha_h
        elif frequency_ghz <= 100:
            k_h = 0.00175 * frequency_ghz**0.5
            k_v = k_h
            alpha_h = 1.0 + 0.00145 * frequency_ghz
            alpha_v = alpha_h
        else:
            # Sub-THz extrapolation
            k_h = 0.0175 * (frequency_ghz / 100)**2
            k_v = k_h
            alpha_h = 1.5
            alpha_v = alpha_h
        
        # Rain attenuation (assume horizontal polarization)
        gamma_rain = k_h * (rain_rate_mm_h**alpha_h)  # dB/km
        
        # Distance factor for rain cell
        effective_distance = min(distance_km, 35.0)  # Rain cell size limit
        
        return gamma_rain * effective_distance
    
    def calculate_fog_attenuation(self,
                                frequency_ghz: float,
                                distance_km: float,
                                visibility_m: float) -> float:
        """
        Calculate fog/cloud attenuation
        """
        if visibility_m > 1000:  # Clear conditions
            return 0.0
        
        # Convert visibility to water content
        if visibility_m >= 500:
            water_content_g_m3 = 0.024  # Light fog
        elif visibility_m >= 200:
            water_content_g_m3 = 0.096  # Moderate fog
        else:
            water_content_g_m3 = 0.384  # Dense fog
        
        # Mie scattering attenuation (Rayleigh approximation)
        wavelength_m = 0.3 / frequency_ghz  # c = 3e8 m/s
        
        # Fog attenuation coefficient
        gamma_fog = (3.91 / visibility_m) * (frequency_ghz / 57.0)**2  # dB/km
        
        return gamma_fog * distance_km
    
    def calculate_snow_attenuation(self,
                                 frequency_ghz: float,
                                 distance_km: float,
                                 params: AtmosphericParameters) -> float:
        """
        Calculate snow attenuation
        """
        if params.condition != AtmosphericCondition.SNOW:
            return 0.0
        
        # Simplified snow attenuation model
        snow_rate_mm_h = 5.0  # Typical snow rate
        
        # Snow attenuation (less than rain due to ice dielectric properties)
        gamma_snow = 0.0001 * frequency_ghz**1.6 * snow_rate_mm_h**0.8
        
        return gamma_snow * distance_km
    
    def calculate_urban_excess_loss(self,
                                  frequency_ghz: float,
                                  distance_km: float,
                                  params: AtmosphericParameters) -> float:
        """
        Calculate urban environment excess loss
        """
        if params.building_density <= 0:
            return 0.0
        
        # Urban excess loss model (empirical)
        base_loss = 0.5 * params.building_density  # dB/km base
        frequency_factor = (frequency_ghz / 28.0)**0.6
        
        gamma_urban = base_loss * frequency_factor  # dB/km
        
        return gamma_urban * distance_km
    
    def calculate_scintillation_variance(self,
                                       frequency_ghz: float,
                                       distance_km: float,
                                       params: AtmosphericParameters) -> float:
        """
        Calculate scintillation variance (amplitude fluctuation)
        """
        # Rytov variance for weak scintillation
        k = 2 * np.pi * frequency_ghz / 300.0  # wave number (1/m)
        
        # Refractive index structure parameter (typical)
        Cn2 = 1e-15  # m^(-2/3), typical for clear air
        
        # Increase Cn2 for weather conditions
        if params.condition in [AtmosphericCondition.LIGHT_RAIN, AtmosphericCondition.FOG_LIGHT]:
            Cn2 *= 5.0
        elif params.condition in [AtmosphericCondition.MODERATE_RAIN, AtmosphericCondition.FOG_DENSE]:
            Cn2 *= 20.0
        elif params.condition == AtmosphericCondition.HEAVY_RAIN:
            Cn2 *= 100.0
        
        # Rytov variance
        sigma2_rytov = 1.23 * Cn2 * k**(7/6) * (distance_km * 1000)**(11/6)
        
        # Convert to dB (standard deviation)
        sigma_scint_db = 4.343 * np.sqrt(sigma2_rytov)  # Convert to dB
        
        return sigma_scint_db
    
    def _relative_to_absolute_humidity(self,
                                     rh_percent: float,
                                     temp_c: float,
                                     pressure_hpa: float) -> float:
        """Convert relative humidity to absolute humidity (g/m³)"""
        temp_k = temp_c + 273.15
        
        # Saturation vapor pressure (Magnus formula)
        es_hpa = 6.1078 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        
        # Actual vapor pressure
        e_hpa = (rh_percent / 100.0) * es_hpa
        
        # Absolute humidity using ideal gas law
        rho_wv = 216.7 * e_hpa / temp_k  # g/m³
        
        return rho_wv
    
    def _line_shape_factor(self,
                         frequency_ghz: float,
                         center_ghz: float,
                         width_ghz: float,
                         temp_c: float,
                         pressure_hpa: float) -> float:
        """
        Calculate absorption line shape factor (Van Vleck-Weisskopf)
        """
        # Temperature and pressure corrections
        temp_factor = (288.15 / (temp_c + 273.15))**0.5
        pressure_factor = pressure_hpa / 1013.25
        
        # Corrected line width
        width_corrected = width_ghz * pressure_factor * temp_factor
        
        # Van Vleck-Weisskopf line shape
        numerator = width_corrected
        denominator = (frequency_ghz - center_ghz)**2 + width_corrected**2
        
        return numerator / denominator
    
    def _water_vapor_continuum(self,
                             frequency_ghz: float,
                             rho_wv: float,
                             temp_c: float,
                             pressure_hpa: float) -> float:
        """Water vapor continuum absorption"""
        temp_k = temp_c + 273.15
        
        # Self-broadening continuum
        gamma_self = 1.013e-14 * rho_wv * frequency_ghz**2 * (288.15 / temp_k)**2.5
        
        # Foreign-broadening continuum
        rho_da = pressure_hpa / 1013.25 * (288.15 / temp_k)
        gamma_foreign = 5.624e-14 * rho_da * rho_wv * frequency_ghz**2 * (288.15 / temp_k)**7.5
        
        return gamma_self + gamma_foreign
    
    def _oxygen_continuum(self,
                        frequency_ghz: float,
                        rho_da: float,
                        temp_c: float,
                        pressure_hpa: float) -> float:
        """Oxygen continuum absorption"""
        temp_k = temp_c + 273.15
        temp_factor = (288.15 / temp_k)**3
        
        # Oxygen continuum coefficient
        gamma_o_cont = 1.40e-15 * rho_da**2 * frequency_ghz**2 * temp_factor
        
        return gamma_o_cont
    
    def is_atmospheric_window(self, frequency_ghz: float, threshold_db_km: float = 0.1) -> bool:
        """
        Check if frequency is in atmospheric window (low absorption)
        
        Args:
            frequency_ghz: Frequency to check
            threshold_db_km: Maximum absorption for window classification
            
        Returns:
            True if in atmospheric window
        """
        # Check predefined windows
        for window_low, window_high in self.atmospheric_windows:
            if window_low <= frequency_ghz <= window_high:
                return True
        
        # Calculate absorption for frequency
        params = AtmosphericParameters()  # Standard conditions
        losses = self.calculate_total_atmospheric_loss(frequency_ghz, 1.0, params)
        
        return losses['molecular_total_db'] <= threshold_db_km
    
    def get_optimal_frequency_band(self,
                                 distance_km: float,
                                 params: AtmosphericParameters,
                                 available_bands_ghz: List[float]) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal frequency band for given conditions
        
        Args:
            distance_km: Propagation distance
            params: Atmospheric conditions
            available_bands_ghz: Available frequency bands
            
        Returns:
            Tuple of (optimal_frequency, loss_analysis)
        """
        band_analysis = {}
        
        for freq_ghz in available_bands_ghz:
            losses = self.calculate_total_atmospheric_loss(freq_ghz, distance_km, params)
            
            band_analysis[freq_ghz] = {
                'total_loss_db': losses['total_mean_db'],
                'molecular_loss_db': losses['molecular_total_db'],
                'weather_loss_db': losses['weather_total_db'],
                'scintillation_std_db': losses['scintillation_std_db'],
                'is_window': self.is_atmospheric_window(freq_ghz)
            }
        
        # Find band with minimum loss
        optimal_freq = min(available_bands_ghz,
                          key=lambda f: band_analysis[f]['total_loss_db'])
        
        return optimal_freq, band_analysis
    
    def predict_weather_impact(self,
                             frequency_ghz: float,
                             distance_km: float,
                             weather_forecast: List[AtmosphericCondition]) -> Dict[str, float]:
        """
        Predict weather impact on link performance
        
        Args:
            frequency_ghz: Operating frequency
            distance_km: Link distance  
            weather_forecast: List of weather conditions
            
        Returns:
            Impact analysis for each condition
        """
        impact_analysis = {}
        
        for condition in weather_forecast:
            # Set parameters for condition
            if condition == AtmosphericCondition.LIGHT_RAIN:
                params = AtmosphericParameters(rain_rate_mm_h=2.5, humidity_percent=80)
            elif condition == AtmosphericCondition.MODERATE_RAIN:
                params = AtmosphericParameters(rain_rate_mm_h=10.0, humidity_percent=90)
            elif condition == AtmosphericCondition.HEAVY_RAIN:
                params = AtmosphericParameters(rain_rate_mm_h=25.0, humidity_percent=95)
            elif condition == AtmosphericCondition.FOG_LIGHT:
                params = AtmosphericParameters(visibility_m=750, humidity_percent=95)
            elif condition == AtmosphericCondition.FOG_DENSE:
                params = AtmosphericParameters(visibility_m=100, humidity_percent=98)
            else:
                params = AtmosphericParameters()
            
            params.condition = condition
            
            losses = self.calculate_total_atmospheric_loss(frequency_ghz, distance_km, params)
            
            impact_analysis[condition.value] = {
                'total_excess_loss_db': losses['total_mean_db'],
                'availability_impact_percent': min(losses['total_mean_db'] / 10.0 * 100, 100),
                'recommended_action': self._get_weather_recommendation(losses)
            }
        
        return impact_analysis
    
    def _get_weather_recommendation(self, losses: Dict[str, float]) -> str:
        """Get recommendation based on atmospheric losses"""
        total_loss = losses['total_mean_db']
        
        if total_loss < 3.0:
            return "Normal operation"
        elif total_loss < 10.0:
            return "Consider power control"
        elif total_loss < 20.0:
            return "Enable adaptive coding"
        else:
            return "Switch to lower frequency band"
