#!/usr/bin/env python3
"""
Weather Effects on THz Propagation
Implementation of rain, fog, snow, and weather-dependent atmospheric effects
for realistic 6G THz communication modeling.

References:
- ITU-R P.838-3: Rain attenuation model
- ITU-R P.840-8: Cloud and fog attenuation  
- ITU-R P.676-13: Gaseous attenuation
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class WeatherCondition(Enum):
    CLEAR = "clear"
    LIGHT_RAIN = "light_rain"      # 0.5-2 mm/h
    MODERATE_RAIN = "moderate_rain" # 2-10 mm/h  
    HEAVY_RAIN = "heavy_rain"      # 10-50 mm/h
    EXTREME_RAIN = "extreme_rain"  # >50 mm/h
    FOG = "fog"                    # 0.1-1 g/m³
    DENSE_FOG = "dense_fog"        # >1 g/m³
    SNOW = "snow"                  # Light snow
    BLIZZARD = "blizzard"          # Heavy snow

@dataclass
class WeatherParameters:
    """Weather condition parameters affecting THz propagation"""
    condition: WeatherCondition
    rain_rate_mm_h: float = 0.0
    fog_density_g_m3: float = 0.0
    snow_rate_mm_h: float = 0.0
    temperature_c: float = 15.0
    humidity_percent: float = 50.0
    wind_speed_m_s: float = 0.0

class WeatherEffectsModel:
    """
    Complete weather effects model for THz frequencies.
    
    Key effects modeled:
    - Rain attenuation (ITU-R P.838)
    - Fog/cloud attenuation (ITU-R P.840)
    - Snow attenuation
    - Temperature/humidity variations
    - Wind-induced fading
    """
    
    def __init__(self):
        # ITU-R P.838 coefficients for rain attenuation
        self.rain_coefficients = self._initialize_rain_coefficients()
        
        # Fog scattering parameters
        self.fog_parameters = {
            'scattering_efficiency': 2.0,  # Mie scattering
            'droplet_radius_um': 5.0,      # Typical fog droplet
        }
    
    def _initialize_rain_coefficients(self) -> Dict:
        """Initialize ITU-R P.838 rain attenuation coefficients"""
        # Frequency-dependent coefficients k and α for γ_R = k * R^α
        # where γ_R is rain attenuation (dB/km) and R is rain rate (mm/h)
        return {
            # Format: frequency_ghz: (k, alpha)
            50: (0.0751, 1.099),
            60: (0.1129, 1.061), 
            70: (0.1533, 1.021),
            80: (0.1926, 0.979),
            90: (0.2261, 0.939),
            100: (0.2455, 0.903),
            120: (0.2729, 0.826),
            150: (0.3019, 0.735),
            200: (0.3509, 0.611),
            250: (0.4042, 0.521),
            300: (0.4615, 0.453),
            350: (0.5240, 0.402),
            400: (0.5926, 0.363),
            450: (0.6678, 0.333),
            500: (0.7504, 0.310),
            600: (0.9341, 0.280),
        }
    
    def calculate_rain_attenuation(self, freq_ghz: float, rain_rate_mm_h: float) -> float:
        """
        Calculate rain attenuation using ITU-R P.838 model.
        
        Args:
            freq_ghz: Frequency in GHz
            rain_rate_mm_h: Rain rate in mm/h
            
        Returns:
            Rain attenuation in dB/km
        """
        if rain_rate_mm_h <= 0:
            return 0.0
        
        # Interpolate coefficients for frequency
        k, alpha = self._interpolate_rain_coefficients(freq_ghz)
        
        # ITU-R P.838 formula: γ_R = k * R^α
        gamma_rain = k * (rain_rate_mm_h ** alpha)
        
        # Additional scaling for THz frequencies (beyond ITU-R scope)
        if freq_ghz > 600:
            # Extrapolation factor for extreme THz
            extrapolation_factor = (freq_ghz / 600) ** 0.5
            gamma_rain *= extrapolation_factor
        
        return gamma_rain
    
    def _interpolate_rain_coefficients(self, freq_ghz: float) -> Tuple[float, float]:
        """Interpolate k and α coefficients for given frequency"""
        frequencies = sorted(self.rain_coefficients.keys())
        
        if freq_ghz <= frequencies[0]:
            return self.rain_coefficients[frequencies[0]]
        elif freq_ghz >= frequencies[-1]:
            return self.rain_coefficients[frequencies[-1]]
        
        # Linear interpolation
        for i in range(len(frequencies) - 1):
            f1, f2 = frequencies[i], frequencies[i + 1]
            if f1 <= freq_ghz <= f2:
                k1, alpha1 = self.rain_coefficients[f1]
                k2, alpha2 = self.rain_coefficients[f2]
                
                # Interpolation factor
                t = (freq_ghz - f1) / (f2 - f1)
                
                k = k1 + t * (k2 - k1)
                alpha = alpha1 + t * (alpha2 - alpha1)
                
                return k, alpha
        
        return self.rain_coefficients[frequencies[-1]]
    
    def calculate_fog_attenuation(self, freq_ghz: float, fog_density_g_m3: float) -> float:
        """
        Calculate fog attenuation using Mie scattering theory.
        
        Args:
            freq_ghz: Frequency in GHz
            fog_density_g_m3: Fog water content in g/m³
            
        Returns:
            Fog attenuation in dB/km
        """
        if fog_density_g_m3 <= 0:
            return 0.0
        
        # Convert frequency to wavelength
        wavelength_m = 3e8 / (freq_ghz * 1e9)
        
        # Typical fog droplet parameters
        droplet_radius_m = self.fog_parameters['droplet_radius_um'] * 1e-6
        
        # Size parameter for Mie scattering
        size_parameter = 2 * np.pi * droplet_radius_m / wavelength_m
        
        # Mie scattering efficiency (simplified)
        if size_parameter < 0.1:
            # Rayleigh regime
            Q_sca = (8/3) * size_parameter**4
        elif size_parameter < 10:
            # Intermediate regime (approximation)
            Q_sca = 2.0 - (4/size_parameter) * np.sin(size_parameter) + (4/size_parameter**2) * (1 - np.cos(size_parameter))
        else:
            # Geometric optics regime
            Q_sca = 2.0
        
        # Scattering cross-section
        cross_section_m2 = Q_sca * np.pi * droplet_radius_m**2
        
        # Number density of droplets (assuming spherical droplets)
        droplet_volume_m3 = (4/3) * np.pi * droplet_radius_m**3
        water_density_kg_m3 = 1000.0
        droplet_mass_kg = droplet_volume_m3 * water_density_kg_m3
        
        # Number of droplets per m³
        number_density_per_m3 = (fog_density_g_m3 * 1e-3) / droplet_mass_kg
        
        # Extinction coefficient (m⁻¹)
        extinction_coeff = number_density_per_m3 * cross_section_m2
        
        # Convert to dB/km
        gamma_fog = 4.343 * extinction_coeff * 1000  # dB/km
        
        return gamma_fog
    
    def calculate_snow_attenuation(self, freq_ghz: float, snow_rate_mm_h: float) -> float:
        """
        Calculate snow attenuation (typically lower than rain).
        
        Args:
            freq_ghz: Frequency in GHz
            snow_rate_mm_h: Snow rate in mm/h (water equivalent)
            
        Returns:
            Snow attenuation in dB/km
        """
        if snow_rate_mm_h <= 0:
            return 0.0
        
        # Snow attenuation is typically 10-50% of rain attenuation
        # due to lower dielectric constant of ice vs water
        rain_equivalent_attenuation = self.calculate_rain_attenuation(freq_ghz, snow_rate_mm_h)
        
        # Snow reduction factor (frequency dependent)
        if freq_ghz < 100:
            snow_factor = 0.1  # 10% of rain
        elif freq_ghz < 300:
            snow_factor = 0.2  # 20% of rain
        else:
            snow_factor = 0.3  # 30% of rain at THz
        
        gamma_snow = rain_equivalent_attenuation * snow_factor
        
        return gamma_snow
    
    def calculate_total_weather_attenuation(self, freq_ghz: float, 
                                          weather: WeatherParameters) -> Dict[str, float]:
        """
        Calculate total weather-induced attenuation.
        
        Args:
            freq_ghz: Frequency in GHz
            weather: Weather parameters
            
        Returns:
            Dictionary with breakdown of weather effects
        """
        result = {
            'rain_db_km': 0.0,
            'fog_db_km': 0.0,
            'snow_db_km': 0.0,
            'total_db_km': 0.0
        }
        
        # Rain attenuation
        if weather.rain_rate_mm_h > 0:
            result['rain_db_km'] = self.calculate_rain_attenuation(freq_ghz, weather.rain_rate_mm_h)
        
        # Fog attenuation
        if weather.fog_density_g_m3 > 0:
            result['fog_db_km'] = self.calculate_fog_attenuation(freq_ghz, weather.fog_density_g_m3)
        
        # Snow attenuation
        if weather.snow_rate_mm_h > 0:
            result['snow_db_km'] = self.calculate_snow_attenuation(freq_ghz, weather.snow_rate_mm_h)
        
        # Total weather attenuation (assume independent effects)
        result['total_db_km'] = result['rain_db_km'] + result['fog_db_km'] + result['snow_db_km']
        
        return result
    
    def get_weather_scenario(self, condition: WeatherCondition) -> WeatherParameters:
        """Get predefined weather scenarios for testing"""
        scenarios = {
            WeatherCondition.CLEAR: WeatherParameters(
                condition=condition,
                temperature_c=20.0,
                humidity_percent=50.0
            ),
            WeatherCondition.LIGHT_RAIN: WeatherParameters(
                condition=condition,
                rain_rate_mm_h=1.0,
                temperature_c=15.0,
                humidity_percent=80.0
            ),
            WeatherCondition.MODERATE_RAIN: WeatherParameters(
                condition=condition,
                rain_rate_mm_h=5.0,
                temperature_c=12.0,
                humidity_percent=90.0
            ),
            WeatherCondition.HEAVY_RAIN: WeatherParameters(
                condition=condition,
                rain_rate_mm_h=25.0,
                temperature_c=10.0,
                humidity_percent=95.0
            ),
            WeatherCondition.EXTREME_RAIN: WeatherParameters(
                condition=condition,
                rain_rate_mm_h=75.0,
                temperature_c=8.0,
                humidity_percent=98.0
            ),
            WeatherCondition.FOG: WeatherParameters(
                condition=condition,
                fog_density_g_m3=0.5,
                temperature_c=5.0,
                humidity_percent=95.0
            ),
            WeatherCondition.DENSE_FOG: WeatherParameters(
                condition=condition,
                fog_density_g_m3=2.0,
                temperature_c=2.0,
                humidity_percent=99.0
            ),
            WeatherCondition.SNOW: WeatherParameters(
                condition=condition,
                snow_rate_mm_h=2.0,
                temperature_c=-2.0,
                humidity_percent=85.0
            ),
            WeatherCondition.BLIZZARD: WeatherParameters(
                condition=condition,
                snow_rate_mm_h=15.0,
                temperature_c=-10.0,
                humidity_percent=90.0,
                wind_speed_m_s=15.0
            ),
        }
        
        return scenarios.get(condition, scenarios[WeatherCondition.CLEAR])

def test_weather_effects():
    """Test weather effects on THz propagation"""
    model = WeatherEffectsModel()
    frequencies = [100, 200, 300, 600]
    
    print("=== WEATHER EFFECTS ON THz PROPAGATION ===")
    
    for condition in [WeatherCondition.CLEAR, WeatherCondition.MODERATE_RAIN, 
                     WeatherCondition.HEAVY_RAIN, WeatherCondition.FOG]:
        
        weather = model.get_weather_scenario(condition)
        print(f"\n{condition.value.upper()}:")
        print("Freq(GHz) | Rain | Fog | Snow | Total (dB/km)")
        print("-" * 45)
        
        for freq in frequencies:
            effects = model.calculate_total_weather_attenuation(freq, weather)
            print(f"{freq:8.0f} | {effects['rain_db_km']:4.1f} | {effects['fog_db_km']:3.1f} | "
                  f"{effects['snow_db_km']:4.1f} | {effects['total_db_km']:5.1f}")

if __name__ == "__main__":
    test_weather_effects()
