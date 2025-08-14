#!/usr/bin/env python3
"""
OAM Beam Physics Implementation
Complete implementation of Orbital Angular Momentum beam physics with Laguerre-Gaussian modes
Includes helical phase structure, mode orthogonality, and atmospheric coupling effects
"""

import numpy as np
import scipy.special as sp
from scipy.integrate import quad, dblquad
from scipy.constants import c as speed_of_light, pi
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
import math

@dataclass
class OAMBeamParameters:
    """OAM beam configuration parameters"""
    wavelength: float  # wavelength in meters
    waist_radius: float  # beam waist w0 in meters
    aperture_radius: float  # transmit/receive aperture radius in meters
    oam_mode_l: int  # azimuthal mode number (topological charge)
    radial_mode_p: int = 0  # radial mode number (default: fundamental)
    truncation_factor: float = 3.0  # aperture truncation at w0 * truncation_factor

@dataclass
class BeamPropagationResult:
    """Result of beam propagation calculation"""
    electric_field: np.ndarray  # complex electric field
    intensity: np.ndarray  # |E|² intensity pattern
    phase: np.ndarray  # phase pattern
    beam_radius: float  # 1/e² beam radius at distance z
    power_fraction: float  # fraction of power captured by aperture
    mode_purity: float  # purity of OAM mode (0-1)

class OAMBeamPhysics:
    """
    Complete OAM beam physics implementation using Laguerre-Gaussian modes
    
    Implements:
    - Laguerre-Gaussian beam field calculations
    - Helical phase structure exp(ilφ)
    - Mode orthogonality and overlap integrals
    - Atmospheric turbulence coupling
    - Beam divergence and propagation
    - Power coupling efficiency
    
    References:
    - Allen, L., et al. (1992). Orbital angular momentum of light. Phys. Rev. A 45, 8185.
    - Padgett, M., & Allen, L. (2000). Light with a twist in its tail. Contemp. Phys. 41, 275.
    - Krenn, M., et al. (2014). Communication with spatially modulated light. New J. Phys. 16, 113028.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OAM beam physics calculator
        
        Args:
            config: Configuration dictionary with beam parameters
        """
        self.config = config or {}
        
        # Cache for expensive calculations
        self._laguerre_cache = {}
        self._overlap_cache = {}
        self._field_cache = {}
        
        # Default parameters
        self.default_waist = 0.01  # 1 cm beam waist
        self.default_aperture = 0.05  # 5 cm aperture
        
    def laguerre_gaussian_field(self, 
                               r: np.ndarray, 
                               phi: np.ndarray, 
                               z: float,
                               params: OAMBeamParameters) -> BeamPropagationResult:
        """
        Calculate Laguerre-Gaussian beam field with OAM mode l, radial mode p
        
        LG_l^p(r,φ,z) = C * (r√2/w)^|l| * L_p^|l|(2r²/w²) * exp(-r²/w²) * exp(ilφ) * exp(ikz) * exp(-ikr²/2R)
        
        Args:
            r: Radial coordinate array (meters)
            phi: Azimuthal coordinate array (radians)
            z: Propagation distance (meters)
            params: OAM beam parameters
            
        Returns:
            BeamPropagationResult with field, intensity, phase, and metrics
        """
        l = params.oam_mode_l
        p = params.radial_mode_p
        w0 = params.waist_radius
        wavelength = params.wavelength
        
        # Wave number
        k = 2 * pi / wavelength
        
        # Rayleigh range
        z_R = pi * w0**2 / wavelength
        
        # Beam radius at distance z
        w_z = w0 * np.sqrt(1 + (z / z_R)**2)
        
        # Radius of curvature
        R_z = z * (1 + (z_R / z)**2) if z != 0 else np.inf
        
        # Gouy phase
        gouy_phase = (2 * p + abs(l) + 1) * np.arctan(z / z_R)
        
        # Normalization constant
        C = np.sqrt(2 * math.factorial(p) / (pi * math.factorial(p + abs(l))))
        C *= np.sqrt(2) / w0  # Additional normalization for power
        
        # Radial component with Laguerre polynomial
        rho = np.sqrt(2) * r / w_z
        
        # Use cached Laguerre polynomial for efficiency
        cache_key = (p, abs(l), rho.shape, np.mean(rho))
        if cache_key not in self._laguerre_cache:
            if abs(l) == 0 and p == 0:
                # Fundamental Gaussian mode
                laguerre_term = np.ones_like(rho)
            else:
                # General Laguerre polynomial L_p^|l|(2r²/w²)
                laguerre_term = sp.genlaguerre(p, abs(l))(rho**2)
            self._laguerre_cache[cache_key] = laguerre_term
        else:
            laguerre_term = self._laguerre_cache[cache_key]
        
        # Radial amplitude
        radial_amplitude = C * (rho**abs(l)) * laguerre_term * np.exp(-rho**2 / 2)
        
        # Azimuthal phase (helical wavefront)
        azimuthal_phase = l * phi
        
        # Longitudinal phase
        longitudinal_phase = k * z - gouy_phase
        
        # Curvature phase
        if R_z != np.inf:
            curvature_phase = -k * r**2 / (2 * R_z)
        else:
            curvature_phase = np.zeros_like(r)
        
        # Total complex field
        total_phase = azimuthal_phase + longitudinal_phase + curvature_phase
        electric_field = radial_amplitude * np.exp(1j * total_phase)
        
        # Intensity pattern
        intensity = np.abs(electric_field)**2
        
        # Phase pattern
        phase = np.angle(electric_field)
        
        # Calculate power fraction captured by aperture
        if params.aperture_radius > 0:
            aperture_mask = r <= params.aperture_radius
            # Use 2D integration with proper coordinate arrays
            dr = r[1, 0] - r[0, 0] if r.shape[0] > 1 else 1.0
            dphi = phi[0, 1] - phi[0, 0] if phi.shape[1] > 1 else 1.0
            
            # Integrate in cylindrical coordinates: ∫∫ I(r,φ) r dr dφ
            total_power = np.sum(intensity * r) * dr * dphi
            captured_power = np.sum(intensity * r * aperture_mask) * dr * dphi
            power_fraction = captured_power / total_power if total_power > 0 else 0.0
        else:
            power_fraction = 1.0
        
        # Calculate mode purity (simplified)
        mode_purity = self._calculate_mode_purity(electric_field, r, phi, params)
        
        return BeamPropagationResult(
            electric_field=electric_field,
            intensity=intensity,
            phase=phase,
            beam_radius=w_z,
            power_fraction=power_fraction,
            mode_purity=mode_purity
        )
    
    def mode_overlap_integral(self, 
                             l1: int, p1: int,
                             l2: int, p2: int,
                             params1: OAMBeamParameters,
                             params2: OAMBeamParameters,
                             perturbation_matrix: Optional[np.ndarray] = None) -> complex:
        """
        Calculate overlap integral between two OAM modes
        
        LG_l1^p1 | LG_l2^p2 = ∫∫ E*_l1,p1(r,φ) × E_l2,p2(r,φ) r dr dφ
        
        Args:
            l1, p1: First mode indices
            l2, p2: Second mode indices  
            params1, params2: Beam parameters for each mode
            perturbation_matrix: Optional atmospheric perturbation
            
        Returns:
            Complex overlap integral (should be δ_l1,l2 * δ_p1,p2 for perfect modes)
        """
        # Check cache first
        cache_key = (l1, p1, l2, p2, params1.wavelength, params1.waist_radius)
        if perturbation_matrix is None and cache_key in self._overlap_cache:
            return self._overlap_cache[cache_key]
        
        # Create coordinate grids for integration
        r_max = max(params1.aperture_radius, params2.aperture_radius) * 2
        nr, nphi = 128, 64  # Integration grid size
        
        r = np.linspace(0, r_max, nr)
        phi = np.linspace(0, 2*pi, nphi)
        R, PHI = np.meshgrid(r, phi, indexing='ij')
        
        # Calculate field for mode 1 (conjugated)
        import copy
        params1_mod = copy.deepcopy(params1)
        params1_mod.oam_mode_l = l1
        params1_mod.radial_mode_p = p1
        result1 = self.laguerre_gaussian_field(R, PHI, 0.0, params1_mod)
        field1_conj = np.conj(result1.electric_field)
        
        # Calculate field for mode 2
        params2_mod = copy.deepcopy(params2)
        params2_mod.oam_mode_l = l2
        params2_mod.radial_mode_p = p2
        result2 = self.laguerre_gaussian_field(R, PHI, 0.0, params2_mod)
        field2 = result2.electric_field
        
        # Apply atmospheric perturbation if provided
        if perturbation_matrix is not None:
            field2 = field2 * perturbation_matrix
        
        # Overlap integrand
        integrand = field1_conj * field2 * R  # R factor for cylindrical coordinates
        
        # Numerical integration using simple 2D sum
        dr = r[1] - r[0] if len(r) > 1 else 1.0
        dphi = phi[1] - phi[0] if len(phi) > 1 else 1.0
        overlap = np.sum(integrand) * dr * dphi
        
        # Cache result if no perturbation
        if perturbation_matrix is None:
            self._overlap_cache[cache_key] = overlap
        
        return overlap
    
    def atmospheric_mode_coupling(self, 
                                 cn2: float,
                                 l_max: int,
                                 frequency: float,
                                 distance: float,
                                 beam_params: OAMBeamParameters) -> np.ndarray:
        """
        Calculate coupling matrix from atmospheric turbulence using Rytov theory
        
        Args:
            cn2: Refractive index structure parameter [m^(-2/3)]
            l_max: Maximum OAM mode to consider
            frequency: Optical frequency [Hz]
            distance: Propagation distance [m]
            beam_params: Beam parameters
            
        Returns:
            Complex coupling matrix H[i,j] = coupling from mode j to mode i
        """
        wavelength = speed_of_light / frequency
        k = 2 * pi / wavelength
        
        # Number of modes (considering both positive and negative l)
        mode_count = 2 * l_max + 1
        modes = list(range(-l_max, l_max + 1))
        
        # Initialize coupling matrix
        H = np.eye(mode_count, dtype=complex)
        
        # Rytov variance for plane wave
        sigma2_rytov = 1.23 * cn2 * k**(7/6) * distance**(11/6)
        
        # Beam parameter for turbulence interaction
        w0 = beam_params.waist_radius
        
        # Calculate cross-coupling between modes
        for i, l_i in enumerate(modes):
            for j, l_j in enumerate(modes):
                if i != j:
                    # Mode coupling strength depends on |l_i - l_j|
                    delta_l = abs(l_i - l_j)
                    
                    # Empirical model for OAM mode coupling in turbulence
                    # Based on: Ren, Y., et al. (2013). "Atmospheric turbulence effects on orbital angular momentum"
                    if delta_l == 1:
                        # Adjacent mode coupling (strongest)
                        coupling_strength = 0.1 * np.sqrt(sigma2_rytov)
                    elif delta_l == 2:
                        # Next-nearest neighbor coupling
                        coupling_strength = 0.03 * np.sqrt(sigma2_rytov)
                    else:
                        # Higher-order coupling (weaker)
                        coupling_strength = 0.01 * np.sqrt(sigma2_rytov) / delta_l
                    
                    # Phase is random due to turbulence
                    phase = np.random.uniform(0, 2*pi)
                    H[i, j] = coupling_strength * np.exp(1j * phase)
        
        # Ensure total power conservation (normalize rows)
        for i in range(mode_count):
            total_power = np.sum(np.abs(H[i, :])**2)
            if total_power > 1:
                H[i, i] = np.sqrt(1 - np.sum(np.abs(H[i, j])**2 for j in range(mode_count) if j != i))
        
        return H
    
    def beam_divergence_evolution(self, 
                                 z_array: np.ndarray,
                                 params: OAMBeamParameters) -> Dict[str, np.ndarray]:
        """
        Calculate beam divergence evolution with distance
        
        w(z) = w0 * sqrt(1 + (z/z_R)²)
        θ ≈ λ/(πw0) for far field
        
        Args:
            z_array: Array of propagation distances [m]
            params: Beam parameters
            
        Returns:
            Dictionary with beam radius, divergence angle, and Rayleigh range
        """
        w0 = params.waist_radius
        wavelength = params.wavelength
        l = params.oam_mode_l
        
        # Rayleigh range
        z_R = pi * w0**2 / wavelength
        
        # Beam radius evolution
        w_z = w0 * np.sqrt(1 + (z_array / z_R)**2)
        
        # Divergence angle (far-field approximation)
        theta_div = wavelength / (pi * w0)
        
        # OAM mode affects divergence (higher l → larger divergence)
        oam_factor = np.sqrt(1 + abs(l) * wavelength / (pi * w0**2))
        theta_div *= oam_factor
        
        # Phase radius of curvature
        R_z = np.where(z_array != 0, 
                      z_array * (1 + (z_R / z_array)**2), 
                      np.inf)
        
        return {
            'beam_radius': w_z,
            'divergence_angle': theta_div,
            'rayleigh_range': z_R,
            'radius_of_curvature': R_z,
            'oam_divergence_factor': oam_factor
        }
    
    def power_coupling_efficiency(self,
                                 tx_params: OAMBeamParameters,
                                 rx_params: OAMBeamParameters,
                                 distance: float,
                                 misalignment: Tuple[float, float] = (0.0, 0.0),
                                 atmospheric_cn2: float = 1e-15) -> Dict[str, float]:
        """
        Calculate power coupling efficiency between transmitter and receiver
        
        Args:
            tx_params: Transmitter beam parameters
            rx_params: Receiver beam parameters  
            distance: Propagation distance [m]
            misalignment: (lateral_offset [m], angular_offset [rad])
            atmospheric_cn2: Atmospheric turbulence strength
            
        Returns:
            Dictionary with coupling efficiency and loss factors
        """
        # Beam evolution at distance
        tx_evolution = self.beam_divergence_evolution(np.array([distance]), tx_params)
        tx_radius_at_rx = tx_evolution['beam_radius'][0]
        
        # Geometric coupling loss (mode mismatch)
        mode_mismatch_loss = 1.0
        if tx_params.oam_mode_l != rx_params.oam_mode_l:
            delta_l = abs(tx_params.oam_mode_l - rx_params.oam_mode_l)
            mode_mismatch_loss = 0.1**delta_l  # Exponential decay with mode difference
        
        # Size mismatch loss
        size_ratio = min(rx_params.waist_radius / tx_radius_at_rx, 1.0)
        size_mismatch_loss = size_ratio**2
        
        # Aperture truncation loss
        if rx_params.aperture_radius > 0:
            # Fraction of Gaussian beam captured by circular aperture
            aperture_ratio = rx_params.aperture_radius / tx_radius_at_rx
            aperture_loss = 1 - np.exp(-2 * aperture_ratio**2)
        else:
            aperture_loss = 1.0
        
        # Pointing loss (lateral misalignment)
        lateral_offset, angular_offset = misalignment
        pointing_loss = np.exp(-2 * (lateral_offset / tx_radius_at_rx)**2)
        
        # Angular misalignment loss
        angular_loss = np.exp(-2 * (angular_offset * distance / tx_radius_at_rx)**2)
        
        # Atmospheric turbulence loss (scintillation)
        sigma2_rytov = 1.23 * atmospheric_cn2 * (2*pi/tx_params.wavelength)**(7/6) * distance**(11/6)
        scintillation_loss = np.exp(-0.5 * sigma2_rytov)  # Log-normal scintillation
        
        # Total coupling efficiency
        total_efficiency = (mode_mismatch_loss * size_mismatch_loss * 
                           aperture_loss * pointing_loss * 
                           angular_loss * scintillation_loss)
        
        return {
            'total_efficiency': total_efficiency,
            'mode_mismatch_loss_db': -10 * np.log10(mode_mismatch_loss),
            'size_mismatch_loss_db': -10 * np.log10(size_mismatch_loss),
            'aperture_loss_db': -10 * np.log10(aperture_loss),
            'pointing_loss_db': -10 * np.log10(pointing_loss),
            'angular_loss_db': -10 * np.log10(angular_loss),
            'scintillation_loss_db': -10 * np.log10(scintillation_loss),
            'total_loss_db': -10 * np.log10(total_efficiency),
            'tx_beam_radius_at_rx': tx_radius_at_rx
        }
    
    def _calculate_mode_purity(self, 
                              field: np.ndarray,
                              r: np.ndarray, 
                              phi: np.ndarray,
                              params: OAMBeamParameters) -> float:
        """
        Calculate mode purity of the field (simplified metric)
        
        Args:
            field: Complex electric field
            r, phi: Coordinate arrays
            params: Beam parameters
            
        Returns:
            Mode purity (0-1, where 1 is perfect mode)
        """
        # Simplified purity calculation based on phase uniformity
        phase = np.angle(field)
        
        if params.oam_mode_l == 0:
            # For l=0, phase should be approximately constant
            phase_variance = np.var(phase)
            purity = np.exp(-phase_variance)
        else:
            # For l≠0, check helical phase structure
            expected_phase = params.oam_mode_l * phi
            phase_diff = np.angle(np.exp(1j * (phase - expected_phase)))
            phase_variance = np.var(phase_diff)
            purity = np.exp(-phase_variance)
        
        return float(np.clip(purity, 0.0, 1.0))
    
    def validate_orthogonality(self, 
                              l_max: int,
                              beam_params: OAMBeamParameters,
                              tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Validate orthogonality of OAM modes
        
        Args:
            l_max: Maximum mode number to test
            beam_params: Base beam parameters
            tolerance: Numerical tolerance for orthogonality
            
        Returns:
            Validation results dictionary
        """
        modes = list(range(-l_max, l_max + 1))
        results = {
            'orthogonal_pairs': 0,
            'non_orthogonal_pairs': 0,
            'max_crosstalk': 0.0,
            'mode_overlaps': {}
        }
        
        for i, l1 in enumerate(modes):
            for j, l2 in enumerate(modes):
                overlap = self.mode_overlap_integral(
                    l1, 0, l2, 0, beam_params, beam_params
                )
                
                overlap_magnitude = abs(overlap)
                results['mode_overlaps'][(l1, l2)] = overlap_magnitude
                
                if i == j:
                    # Diagonal elements should be ≈ 1
                    if abs(overlap_magnitude - 1.0) > tolerance:
                        results['non_orthogonal_pairs'] += 1
                else:
                    # Off-diagonal elements should be ≈ 0
                    if overlap_magnitude > tolerance:
                        results['non_orthogonal_pairs'] += 1
                        results['max_crosstalk'] = max(results['max_crosstalk'], overlap_magnitude)
                    else:
                        results['orthogonal_pairs'] += 1
        
        results['orthogonality_valid'] = results['non_orthogonal_pairs'] == 0
        
        return results
    
    def generate_mode_set(self, 
                         l_values: List[int],
                         frequency: float,
                         beam_waist: float,
                         aperture_radius: float) -> List[OAMBeamParameters]:
        """
        Generate a set of OAM beam parameters for multiple modes
        
        Args:
            l_values: List of OAM mode numbers
            frequency: Operating frequency [Hz]
            beam_waist: Beam waist radius [m]
            aperture_radius: Aperture radius [m]
            
        Returns:
            List of OAMBeamParameters for each mode
        """
        wavelength = speed_of_light / frequency
        
        beam_set = []
        for l in l_values:
            params = OAMBeamParameters(
                wavelength=wavelength,
                waist_radius=beam_waist,
                aperture_radius=aperture_radius,
                oam_mode_l=l,
                radial_mode_p=0,
                truncation_factor=3.0
            )
            beam_set.append(params)
        
        return beam_set
