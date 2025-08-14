#!/usr/bin/env python3
"""
Test suite for OAM beam physics implementation
Validates Laguerre-Gaussian modes, orthogonality, and atmospheric coupling
"""

import pytest
import numpy as np
import sys
import os
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import directly to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location(
    "oam_beam_physics", 
    os.path.join(os.path.dirname(__file__), '..', '..', 'simulator', 'oam_beam_physics.py')
)
oam_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oam_module)

OAMBeamPhysics = oam_module.OAMBeamPhysics
OAMBeamParameters = oam_module.OAMBeamParameters
BeamPropagationResult = oam_module.BeamPropagationResult

from scipy.constants import c as speed_of_light, pi

class TestOAMBeamPhysics:
    """Test class for OAM beam physics"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.oam_physics = OAMBeamPhysics()
        self.frequency = 300e9  # 300 GHz
        self.wavelength = speed_of_light / self.frequency
        self.beam_waist = 0.01  # 1 cm
        self.aperture_radius = 0.05  # 5 cm
        
        self.test_params = OAMBeamParameters(
            wavelength=self.wavelength,
            waist_radius=self.beam_waist,
            aperture_radius=self.aperture_radius,
            oam_mode_l=1,
            radial_mode_p=0
        )
    
    def test_fundamental_gaussian_mode(self):
        """Test l=0 mode (fundamental Gaussian)"""
        import copy
        params = copy.deepcopy(self.test_params)
        params.oam_mode_l = 0
        
        # Create coordinate grid
        r = np.linspace(0, 0.03, 50)  # 3 cm radial extent
        phi = np.linspace(0, 2*pi, 64)
        R, PHI = np.meshgrid(r, phi, indexing='ij')
        
        # Calculate field
        result = self.oam_physics.laguerre_gaussian_field(R, PHI, 0.0, params)
        
        # Assertions for fundamental mode
        assert isinstance(result, BeamPropagationResult)
        assert result.electric_field.shape == R.shape
        assert np.all(np.isfinite(result.electric_field))
        assert np.all(result.intensity >= 0)
        
        # Phase should be approximately constant for l=0
        phase_variance = np.var(result.phase)
        assert phase_variance < 0.1, f"Phase variance {phase_variance} too large for l=0 mode"
        
        # Beam should be circularly symmetric (relaxed test due to discrete grid)
        intensity_center = result.intensity[R <= self.beam_waist/2]
        cv = np.std(intensity_center) / np.mean(intensity_center)
        assert cv < 0.2, f"Coefficient of variation {cv} too high for circular symmetry"
    
    def test_oam_helical_phase(self):
        """Test helical phase structure for l≠0"""
        import copy
        for l in [1, -1, 2, -2]:
            params = copy.deepcopy(self.test_params)
            params.oam_mode_l = l
            
            # Create coordinate grid focused on phase structure
            r = np.full((64, 64), self.beam_waist)  # Fixed radius
            phi = np.linspace(0, 2*pi, 64)
            PHI = np.tile(phi, (64, 1))
            R = r
            
            # Calculate field
            result = self.oam_physics.laguerre_gaussian_field(R, PHI, 0.0, params)
            
            # Check helical phase structure at constant radius
            # Phase should vary as l*φ around the circle
            phi_line = PHI[32, :]  # Middle row
            phase_line = result.phase[32, :]
            
            # Calculate phase gradient (should be approximately l)
            phase_diff = np.diff(phase_line)
            phi_diff = np.diff(phi_line)
            
            # Handle phase wrapping in differences
            phase_diff = np.where(phase_diff > pi, phase_diff - 2*pi, phase_diff)
            phase_diff = np.where(phase_diff < -pi, phase_diff + 2*pi, phase_diff)
            
            # Average gradient should be l
            avg_gradient = np.mean(phase_diff / phi_diff)
            expected_gradient = l
            
            gradient_error = abs(avg_gradient - expected_gradient)
            assert gradient_error < 0.2, f"Phase gradient error {gradient_error} for l={l} (expected {expected_gradient}, got {avg_gradient})"
    
    def test_beam_divergence_evolution(self):
        """Test beam divergence with propagation distance"""
        z_array = np.logspace(-3, 1, 50)  # 1 mm to 10 m
        
        evolution = self.oam_physics.beam_divergence_evolution(z_array, self.test_params)
        
        # Check Rayleigh range calculation
        expected_zR = pi * self.beam_waist**2 / self.wavelength
        assert abs(evolution['rayleigh_range'] - expected_zR) / expected_zR < 0.01
        
        # Check beam radius evolution
        w_z = evolution['beam_radius']
        assert abs(w_z[0] - self.beam_waist) / self.beam_waist < 0.01  # At z=0 within 1%
        assert np.all(w_z >= self.beam_waist)  # Always larger than waist
        assert np.all(np.diff(w_z) >= 0)  # Monotonically increasing
        
        # Check far-field divergence (including OAM factor)
        far_field_indices = z_array > 10 * expected_zR
        if np.any(far_field_indices):
            theta_div = evolution['divergence_angle']
            expected_theta_base = self.wavelength / (pi * self.beam_waist)
            oam_factor = evolution['oam_divergence_factor']
            expected_theta = expected_theta_base * oam_factor
            assert abs(theta_div - expected_theta) / expected_theta < 0.1
    
    def test_mode_orthogonality(self):
        """Test orthogonality between different OAM modes"""
        l_values = [-2, -1, 0, 1, 2]
        
        # Test mode overlaps
        import copy
        for l1 in l_values:
            for l2 in l_values:
                params1 = copy.deepcopy(self.test_params)
                params1.oam_mode_l = l1
                params2 = copy.deepcopy(self.test_params)
                params2.oam_mode_l = l2
                
                overlap = self.oam_physics.mode_overlap_integral(l1, 0, l2, 0, params1, params2)
                overlap_magnitude = abs(overlap)
                
                if l1 == l2:
                    # Same modes should have high overlap (≈1)
                    assert overlap_magnitude > 0.9, f"Self-overlap {overlap_magnitude} too low for l={l1}"
                else:
                    # Different modes should have low overlap (≈0)
                    assert overlap_magnitude < 0.1, f"Cross-overlap {overlap_magnitude} too high for l1={l1}, l2={l2}"
    
    def test_power_coupling_efficiency(self):
        """Test power coupling efficiency calculations"""
        import copy
        tx_params = copy.deepcopy(self.test_params)
        tx_params.oam_mode_l = 1
        rx_params = copy.deepcopy(self.test_params)
        rx_params.oam_mode_l = 1
        distance = 1000.0  # 1 km
        
        # Perfect alignment case
        efficiency = self.oam_physics.power_coupling_efficiency(
            tx_params, rx_params, distance, misalignment=(0.0, 0.0)
        )
        
        assert 0 <= efficiency['total_efficiency'] <= 1
        assert efficiency['total_loss_db'] >= 0
        assert efficiency['mode_mismatch_loss_db'] < 0.1  # Same modes
        
        # Mode mismatch case
        rx_params_mismatch = copy.deepcopy(rx_params)
        rx_params_mismatch.oam_mode_l = 2
        efficiency_mismatch = self.oam_physics.power_coupling_efficiency(
            tx_params, rx_params_mismatch, distance
        )
        
        assert efficiency_mismatch['total_efficiency'] < efficiency['total_efficiency']
        assert efficiency_mismatch['mode_mismatch_loss_db'] > 5.0  # Significant loss
        
        # Misalignment case
        lateral_offset = 0.1  # 10 cm offset
        efficiency_misaligned = self.oam_physics.power_coupling_efficiency(
            tx_params, rx_params, distance, misalignment=(lateral_offset, 0.0)
        )
        
        assert efficiency_misaligned['total_efficiency'] < efficiency['total_efficiency']
        assert efficiency_misaligned['pointing_loss_db'] > 1.0
    
    def test_atmospheric_mode_coupling(self):
        """Test atmospheric turbulence coupling matrix"""
        cn2 = 1e-14  # Strong turbulence
        l_max = 3
        distance = 1000.0
        
        coupling_matrix = self.oam_physics.atmospheric_mode_coupling(
            cn2, l_max, self.frequency, distance, self.test_params
        )
        
        mode_count = 2 * l_max + 1
        assert coupling_matrix.shape == (mode_count, mode_count)
        
        # Check matrix properties
        assert np.allclose(np.diag(coupling_matrix).real, 1.0, atol=0.1)  # Diagonal ≈ 1
        assert np.all(np.abs(coupling_matrix) <= 1.0)  # All elements ≤ 1
        
        # Check power conservation (approximately)
        for i in range(mode_count):
            total_power = np.sum(np.abs(coupling_matrix[i, :])**2)
            assert 0.8 <= total_power <= 1.2  # Allow some numerical error
    
    def test_validate_orthogonality_function(self):
        """Test the orthogonality validation function"""
        l_max = 2
        
        validation = self.oam_physics.validate_orthogonality(l_max, self.test_params)
        
        assert isinstance(validation, dict)
        assert 'orthogonality_valid' in validation
        assert 'max_crosstalk' in validation
        assert 'mode_overlaps' in validation
        
        # Should have reasonable orthogonality for small l_max
        assert validation['max_crosstalk'] < 0.5
        
    def test_generate_mode_set(self):
        """Test mode set generation"""
        l_values = [-2, -1, 0, 1, 2]
        
        mode_set = self.oam_physics.generate_mode_set(
            l_values, self.frequency, self.beam_waist, self.aperture_radius
        )
        
        assert len(mode_set) == len(l_values)
        
        for i, params in enumerate(mode_set):
            assert isinstance(params, OAMBeamParameters)
            assert params.oam_mode_l == l_values[i]
            assert params.wavelength == self.wavelength
            assert params.waist_radius == self.beam_waist
            assert params.aperture_radius == self.aperture_radius
    
    def test_high_order_modes(self):
        """Test higher-order OAM modes (l > 5)"""
        import copy
        high_l_values = [5, 10, -8]
        
        for l in high_l_values:
            params = copy.deepcopy(self.test_params)
            params.oam_mode_l = l
            
            # Create small coordinate grid for computational efficiency
            r = np.linspace(0, 0.02, 30)
            phi = np.linspace(0, 2*pi, 32)
            R, PHI = np.meshgrid(r, phi, indexing='ij')
            
            # Should not raise exceptions
            result = self.oam_physics.laguerre_gaussian_field(R, PHI, 0.0, params)
            
            assert np.all(np.isfinite(result.electric_field))
            assert np.all(result.intensity >= 0)
            assert 0 <= result.mode_purity <= 1
    
    def test_radial_modes(self):
        """Test radial modes (p > 0)"""
        import copy
        for p in [1, 2]:
            params = copy.deepcopy(self.test_params)
            params.radial_mode_p = p
            
            r = np.linspace(0, 0.03, 40)
            phi = np.linspace(0, 2*pi, 32)
            R, PHI = np.meshgrid(r, phi, indexing='ij')
            
            result = self.oam_physics.laguerre_gaussian_field(R, PHI, 0.0, params)
            
            # Radial modes should have intensity nulls
            radial_profile = result.intensity[:, 0]  # Along phi=0
            
            # Should have at least one minimum (characteristic of p>0 modes)
            min_indices = np.where(np.diff(np.sign(np.diff(radial_profile))) > 0)[0]
            assert len(min_indices) >= p, f"Insufficient radial nulls for p={p} mode"
    
    def test_aperture_effects(self):
        """Test aperture truncation effects"""
        import copy
        # Small aperture
        params_small = copy.deepcopy(self.test_params)
        params_small.aperture_radius = 0.01  # 1 cm
        # Large aperture  
        params_large = copy.deepcopy(self.test_params)
        params_large.aperture_radius = 0.1   # 10 cm
        
        distance = 100.0  # 100 m
        
        eff_small = self.oam_physics.power_coupling_efficiency(
            params_small, params_small, distance
        )
        eff_large = self.oam_physics.power_coupling_efficiency(
            params_large, params_large, distance
        )
        
        # Larger aperture should have higher efficiency
        assert eff_large['total_efficiency'] >= eff_small['total_efficiency']
        assert eff_small['aperture_loss_db'] >= eff_large['aperture_loss_db']
    
    def test_numerical_stability(self):
        """Test numerical stability at edge cases"""
        import copy
        # Very small beam waist
        params_small = copy.deepcopy(self.test_params)
        params_small.waist_radius = 1e-6  # 1 μm
        
        # Very large distance
        large_distance = 1e6  # 1000 km
        
        # Should not raise exceptions or produce NaN/inf
        evolution = self.oam_physics.beam_divergence_evolution(
            np.array([large_distance]), params_small
        )
        
        assert np.all(np.isfinite(evolution['beam_radius']))
        assert np.all(np.isfinite(evolution['divergence_angle']))
        
        # Very high OAM mode
        params_high_l = copy.deepcopy(self.test_params)
        params_high_l.oam_mode_l = 50
        
        r = np.linspace(0, 0.01, 20)  # Small grid for speed
        phi = np.linspace(0, 2*pi, 20)
        R, PHI = np.meshgrid(r, phi, indexing='ij')
        
        result = self.oam_physics.laguerre_gaussian_field(R, PHI, 0.0, params_high_l)
        assert np.all(np.isfinite(result.electric_field))

if __name__ == "__main__":
    pytest.main([__file__])
