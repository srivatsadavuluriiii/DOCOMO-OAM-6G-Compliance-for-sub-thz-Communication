#!/usr/bin/env python3
"""
3D Ray Tracing Engine for THz Propagation
Implementation of realistic multipath propagation modeling with specular/diffuse
reflections, diffraction, and human body blockage for 6G THz systems.

References:
- Molisch, A. F. (2011). Wireless Communications. 2nd Edition.
- ITU-R P.526-15: Propagation by diffraction
- ITU-R P.2040-1: Effects of building materials and structures
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import scipy.spatial.distance as distance

class MaterialType(Enum):
    CONCRETE = "concrete"
    GLASS = "glass" 
    METAL = "metal"
    DRYWALL = "drywall"
    WOOD = "wood"
    BRICK = "brick"
    HUMAN_TISSUE = "human_tissue"
    VEGETATION = "vegetation"

@dataclass
class MaterialProperties:
    """Material properties for THz propagation"""
    relative_permittivity: complex
    conductivity: float  # S/m
    roughness_rms_m: float
    penetration_loss_db_m: float

@dataclass
class Ray:
    """Ray for propagation modeling"""
    origin: np.ndarray
    direction: np.ndarray
    power_dbm: float
    phase_rad: float
    polarization: str  # 'V' or 'H'
    path_length_m: float
    num_reflections: int
    path_type: str  # 'direct', 'reflected', 'diffracted'

@dataclass
class Obstacle:
    """3D obstacle for ray tracing"""
    vertices: np.ndarray  # 3D vertices
    material: MaterialType
    surface_normal: np.ndarray

class RayTracingEngine:
    """
    3D ray tracing engine for realistic THz propagation.
    
    Features:
    - Specular and diffuse reflections
    - Knife-edge diffraction
    - Material-dependent losses
    - Human body blockage modeling
    - Multiple-ray multipath synthesis
    """
    
    def __init__(self, environment_config: Optional[Dict] = None):
        self.config = environment_config or {}
        
        # Material properties database for THz frequencies
        self.material_db = self._initialize_material_database()
        
        # Ray tracing parameters
        self.max_reflections = self.config.get('max_reflections', 3)
        self.min_ray_power_dbm = self.config.get('min_ray_power_dbm', -100)
        self.angular_resolution_deg = self.config.get('angular_resolution_deg', 5.0)
        
        # Human body model parameters
        self.human_body_params = self._initialize_human_body_model()
        
    def _initialize_material_database(self) -> Dict[MaterialType, MaterialProperties]:
        """Initialize material properties for THz frequencies (200-600 GHz)"""
        return {
            MaterialType.CONCRETE: MaterialProperties(
                relative_permittivity=6.0 + 1.5j,
                conductivity=0.01,
                roughness_rms_m=0.001,  # 1 mm roughness
                penetration_loss_db_m=50.0  # High loss
            ),
            MaterialType.GLASS: MaterialProperties(
                relative_permittivity=6.5 + 0.1j,
                conductivity=1e-12,
                roughness_rms_m=0.0001,  # 0.1 mm smoothness
                penetration_loss_db_m=10.0  # Moderate loss
            ),
            MaterialType.METAL: MaterialProperties(
                relative_permittivity=1.0 + 1000j,  # High conductivity
                conductivity=1e7,
                roughness_rms_m=0.0005,  # 0.5 mm roughness
                penetration_loss_db_m=200.0  # Very high loss
            ),
            MaterialType.DRYWALL: MaterialProperties(
                relative_permittivity=2.5 + 0.2j,
                conductivity=0.001,
                roughness_rms_m=0.002,  # 2 mm roughness
                penetration_loss_db_m=15.0
            ),
            MaterialType.WOOD: MaterialProperties(
                relative_permittivity=2.0 + 0.1j,
                conductivity=1e-4,
                roughness_rms_m=0.003,  # 3 mm roughness
                penetration_loss_db_m=8.0
            ),
            MaterialType.BRICK: MaterialProperties(
                relative_permittivity=4.0 + 0.8j,
                conductivity=0.005,
                roughness_rms_m=0.005,  # 5 mm roughness
                penetration_loss_db_m=30.0
            ),
            MaterialType.HUMAN_TISSUE: MaterialProperties(
                relative_permittivity=50.0 + 25.0j,  # High water content
                conductivity=2.0,
                roughness_rms_m=0.001,
                penetration_loss_db_m=100.0  # Very high absorption
            ),
            MaterialType.VEGETATION: MaterialProperties(
                relative_permittivity=8.0 + 3.0j,
                conductivity=0.1,
                roughness_rms_m=0.01,  # 1 cm roughness
                penetration_loss_db_m=25.0
            )
        }
    
    def _initialize_human_body_model(self) -> Dict[str, Any]:
        """Initialize simplified human body model for blockage"""
        return {
            'height_m': 1.75,
            'width_m': 0.4,
            'depth_m': 0.2,
            'head_radius_m': 0.1,
            'torso_height_m': 0.6,
            'cross_section_m2': 0.7,  # Effective cross-section for THz
            'movement_speed_m_s': 1.0  # Walking speed
        }
    
    def trace_rays(self, tx_pos: np.ndarray, rx_pos: np.ndarray, 
                   freq_ghz: float, obstacles: List[Obstacle],
                   humans: List[Dict] = None) -> List[Ray]:
        """
        Perform 3D ray tracing between transmitter and receiver.
        
        Args:
            tx_pos: Transmitter position [x, y, z] in meters
            rx_pos: Receiver position [x, y, z] in meters
            freq_ghz: Frequency in GHz
            obstacles: List of 3D obstacles
            humans: List of human positions and orientations
            
        Returns:
            List of Ray objects representing multipath components
        """
        rays = []
        wavelength_m = 3e8 / (freq_ghz * 1e9)
        
        # Direct ray (line-of-sight)
        direct_ray = self._trace_direct_ray(tx_pos, rx_pos, freq_ghz, obstacles, humans)
        if direct_ray.power_dbm > self.min_ray_power_dbm:
            rays.append(direct_ray)
        
        # First-order reflections
        reflected_rays = self._trace_reflected_rays(tx_pos, rx_pos, freq_ghz, obstacles, 1)
        rays.extend([r for r in reflected_rays if r.power_dbm > self.min_ray_power_dbm])
        
        # Higher-order reflections (if enabled)
        if self.max_reflections > 1:
            for order in range(2, self.max_reflections + 1):
                higher_order_rays = self._trace_reflected_rays(tx_pos, rx_pos, freq_ghz, obstacles, order)
                rays.extend([r for r in higher_order_rays if r.power_dbm > self.min_ray_power_dbm])
        
        # Diffracted rays (knife-edge)
        diffracted_rays = self._trace_diffracted_rays(tx_pos, rx_pos, freq_ghz, obstacles)
        rays.extend([r for r in diffracted_rays if r.power_dbm > self.min_ray_power_dbm])
        
        return rays
    
    def _trace_direct_ray(self, tx_pos: np.ndarray, rx_pos: np.ndarray, 
                         freq_ghz: float, obstacles: List[Obstacle],
                         humans: List[Dict] = None) -> Ray:
        """Trace direct line-of-sight ray"""
        distance_m = np.linalg.norm(rx_pos - tx_pos)
        direction = (rx_pos - tx_pos) / distance_m
        
        # Free space path loss
        fspl_db = 20 * math.log10(distance_m) + 20 * math.log10(freq_ghz * 1e9) + 20 * math.log10(4 * math.pi / 3e8)
        power_dbm = 30.0 - fspl_db  # Assume 30 dBm transmit power
        
        # Check for blockage by obstacles
        for obstacle in obstacles:
            if self._ray_intersects_obstacle(tx_pos, rx_pos, obstacle):
                # Calculate penetration loss
                material_props = self.material_db[obstacle.material]
                penetration_loss = material_props.penetration_loss_db_m * 0.1  # Assume 10 cm thickness
                power_dbm -= penetration_loss
        
        # Check for human body blockage
        if humans:
            human_loss = self._calculate_human_blockage(tx_pos, rx_pos, humans)
            power_dbm -= human_loss
        
        return Ray(
            origin=tx_pos.copy(),
            direction=direction,
            power_dbm=power_dbm,
            phase_rad=2 * math.pi * distance_m / (3e8 / (freq_ghz * 1e9)),
            polarization='V',
            path_length_m=distance_m,
            num_reflections=0,
            path_type='direct'
        )
    
    def _trace_reflected_rays(self, tx_pos: np.ndarray, rx_pos: np.ndarray,
                            freq_ghz: float, obstacles: List[Obstacle],
                            max_order: int) -> List[Ray]:
        """Trace reflected rays up to given order"""
        reflected_rays = []
        
        # Simple implementation: trace rays to each obstacle surface
        for obstacle in obstacles:
            # Calculate reflection point (simplified - assume planar surfaces)
            reflection_points = self._find_reflection_points(tx_pos, rx_pos, obstacle)
            
            for refl_point in reflection_points:
                # Calculate reflected ray path
                incident_ray = refl_point - tx_pos
                reflected_ray = rx_pos - refl_point
                
                total_distance = np.linalg.norm(incident_ray) + np.linalg.norm(reflected_ray)
                
                # Free space path loss
                fspl_db = 20 * math.log10(total_distance) + 20 * math.log10(freq_ghz * 1e9) + 20 * math.log10(4 * math.pi / 3e8)
                power_dbm = 30.0 - fspl_db
                
                # Reflection loss
                reflection_loss = self._calculate_reflection_loss(obstacle.material, freq_ghz)
                power_dbm -= reflection_loss
                
                # Roughness scattering loss
                roughness_loss = self._calculate_roughness_loss(obstacle.material, freq_ghz)
                power_dbm -= roughness_loss
                
                if power_dbm > self.min_ray_power_dbm:
                    reflected_rays.append(Ray(
                        origin=tx_pos.copy(),
                        direction=reflected_ray / np.linalg.norm(reflected_ray),
                        power_dbm=power_dbm,
                        phase_rad=2 * math.pi * total_distance / (3e8 / (freq_ghz * 1e9)),
                        polarization='V',
                        path_length_m=total_distance,
                        num_reflections=1,
                        path_type='reflected'
                    ))
        
        return reflected_rays
    
    def _trace_diffracted_rays(self, tx_pos: np.ndarray, rx_pos: np.ndarray,
                             freq_ghz: float, obstacles: List[Obstacle]) -> List[Ray]:
        """Trace diffracted rays using knife-edge model"""
        diffracted_rays = []
        wavelength_m = 3e8 / (freq_ghz * 1e9)
        
        for obstacle in obstacles:
            # Find diffraction edges (simplified)
            edge_points = self._find_diffraction_edges(obstacle)
            
            for edge_point in edge_points:
                # Knife-edge diffraction loss (ITU-R P.526)
                diffraction_loss = self._calculate_knife_edge_loss(tx_pos, rx_pos, edge_point, wavelength_m)
                
                total_distance = np.linalg.norm(edge_point - tx_pos) + np.linalg.norm(rx_pos - edge_point)
                fspl_db = 20 * math.log10(total_distance) + 20 * math.log10(freq_ghz * 1e9) + 20 * math.log10(4 * math.pi / 3e8)
                
                power_dbm = 30.0 - fspl_db - diffraction_loss
                
                if power_dbm > self.min_ray_power_dbm:
                    diffracted_rays.append(Ray(
                        origin=tx_pos.copy(),
                        direction=(rx_pos - edge_point) / np.linalg.norm(rx_pos - edge_point),
                        power_dbm=power_dbm,
                        phase_rad=2 * math.pi * total_distance / wavelength_m,
                        polarization='V',
                        path_length_m=total_distance,
                        num_reflections=0,
                        path_type='diffracted'
                    ))
        
        return diffracted_rays
    
    def _calculate_human_blockage(self, tx_pos: np.ndarray, rx_pos: np.ndarray,
                                humans: List[Dict]) -> float:
        """Calculate human body blockage loss"""
        total_loss_db = 0.0
        
        for human in humans:
            human_pos = np.array(human['position'])
            
            # Check if human is in path
            if self._point_in_path(tx_pos, rx_pos, human_pos, self.human_body_params['width_m']):
                # THz human body loss is very high (20-40 dB)
                body_loss_db = 30.0  # Typical loss through human torso at THz
                total_loss_db += body_loss_db
        
        return total_loss_db
    
    def _calculate_reflection_loss(self, material: MaterialType, freq_ghz: float) -> float:
        """Calculate reflection loss for material at THz frequencies"""
        material_props = self.material_db[material]
        eps_r = material_props.relative_permittivity
        
        # Fresnel reflection coefficient (normal incidence, simplified)
        eps_abs = abs(eps_r)
        reflection_coeff = abs((1 - math.sqrt(eps_abs)) / (1 + math.sqrt(eps_abs)))**2
        
        # Convert to dB loss
        reflection_loss_db = -10 * math.log10(reflection_coeff)
        
        return reflection_loss_db
    
    def _calculate_roughness_loss(self, material: MaterialType, freq_ghz: float) -> float:
        """Calculate additional loss due to surface roughness"""
        material_props = self.material_db[material]
        roughness_m = material_props.roughness_rms_m
        wavelength_m = 3e8 / (freq_ghz * 1e9)
        
        # Rayleigh roughness parameter
        roughness_param = 4 * math.pi * roughness_m / wavelength_m
        
        # Roughness scattering loss (empirical)
        if roughness_param < 0.1:
            roughness_loss_db = 0.0  # Smooth surface
        elif roughness_param < 1.0:
            roughness_loss_db = 10 * roughness_param  # Moderate roughness
        else:
            roughness_loss_db = 20 * math.log10(roughness_param)  # Rough surface
        
        return min(roughness_loss_db, 20.0)  # Cap at 20 dB
    
    def _calculate_knife_edge_loss(self, tx_pos: np.ndarray, rx_pos: np.ndarray,
                                 edge_pos: np.ndarray, wavelength_m: float) -> float:
        """Calculate knife-edge diffraction loss (ITU-R P.526)"""
        
        d1 = np.linalg.norm(edge_pos - tx_pos)
        d2 = np.linalg.norm(rx_pos - edge_pos)
        
        # Fresnel parameter
        h = 0.1  # Assume 10 cm clearance for knife edge
        v = h * math.sqrt(2 * (d1 + d2) / (wavelength_m * d1 * d2))
        
        # Diffraction loss (ITU-R P.526 approximation)
        if v <= -0.7:
            diffraction_loss_db = 0
        elif v <= 0:
            diffraction_loss_db = 20 * math.log10(0.5 - 0.62 * v)
        elif v <= 2.4:
            diffraction_loss_db = 20 * math.log10(0.5 * math.exp(-0.95 * v))
        else:
            diffraction_loss_db = 20 * math.log10(0.4 / v)
        
        return max(diffraction_loss_db, 0.0)
    
    def _ray_intersects_obstacle(self, start: np.ndarray, end: np.ndarray, obstacle: Obstacle) -> bool:
        """Check if ray intersects with obstacle (simplified)"""
        # Simplified: check if path passes through obstacle bounding box
        # In practice, would use proper ray-triangle intersection
        return False  # Placeholder
    
    def _point_in_path(self, start: np.ndarray, end: np.ndarray, point: np.ndarray, radius: float) -> bool:
        """Check if point is within radius of line path"""
        # Distance from point to line
        path_vec = end - start
        point_vec = point - start
        
        # Project point onto path
        path_length = np.linalg.norm(path_vec)
        if path_length == 0:
            return False
        
        proj_length = np.dot(point_vec, path_vec) / path_length
        
        # Check if projection is within path segment
        if proj_length < 0 or proj_length > path_length:
            return False
        
        # Calculate perpendicular distance
        proj_point = start + (proj_length / path_length) * path_vec
        perp_distance = np.linalg.norm(point - proj_point)
        
        return perp_distance <= radius
    
    def _find_reflection_points(self, tx_pos: np.ndarray, rx_pos: np.ndarray, 
                              obstacle: Obstacle) -> List[np.ndarray]:
        """Find reflection points on obstacle surface (simplified)"""
        # Simplified: return center of obstacle as reflection point
        if len(obstacle.vertices) > 0:
            center = np.mean(obstacle.vertices, axis=0)
            return [center]
        return []
    
    def _find_diffraction_edges(self, obstacle: Obstacle) -> List[np.ndarray]:
        """Find diffraction edges of obstacle (simplified)"""
        # Simplified: return vertices as potential diffraction points
        return list(obstacle.vertices)
    
    def synthesize_channel_response(self, rays: List[Ray], freq_ghz: float) -> Dict[str, float]:
        """Synthesize channel response from ray collection"""
        if not rays:
            return {
                'total_power_dbm': -float('inf'),
                'rms_delay_spread_ns': 0.0,
                'coherence_bandwidth_mhz': float('inf'),
                'num_paths': 0
            }
        
        # Convert to linear power and sum with phases
        total_power_linear = 0.0
        power_delay_profile = []
        
        for ray in rays:
            power_linear = 10**(ray.power_dbm / 10) / 1000  # Convert to Watts
            delay_ns = ray.path_length_m / 3e8 * 1e9
            
            total_power_linear += power_linear
            power_delay_profile.append((delay_ns, power_linear))
        
        # Calculate metrics
        total_power_dbm = 10 * math.log10(total_power_linear * 1000) if total_power_linear > 0 else -float('inf')
        
        # RMS delay spread
        if len(power_delay_profile) > 1:
            delays = [pdp[0] for pdp in power_delay_profile]
            powers = [pdp[1] for pdp in power_delay_profile]
            
            mean_delay = sum(d * p for d, p in zip(delays, powers)) / sum(powers)
            rms_delay_spread = math.sqrt(sum(p * (d - mean_delay)**2 for d, p in zip(delays, powers)) / sum(powers))
            
            # Coherence bandwidth (approximate)
            coherence_bandwidth_mhz = 1000 / (2 * math.pi * rms_delay_spread) if rms_delay_spread > 0 else float('inf')
        else:
            rms_delay_spread = 0.0
            coherence_bandwidth_mhz = float('inf')
        
        return {
            'total_power_dbm': total_power_dbm,
            'rms_delay_spread_ns': rms_delay_spread,
            'coherence_bandwidth_mhz': coherence_bandwidth_mhz,
            'num_paths': len(rays)
        }

def test_ray_tracing():
    """Test ray tracing engine"""
    engine = RayTracingEngine()
    
    # Simple test scenario
    tx_pos = np.array([0.0, 0.0, 2.0])    # 2m height
    rx_pos = np.array([10.0, 0.0, 1.5])   # 10m away, 1.5m height
    
    # Simple wall obstacle
    wall = Obstacle(
        vertices=np.array([[5.0, -2.0, 0.0], [5.0, 2.0, 0.0], [5.0, 2.0, 3.0], [5.0, -2.0, 3.0]]),
        material=MaterialType.CONCRETE,
        surface_normal=np.array([1.0, 0.0, 0.0])
    )
    
    # Human blockage
    human = {'position': [3.0, 0.0, 1.0]}
    
    print("=== RAY TRACING ENGINE TEST ===")
    
    for freq in [100, 300, 600]:
        print(f"\nFrequency: {freq} GHz")
        
        rays = engine.trace_rays(tx_pos, rx_pos, freq, [wall], [human])
        channel = engine.synthesize_channel_response(rays, freq)
        
        print(f"  Total paths: {channel['num_paths']}")
        print(f"  Total power: {channel['total_power_dbm']:.1f} dBm")
        print(f"  RMS delay spread: {channel['rms_delay_spread_ns']:.1f} ns")
        print(f"  Coherence BW: {channel['coherence_bandwidth_mhz']:.1f} MHz")
        
        for i, ray in enumerate(rays):
            print(f"    Path {i+1}: {ray.path_type}, {ray.power_dbm:.1f} dBm, {ray.path_length_m:.1f}m")

if __name__ == "__main__":
    test_ray_tracing()
