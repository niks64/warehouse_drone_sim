from typing import Dict, List, Tuple

import numpy as np
import pybullet as p


class WarehouseMapper:
    """Maps rectangular warehouse layout and generates inspection waypoints"""
    
    def __init__(self,
                 initial_pos: np.ndarray,
                 scan_height: float = 2.0,
                 grid_resolution: float = 0.5,
                 client_id: int = 0
                 ):
        """Initialize warehouse mapper
        
        Args:
            initial_pos: Starting position of drone
            scan_height: Height for initial scanning
            grid_resolution: Resolution of occupancy grid
            client_id: PyBullet client ID
        """
        self.initial_pos = initial_pos
        self.scan_height = scan_height
        self.grid_resolution = grid_resolution
        self.client_id = client_id
        
        # Discovered layout information
        self.boundaries = None  # (min_x, max_x, min_y, max_y)
        self.occupancy_grid = None
        self.main_aisles = []
        self.cross_aisles = []
        self.inspection_points = []
        
    def detect_boundaries(self) -> Tuple[float, float, float, float]:
        """Detect warehouse boundaries using raycasting
        
        Returns:
            boundaries: (min_x, max_x, min_y, max_y) of warehouse
        """
        # Cast rays in cardinal directions to find walls
        max_distance = 50.0
        start = np.array([
            self.initial_pos[0],
            self.initial_pos[1],
            self.scan_height
        ])
        
        directions = [
            [1, 0, 0],   # +X
            [-1, 0, 0],  # -X
            [0, 1, 0],   # +Y
            [0, -1, 0]   # -Y
        ]
        
        bounds = []
        for direction in directions:
            result = p.rayTest(
                start,
                start + np.array(direction) * max_distance,
                physicsClientId=self.client_id
            )[0]
            
            if result[0] >= 0:  # Hit something
                hit_distance = result[2] * max_distance
                bounds.append(start + np.array(direction) * hit_distance)
        
        # Extract boundaries from hit points
        hit_points = np.array(bounds)
        min_x = np.min(hit_points[:, 0])
        max_x = np.max(hit_points[:, 0])
        min_y = np.min(hit_points[:, 1])
        max_y = np.max(hit_points[:, 1])
        
        self.boundaries = (min_x, max_x, min_y, max_y)
        return self.boundaries
    
    def create_occupancy_grid(self):
        """Create occupancy grid of warehouse space"""
        if self.boundaries is None:
            self.detect_boundaries()
            
        min_x, max_x, min_y, max_y = self.boundaries
        
        # Create grid
        x_size = int((max_x - min_x) / self.grid_resolution)
        y_size = int((max_y - min_y) / self.grid_resolution)
        self.occupancy_grid = np.zeros((x_size, y_size))
        
        # Scan grid points
        for i in range(x_size):
            for j in range(y_size):
                x = min_x + i * self.grid_resolution
                y = min_y + j * self.grid_resolution
                
                # Cast ray down from scan height
                start = np.array([x, y, self.scan_height])
                end = np.array([x, y, 0])
                
                result = p.rayTest(
                    start,
                    end,
                    physicsClientId=self.client_id
                )[0]
                
                # Mark as occupied if ray hits something above floor level
                if result[0] >= 0 and result[2] * self.scan_height > 0.1:
                    self.occupancy_grid[i, j] = 1
                    
        return self.occupancy_grid
    
    def detect_aisles(self):
        """Detect main and cross aisles from occupancy grid"""
        if self.occupancy_grid is None:
            self.create_occupancy_grid()
            
        # Find continuous free spaces in x and y directions
        x_size, y_size = self.occupancy_grid.shape
        
        # Detect main aisles (along y-axis)
        for i in range(x_size):
            if np.sum(self.occupancy_grid[i, :]) < y_size * 0.3:  # Mostly free space
                self.main_aisles.append(i)
                
        # Detect cross aisles (along x-axis)
        for j in range(y_size):
            if np.sum(self.occupancy_grid[:, j]) < x_size * 0.3:  # Mostly free space
                self.cross_aisles.append(j)
                
        return self.main_aisles, self.cross_aisles
    
    def generate_inspection_waypoints(self,
                                    inspection_height: float = 1.0,
                                    points_per_aisle: int = 10  # Number of points along each aisle
                                    ) -> List[Tuple[float, float, float]]:
        """Generate waypoints for warehouse inspection in a snake pattern
        
        Args:
            inspection_height: Height for inspection points
            points_per_aisle: Number of points to generate along each aisle
                
        Returns:
            waypoints: List of (x,y,z) inspection points
        """
        if not self.main_aisles:
            self.detect_aisles()
                
        min_x, max_x, min_y, max_y = self.boundaries
        waypoints = []
        
        # Generate snake pattern through main aisles
        moving_up = True  # Direction flag for snake pattern
        
        for i, aisle_x in enumerate(self.main_aisles):
            x = min_x + aisle_x * self.grid_resolution
            
            # Generate points along current aisle
            if moving_up:
                # Move up the aisle
                y_points = np.linspace(min_y + self.grid_resolution, 
                                    max_y - self.grid_resolution, 
                                    points_per_aisle)
            else:
                # Move down the aisle
                y_points = np.linspace(max_y - self.grid_resolution, 
                                    min_y + self.grid_resolution, 
                                    points_per_aisle)
                
            # Add points along current aisle
            for y in y_points:
                waypoints.append((x, y, inspection_height))
                
            # If not the last aisle, add transition to next aisle
            if i < len(self.main_aisles) - 1:
                next_x = min_x + self.main_aisles[i + 1] * self.grid_resolution
                # Add a point to transition to next aisle at current height
                if moving_up:
                    waypoints.append((next_x, max_y - self.grid_resolution, inspection_height))
                else:
                    waypoints.append((next_x, min_y + self.grid_resolution, inspection_height))
                    
            # Flip direction for next aisle
            moving_up = not moving_up
        
        print(f"Generated {len(waypoints)} waypoints in snake pattern")
        self.inspection_points = waypoints
        return waypoints

    def optimize_inspection_route(self) -> List[Tuple[float, float, float]]:
        """Return inspection waypoints - no additional optimization needed 
        since snake pattern is already efficient"""
        if not self.inspection_points:
            self.generate_inspection_waypoints()
        return self.inspection_points