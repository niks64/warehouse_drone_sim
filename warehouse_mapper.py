import pickle
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p


class WarehouseMapper:
    """Handles warehouse mapping using raycasting and occupancy grid"""
    
    def __init__(self,
                 warehouse_dims: Tuple[float, float, float],
                 safety_margin: float = 1.0,
                 grid_resolution: float = 0.05,
                 mapping_height: float = None,
                 shelf_height_threshold: float = 0.5
                 ):
        """Initialize warehouse mapper
        
        Args:
            warehouse_dims: (width, length, height) of warehouse
            safety_margin: Distance to keep from walls
            grid_resolution: Size of each grid cell in meters
            mapping_height: Height at which to perform mapping (defaults to warehouse height)
            shelf_height_threshold: Threshold for detecting shelves vs ground
        """
        self.warehouse_dims = warehouse_dims
        self.safety_margin = safety_margin
        self.grid_resolution = grid_resolution
        self.mapping_height = mapping_height or warehouse_dims[2]
        self.shelf_height_threshold = shelf_height_threshold
        
        # Calculate effective mapping area accounting for safety margin
        self.effective_width = warehouse_dims[0] - 2 * safety_margin
        self.effective_length = warehouse_dims[1] - 2 * safety_margin
        
        # Calculate grid dimensions
        self.grid_width = int(self.effective_width / grid_resolution)
        self.grid_length = int(self.effective_length / grid_resolution)
        
        # Initialize occupancy grid (-1: unknown, 0: no shelf, 1: shelf)
        self.occupancy_grid = -np.ones((self.grid_width, self.grid_length))
        
        # Store detected shelf positions
        self.shelf_positions = []

    def generate_mapping_path(self) -> List[Tuple[float, float, float]]:
        """Generate snake-like path to cover warehouse area
        
        Returns:
            List of (x, y, z) waypoints for mapping path
        """
        waypoints = []
        
        # Calculate start position (bottom left of effective area)
        start_x = -self.warehouse_dims[0]/2 + self.safety_margin
        start_y = -self.warehouse_dims[1]/2 + self.safety_margin
        
        # Generate snake pattern
        move_right = True
        for i in range(self.grid_width):
            x = start_x + i * self.grid_resolution
            
            if move_right:
                y_range = np.arange(start_y, start_y + self.effective_length, self.grid_resolution)
            else:
                y_range = np.arange(start_y + self.effective_length, start_y, -self.grid_resolution)
                
            for y in y_range:
                waypoints.append((x, y, self.mapping_height))
            
            move_right = not move_right
            
        return waypoints

    def process_raycast(self, 
                       position: Tuple[float, float, float],
                       client_id: int
                       ) -> None:
        """Process raycast from current position and update occupancy grid
        
        Args:
            position: Current (x,y,z) position
            client_id: PyBullet client ID
        """
        # Cast ray straight down
        ray_start = position
        ray_end = (position[0], position[1], 0)
        
        result = p.rayTest(ray_start, ray_end, physicsClientId=client_id)[0]
        hit_distance = result[2] * self.mapping_height  # Scale by height to get actual distance
        print(f"Hit distance: {hit_distance}")
        
        # Convert position to grid coordinates
        grid_x = int((position[0] + self.warehouse_dims[0]/2 - self.safety_margin) / self.grid_resolution)
        grid_y = int((position[1] + self.warehouse_dims[1]/2 - self.safety_margin) / self.grid_resolution)
        
        # Check if point is within grid bounds
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_length:
            # Update occupancy grid based on hit distance
            # If hit distance is significantly less than mapping height, we've hit a shelf
            if hit_distance < (self.mapping_height - self.shelf_height_threshold):
                self.occupancy_grid[grid_x, grid_y] = 1
                # Store shelf position
                shelf_pos = (position[0], position[1])
                if shelf_pos not in self.shelf_positions:
                    self.shelf_positions.append(shelf_pos)
            else:
                self.occupancy_grid[grid_x, grid_y] = 0

    def extract_shelf_positions(self) -> List[Dict]:
        """Extract shelf positions from occupancy grid
        
        Returns:
            List of shelf positions with their dimensions
        """
        shelf_positions = []
        
        # Group adjacent shelf cells into shelf units
        visited = np.zeros_like(self.occupancy_grid, dtype=bool)
        
        for i in range(self.grid_width):
            for j in range(self.grid_length):
                if self.occupancy_grid[i, j] == 1 and not visited[i, j]:
                    # Found unvisited shelf cell, expand to find full shelf
                    shelf_cells = self._flood_fill(i, j, visited)
                    
                    if len(shelf_cells) > 4:  # Minimum size threshold to filter noise
                        # Convert grid coordinates to world coordinates
                        min_x = min(x for x, _ in shelf_cells) * self.grid_resolution + \
                               (-self.warehouse_dims[0]/2 + self.safety_margin)
                        max_x = (max(x for x, _ in shelf_cells) + 1) * self.grid_resolution + \
                               (-self.warehouse_dims[0]/2 + self.safety_margin)
                        min_y = min(y for _, y in shelf_cells) * self.grid_resolution + \
                               (-self.warehouse_dims[1]/2 + self.safety_margin)
                        max_y = (max(y for _, y in shelf_cells) + 1) * self.grid_resolution + \
                               (-self.warehouse_dims[1]/2 + self.safety_margin)
                        
                        shelf_positions.append({
                            'position': ((min_x + max_x)/2, (min_y + max_y)/2, 0),
                            'dimensions': (max_x - min_x, max_y - min_y, self.warehouse_dims[2])
                        })
        
        return shelf_positions

    def _flood_fill(self, 
                   start_x: int, 
                   start_y: int, 
                   visited: np.ndarray
                   ) -> List[Tuple[int, int]]:
        """Flood fill algorithm to find connected shelf cells
        
        Args:
            start_x: Starting x grid coordinate
            start_y: Starting y grid coordinate
            visited: Array tracking visited cells
            
        Returns:
            List of (x,y) grid coordinates belonging to the shelf
        """
        shelf_cells = []
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack.pop()
            
            if (x < 0 or x >= self.grid_width or 
                y < 0 or y >= self.grid_length or 
                visited[x, y] or 
                self.occupancy_grid[x, y] != 1):
                continue
                
            visited[x, y] = True
            shelf_cells.append((x, y))
            
            # Add adjacent cells to stack
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
            
        return shelf_cells

    def save_occupancy_grid(self, filename: str) -> None:
        """Save occupancy grid and shelf positions to file
        
        Args:
            filename: Path to save file
        """
        data = {
            'occupancy_grid': self.occupancy_grid,
            'shelf_positions': self.shelf_positions,
            'warehouse_dims': self.warehouse_dims,
            'grid_resolution': self.grid_resolution
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            
    @classmethod
    def load_from_file(cls, filename: str) -> 'WarehouseMapper':
        """Load occupancy grid and create mapper instance from file
        
        Args:
            filename: Path to load file
            
        Returns:
            WarehouseMapper instance
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        mapper = cls(
            warehouse_dims=data['warehouse_dims'],
            grid_resolution=data['grid_resolution']
        )
        mapper.occupancy_grid = data['occupancy_grid']
        mapper.shelf_positions = data['shelf_positions']
        
        return mapper