from typing import Dict, List, Tuple

import numpy as np


class WarehousePathPlanner:
    """Path planner for warehouse drone navigation with multi-drone support"""
    
    def __init__(self, 
                warehouse_dims: Tuple[float, float, float],
                shelf_dims: Tuple[float, float, float],
                num_drones: int = 2,
                aisle_x_width: float = 1.5,
                aisle_y_width: float = 1.0,
                inspection_height: float = 1.0,
                safety_margin: float = 0.75,
                map_data: Dict = None
                ):
        """Initialize warehouse path planner
        
        Args:
            warehouse_dims: (width, length, height) of warehouse
            shelf_dims: (width, length, height) of shelves
            num_drones: Number of drones to plan for
            aisle_width: Width of aisles between shelves
            inspection_height: Height at which drone inspects shelves
            safety_margin: Minimum distance to keep from obstacles
        """
        self.warehouse_dims = warehouse_dims
        self.shelf_dims = shelf_dims
        self.num_drones = num_drones
        self.aisle_x_width = aisle_x_width
        self.aisle_y_width = aisle_y_width
        self.inspection_height = inspection_height
        self.safety_margin = safety_margin
        self.num_waypoints_interpolation = 20
        self.map_data = map_data
        
        # Calculate grid of shelf positions
        self.shelf_positions = self._generate_shelf_grid() if map_data is None else map_data['shelf_positions']
        
        # Will be populated with package positions later
        self.package_positions = []
        if self.map_data is not None:
            self.package_positions = self._generate_package_positions()
        
        # Store drone-specific paths
        self.drone_paths = {}
        self.drone_assignments = {}

    def interpolate_path(self, start_pos: Tuple[float, float, float], 
                        end_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Create smooth path between two points"""
        points = []
        for i in range(self.num_waypoints_interpolation):
            t = i / (self.num_waypoints_interpolation - 1)
            point = tuple(start + t * (end - start) 
                        for start, end in zip(start_pos, end_pos))
            points.append(point)
        return points

    def _generate_package_positions(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Generate package positions from shelf positions"""
        package_offsets = [
            # Bottom row (row 1)
            [0.0, -0.8, 0.3],  # Left 
            [0.0, 0.0, 0.3],   # Center
            [0.0, 0.8, 0.3],   # Right
            
            # Middle row (row 2)
            [0.0, -0.8, 0.9],  # Left
            [0.0, 0.0, 0.9],   # Center
            [0.0, 0.8, 0.9],   # Right
            
            # Top row (row 3)
            [0.0, -0.8, 1.5],  # Left
            [0.0, 0.0, 1.5],   # Center
            [0.0, 0.8, 1.5],   # Right
        ]

        # Correct shelf positions
        for i in range(1, len(self.shelf_positions)):
            current_x = self.shelf_positions[i]['position'][0]
            prev_x = self.shelf_positions[i - 1]['position'][0]
            if current_x - prev_x > 0.1 and current_x - prev_x < self.aisle_x_width:
                self.shelf_positions[i]['position'] = (prev_x, self.shelf_positions[i]['position'][1], self.shelf_positions[i]['position'][2])

        package_positions = []
        for shelf_id, shelf in enumerate(self.shelf_positions):
            pos = shelf['position']
            for i, offset in enumerate(package_offsets):
                package_positions.append((f"{shelf_id}_{i}", 
                                        (pos[0] + offset[0], pos[1] + offset[1], offset[2])))
                
        return package_positions

    def _generate_shelf_grid(self) -> List[Dict]:
        """Generate grid of shelf boxes with their dimensions"""
        shelf_boxes = []
        x_start = -self.warehouse_dims[0]/2 + self.aisle_x_width
        y_start = -self.warehouse_dims[1]/2 + self.aisle_y_width
        
        for x in np.arange(
            x_start, self.warehouse_dims[0]/2 - self.aisle_x_width - self.shelf_dims[0], 
            self.aisle_x_width + self.shelf_dims[0]
        ):
            for y in np.arange(
                y_start, self.warehouse_dims[1]/2, 
                self.aisle_y_width + self.shelf_dims[1]    
            ):
                shelf_boxes.append({
                    'position': (x, y, 0),
                    'dimensions': self.shelf_dims
                })
        
        return shelf_boxes

    def assign_packages_to_drones(self) -> Dict[int, List[str]]:
        """Assign packages to drones based on aisle positions
        
        Divides packages among drones by assigning alternating aisles to each drone.
        This ensures drones operate in separate aisles to avoid collisions.
        
        Returns:
            Dict mapping drone IDs to lists of assigned package IDs
        """
        # Group packages by x-coordinate (aisle)
        aisle_packages = {}
        for package_id, pos in self.package_positions:
            x_coord = pos[0]
            if x_coord not in aisle_packages:
                aisle_packages[x_coord] = []
            aisle_packages[x_coord].append((package_id, pos))
        
        # Sort aisles by x-coordinate
        sorted_aisles = sorted(aisle_packages.keys())
        print(f"SORTED AISLES: {sorted_aisles}")
        
        # Initialize assignments
        assignments = {i: [] for i in range(self.num_drones)}
        
        # Assign equal number of aisles to each drone. If odd number, last drone gets one less aisle
        aisles_per_drone = len(sorted_aisles) // self.num_drones
        for i, aisle in enumerate(sorted_aisles):
            drone_id = min(i // aisles_per_drone, self.num_drones - 1)
            for package_id, pos in aisle_packages[aisle]:
                assignments[drone_id].append(package_id)
        
        self.drone_assignments = assignments
        return assignments

    def generate_inspection_paths(self, start_positions: List[Tuple[float, float, float]]) -> Dict[int, List[Tuple[float, float, float]]]:
        """Generate inspection paths for all drones including initial positioning"""
        # First assign packages to drones
        self.assign_packages_to_drones()
        
        drone_paths = {}
        for drone_id in range(self.num_drones):
            main_path = self._generate_drone_path(drone_id)
            
            # Add interpolated path from start position to first inspection point
            start_to_inspection = self.interpolate_path(
                start_positions[drone_id], 
                main_path[0]
            )
            
            # Combine paths
            drone_paths[drone_id] = start_to_inspection + main_path
            
        self.drone_paths = drone_paths
        return drone_paths

    def _generate_drone_path(self, drone_id: int) -> List[Tuple[float, float, float]]:
        """Generate inspection path for a specific drone"""
        points = []
        
        # Get package positions for this drone
        drone_packages = [(pid, pos) for pid, pos in self.package_positions 
                         if pid in self.drone_assignments[drone_id]]
        
        # Group packages by aisle (x-coordinate)
        aisle_packages = {}
        for pid, pos in drone_packages:
            x_coord = pos[0]
            if x_coord not in aisle_packages:
                aisle_packages[x_coord] = []
            aisle_packages[x_coord].append((pid, pos))
        
        # Sort aisles
        sorted_aisles = sorted(aisle_packages.keys())
        
        # Generate path through assigned aisles
        for i, aisle_x in enumerate(sorted_aisles):
            # Get packages in this aisle
            aisle_packages_sorted = sorted(aisle_packages[aisle_x], 
                                         key=lambda x: (x[1][2], x[1][1]))  # Sort by z, then y
            
            # Calculate inspection position to left of aisle
            aisle_center = aisle_x - self.safety_margin
            
            # Add points for each package height level
            z_levels = sorted(set(pos[2] for _, pos in aisle_packages_sorted))
            
            for j, z in enumerate(z_levels):
                # Get y-coordinates of packages at this height
                y_coords = [pos[1] for _, pos in aisle_packages_sorted 
                          if pos[2] == z]
                min_y = min(y_coords) - self.aisle_y_width/4
                max_y = max(y_coords) + self.aisle_y_width/4
                
                if j % 2 == 0:
                    y_points = np.arange(min_y, max_y + self.aisle_y_width/4, 
                                   self.aisle_y_width/4)
                else:
                    y_points = np.arange(max_y, min_y - self.aisle_y_width/4, 
                                   -self.aisle_y_width/4)
                
                for y in y_points:
                    points.append((aisle_center, y, z))
            
            # Add transition points to next aisle if needed
            if i < len(sorted_aisles) - 1:
                next_x = sorted_aisles[i + 1]
                transition_x = (aisle_center + next_x - self.safety_margin) / 2
                points.append((transition_x, y, z))
        
        return points

    def get_next_waypoint(self, 
                         drone_id: int,
                         current_pos: Tuple[float, float, float],
                         current_waypoint_idx: int
                         ) -> Tuple[Tuple[float, float, float], int]:
        """Get next waypoint in inspection sequence for specific drone
        
        Args:
            drone_id: ID of the drone
            current_pos: Current (x,y,z) position of drone
            current_waypoint_idx: Index of current waypoint
        
        Returns:
            next_pos: Next waypoint position
            next_idx: Index of next waypoint
        """
        drone_path = self.drone_paths[drone_id]
        next_idx = (current_waypoint_idx + 1) % len(drone_path)
        return drone_path[next_idx], next_idx
        
    def get_required_yaw(self,
                        current_pos: Tuple[float, float, float],
                        target_pos: Tuple[float, float, float]
                        ) -> float:
        """Calculate required yaw angle to face inspection target
        
        Args:
            current_pos: Current drone position
            target_pos: Target waypoint position
            
        Returns:
            yaw: Required yaw angle in radians
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        return np.arctan2(dy, dx)