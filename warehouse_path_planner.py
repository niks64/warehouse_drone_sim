import numpy as np
from typing import List, Tuple

class WarehousePathPlanner:
    """Path planner for warehouse drone navigation"""
    
    def __init__(self, 
                 warehouse_dims: Tuple[float, float, float],
                 shelf_dims: Tuple[float, float, float],
                 aisle_width: float,
                 inspection_height: float = 1.0,
                 safety_margin: float = 0.3
                ):
        """Initialize warehouse path planner
        
        Args:
            warehouse_dims: (width, length, height) of warehouse
            shelf_dims: (width, length, height) of shelves
            aisle_width: Width of aisles between shelves
            inspection_height: Height at which drone inspects shelves
            safety_margin: Minimum distance to keep from obstacles
        """
        self.warehouse_dims = warehouse_dims
        self.shelf_dims = shelf_dims  
        self.aisle_width = aisle_width
        self.inspection_height = inspection_height
        self.safety_margin = safety_margin
        
        # Calculate grid of shelf positions
        self.shelf_positions = self._generate_shelf_grid()
        
        # Generate inspection waypoints
        self.inspection_points = self._generate_inspection_points()

    def _generate_shelf_grid(self) -> List[Tuple[float, float, float]]:
        """Generate grid of shelf positions based on warehouse layout"""
        positions = []
        x_start = -self.warehouse_dims[0]/2 + 2.0
        y_start = -self.warehouse_dims[1]/2 + 2.0
        
        for x in np.arange(x_start, self.warehouse_dims[0]/2-1, 
                          self.aisle_width + self.shelf_dims[0]):
            for y in np.arange(y_start, self.warehouse_dims[1]/2-1, 3.0):
                positions.append((x, y, self.shelf_dims[2]/2))
                
        return positions

    def _generate_inspection_points(self) -> List[Tuple[float, float, float]]:
        """Generate waypoints for inspecting all shelves
        
        Creates a path that:
        1. Maintains safe distance from obstacles
        2. Positions drone at proper height and orientation for scanning
        3. Covers all shelves systematically
        4. Includes smooth transitions between inspection points
        """
        inspection_points = []
        
        # For each shelf, generate multiple inspection points
        for shelf_x, shelf_y, _ in self.shelf_positions:
            # Front viewing position
            front_point = (
                shelf_x + self.shelf_dims[0]/2 + self.safety_margin,
                shelf_y,
                self.inspection_height
            )
            inspection_points.append(front_point)
            
            # Add points to inspect along the shelf length
            num_side_points = 3  # Number of inspection points along each shelf
            for i in range(num_side_points):
                ratio = (i + 1)/(num_side_points + 1)
                side_point = (
                    shelf_x + self.shelf_dims[0]/2 + self.safety_margin,
                    shelf_y - self.shelf_dims[1]/2 + ratio*self.shelf_dims[1],
                    self.inspection_height
                )
                inspection_points.append(side_point)
        
        # Add transition points between aisles
        aisle_transition_height = self.inspection_height + 0.5
        for i in range(len(self.shelf_positions)-1):
            curr_shelf = self.shelf_positions[i]
            next_shelf = self.shelf_positions[i+1]
            
            # If moving to new aisle, add transition point at higher altitude
            if abs(next_shelf[0] - curr_shelf[0]) > self.aisle_width:
                transition_point = (
                    (curr_shelf[0] + next_shelf[0])/2,
                    curr_shelf[1],
                    aisle_transition_height
                )
                inspection_points.append(transition_point)
        
        return inspection_points

    def get_next_waypoint(self, 
                         current_pos: Tuple[float, float, float],
                         current_waypoint_idx: int
                         ) -> Tuple[Tuple[float, float, float], int]:
        """Get next waypoint in inspection sequence
        
        Args:
            current_pos: Current (x,y,z) position of drone
            current_waypoint_idx: Index of current waypoint
        
        Returns:
            next_pos: Next waypoint position
            next_idx: Index of next waypoint
        """
        # Simple waypoint cycling for now - could add dynamic replanning
        next_idx = (current_waypoint_idx + 1) % len(self.inspection_points)
        return self.inspection_points[next_idx], next_idx
        
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

    def get_full_inspection_path(self) -> List[Tuple[float, float, float]]:
        """Return complete list of inspection waypoints"""
        return self.inspection_points