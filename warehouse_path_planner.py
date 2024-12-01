from typing import List, Tuple

import numpy as np


class WarehousePathPlanner:
    """Path planner for warehouse drone navigation"""
    
    def __init__(self, 
                warehouse_dims: Tuple[float, float, float],
                shelf_dims: Tuple[float, float, float],
                aisle_x_width: float = 1.5,
                aisle_y_width: float = 1.0,
                inspection_height: float = 1.0,
                safety_margin: float = 0.75
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

        self.aisle_x_width = aisle_x_width
        self.aisle_y_width = aisle_y_width

        self.inspection_height = inspection_height
        self.safety_margin = safety_margin
        
        # Calculate grid of shelf positions
        self.shelf_positions = self._generate_shelf_grid()

    def _generate_shelf_grid(self) -> List[Tuple[float, float, float]]:
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
                # Store shelf as (position, dimensions)
                shelf_boxes.append({
                    'position': (x, y, 0),
                    'dimensions': self.shelf_dims
                })
        
        return shelf_boxes

    def _generate_inspection_points(self) -> List[Tuple[float, float, float]]:
        """Generate inspection points in a snake pattern through the aisles"""
        points = []
       
        aisle_x_coords = sorted(list(set([shelf['position'][0] for shelf in self.shelf_positions])))
        min_y = min([shelf['position'][1] for shelf in self.shelf_positions])
        max_y = max([shelf['position'][1] + shelf['dimensions'][1] for shelf in self.shelf_positions])
        
        moving_up = True
        for i, x in enumerate(aisle_x_coords):
            aisle_center = x - self.safety_margin
            
            if moving_up:
                for j in np.arange(min_y - self.aisle_y_width/2, self.warehouse_dims[1]/2, self.aisle_y_width/2):
                    points.append((aisle_center, j, self.inspection_height))
            else:
                for j in np.arange(max_y + self.aisle_y_width/2, -self.warehouse_dims[1]/2, -self.aisle_y_width/2):
                    points.append((aisle_center, j, self.inspection_height))

            # Transition to next aisle
            if i < len(aisle_x_coords) - 1:
                next_x = (aisle_center + aisle_x_coords[i + 1] - self.safety_margin) / 2
                points.append((next_x, j, self.inspection_height))
            
            moving_up = not moving_up
        
        self.inspection_points = points

        return points
    
    def generate_inspection_waypoints_package_positions(self) -> List[Tuple[float, float, float]]:
        # Arrange self.package_positions in a snake pattern. Generate inspect points to the left of each position. when you are about to switch to the next aisle and the x changes, add tow transition points that move close to the end of the aisle, then move over to the next aisle so that it can start moving down.
        points = []

        aisle_x_coords = sorted(list(set([package[1][0] for package in self.package_positions])))
        min_y = min([package[1][1] for package in self.package_positions]) - self.aisle_y_width/2
        max_y = max([package[1][1] for package in self.package_positions]) + self.aisle_y_width/2
        aisle_z_coords = sorted(list(set([package[1][2] for package in self.package_positions])))

        print(f"There are {len(aisle_z_coords)} rows of packages")

        moving_up = True
        for i, x in enumerate(aisle_x_coords):
            aisle_center = x - self.safety_margin

            for z in aisle_z_coords:
                if moving_up:
                    for j in np.arange(min_y - self.aisle_y_width/4, max_y + self.aisle_y_width/4, self.aisle_y_width/4):
                        points.append((aisle_center, j, z))
                else:
                    for j in np.arange(max_y + self.aisle_y_width/4, min_y - self.aisle_y_width/4, -self.aisle_y_width/4):
                        points.append((aisle_center, j, z))

                moving_up = not moving_up

            # Transition to next aisle
            if i < len(aisle_x_coords) - 1:
                next_x = (aisle_center + aisle_x_coords[i + 1] - self.safety_margin) / 2
                points.append((next_x, j, z))

        self.inspection_points = points

        return points

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