import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class WarehouseMapper:
    def __init__(self, 
                 warehouse_dims=(10.0, 14.0, 3.0),
                 resolution=0.1,  # Grid resolution in meters
                 mapping_height=2.5,  # Height for overhead scanning
                 client_id=0):
        self.warehouse_dims = warehouse_dims
        self.resolution = resolution
        self.mapping_height = mapping_height
        self.CLIENT = client_id
        
        # Calculate grid dimensions
        self.grid_dims = (
            int(warehouse_dims[0] / resolution),
            int(warehouse_dims[1] / resolution)
        )
        self.occupancy_grid = np.zeros(self.grid_dims)

    def perform_mapping(self):
        """Perform overhead raycasting to map warehouse"""
        print("Starting warehouse mapping...")
        # Generate grid of overhead points
        x_coords = np.linspace(-self.warehouse_dims[0]/2, self.warehouse_dims[0]/2, self.grid_dims[0])
        y_coords = np.linspace(-self.warehouse_dims[1]/2, self.warehouse_dims[1]/2, self.grid_dims[1])
        
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                # Cast ray from above
                start = [x, y, self.mapping_height]
                end = [x, y, 0]
                result = p.rayTest(start, end, physicsClientId=self.CLIENT)[0]
                
                if result[0] != -1:  # If ray hit something
                    hit_height = self.mapping_height - result[2] * (self.mapping_height)
                    if hit_height > 0.5:  # Filter out ground hits
                        self.occupancy_grid[i, j] = 1
        
        print("Mapping complete")
        return self.occupancy_grid

    def process_map(self):
        """Process occupancy grid to identify aisles and shelves"""
        from scipy import ndimage

        # Apply smoothing to remove noise
        smoothed_grid = ndimage.gaussian_filter(self.occupancy_grid, sigma=1.0)
        
        # Detect edges to find shelf boundaries
        edges = ndimage.sobel(smoothed_grid)
        
        # Find connected components (shelves)
        labeled_array, num_features = ndimage.label(self.occupancy_grid > 0.5)
        
        shelf_info = []
        for i in range(1, num_features + 1):
            # Get shelf coordinates
            shelf_coords = np.where(labeled_array == i)
            x_min = min(shelf_coords[0]) * self.resolution - self.warehouse_dims[0]/2
            x_max = max(shelf_coords[0]) * self.resolution - self.warehouse_dims[0]/2
            y_min = min(shelf_coords[1]) * self.resolution - self.warehouse_dims[1]/2
            y_max = max(shelf_coords[1]) * self.resolution - self.warehouse_dims[1]/2
            
            shelf_info.append({
                'center': [(x_min + x_max)/2, (y_min + y_max)/2],
                'dimensions': [x_max - x_min, y_max - y_min]
            })
        
        return shelf_info

    def visualize_map(self, show=True, save=False):
        """Visualize the occupancy grid"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.occupancy_grid.T, origin='lower')
        plt.colorbar(label='Occupancy')
        plt.title('Warehouse Map')
        
        if save:
            plt.savefig('warehouse_map.png')
        if show:
            plt.show()

    def save_map(self, filename='warehouse_map.npz'):
        """Save map data"""
        np.savez(filename, 
                 occupancy_grid=self.occupancy_grid,
                 resolution=self.resolution,
                 warehouse_dims=self.warehouse_dims)

    def load_map(self, filename='warehouse_map.npz'):
        """Load map data"""
        data = np.load(filename)
        self.occupancy_grid = data['occupancy_grid']
        self.resolution = data['resolution']
        self.warehouse_dims = data['warehouse_dims']