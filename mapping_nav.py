import time

import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from warehouse_env import WarehouseAviary
from warehouse_mapper import WarehouseMapper

# Configuration
WAREHOUSE_DIMS = (10.0, 14.0, 3.0)
SHELF_DIMS = (0.5, 2.0, 2.0)
AISLE_X_WIDTH = 2.0
AISLE_Y_WIDTH = 2.0

def run_mapping():
    # Initialize empty environment
    env = WarehouseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=0,  # No drones needed for mapping
        physics=Physics.PYB,
        gui=True,
        warehouse_dims=WAREHOUSE_DIMS,
        shelf_dims=SHELF_DIMS,
        aisle_x_width=AISLE_X_WIDTH,
        aisle_y_width=AISLE_Y_WIDTH
    )

    # Create and run mapper
    mapper = WarehouseMapper(
        warehouse_dims=WAREHOUSE_DIMS,
        resolution=0.1,
        client_id=env.CLIENT
    )

    # Perform mapping
    occupancy_grid = mapper.perform_mapping()
    
    # Process map to find shelves
    shelf_info = mapper.process_map()
    print("\nDetected Shelves:")
    for i, shelf in enumerate(shelf_info):
        print(f"Shelf {i}:")
        print(f"  Center: {shelf['center']}")
        print(f"  Dimensions: {shelf['dimensions']}")

    # Visualize and save
    mapper.visualize_map(save=True)
    mapper.save_map()

    env.close()

if __name__ == "__main__":
    run_mapping()