import time

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

from inventory_logger import InventoryLogger
from warehouse_env import WarehouseAviary
from warehouse_mapper import WarehouseMapper
from warehouse_path_planner import WarehousePathPlanner

# Configuration
WAREHOUSE_DIMS = (10.0, 14.0, 3.0)
SHELF_DIMS = (0.5, 2.0, 2.0)
AISLE_X_WIDTH = 2.0
AISLE_Y_WIDTH = 2.0
INSPECTION_HEIGHT = 1.0

# Initialize starting position in bottom left corner
INITIAL_XYZS = np.array([[-4, -6, 1.0]])
INITIAL_RPYS = np.array([[0, 0, 0]])

def move_drone_to_position(env, ctrl, current_pos, target_pos, target_yaw=0):
    """Helper function to move drone to a specific position
    
    Args:
        env: WarehouseAviary environment
        ctrl: DSLPIDControl controller
        current_pos: Current drone position
        target_pos: Target position to reach
        target_yaw: Target yaw angle
    
    Returns:
        reached: Whether target was reached
    """
    # Compute control action
    state = env._getDroneStateVector(0)
    action, _, _ = ctrl.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP,
        state=state,
        target_pos=target_pos,
        target_rpy=np.array([0, 0, target_yaw])
    )
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action.reshape(1,4))
    
    # Check if position reached
    dist_to_target = np.linalg.norm(target_pos - current_pos)
    return dist_to_target < 0.2  # 20cm threshold

# Initialize environment
env = WarehouseAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    initial_xyzs=INITIAL_XYZS,
    initial_rpys=INITIAL_RPYS,
    physics=Physics.PYB,
    pyb_freq=240,
    ctrl_freq=240,
    gui=True,
    warehouse_dims=WAREHOUSE_DIMS,
    shelf_dims=SHELF_DIMS,
    aisle_x_width=AISLE_X_WIDTH,
    aisle_y_width=AISLE_Y_WIDTH
)

# Initialize controllers and logger
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
logger = InventoryLogger()
logger.log_initial_inventory(env.inventory_system.inventory)

# Initialize mapper
mapper = WarehouseMapper(
    initial_pos=INITIAL_XYZS[0],
    scan_height=2.0,
    grid_resolution=0.5,
    client_id=env.CLIENT
)

print("\n=== Starting Warehouse Mapping ===")

# First phase: Mapping
START = time.time()
print("\nPhase 1: Detecting warehouse boundaries...")
boundaries = mapper.detect_boundaries()
print(f"Warehouse boundaries detected: {boundaries}")

print("\nPhase 2: Creating occupancy grid...")
occupancy_grid = mapper.create_occupancy_grid()
print(f"Occupancy grid created with shape: {occupancy_grid.shape}")

print("\nPhase 3: Detecting aisles...")
main_aisles, cross_aisles = mapper.detect_aisles()
print(f"Detected {len(main_aisles)} main aisles and {len(cross_aisles)} cross aisles")

print("\nPhase 4: Generating inspection waypoints...")
waypoints = mapper.generate_inspection_waypoints(inspection_height=INSPECTION_HEIGHT)
inspection_path = mapper.optimize_inspection_route()
print(f"Generated {len(inspection_path)} inspection waypoints")

print("Inspection path:")
for i, waypoint in enumerate(inspection_path):
    print(f"Waypoint {i}: {waypoint}")

print("\n=== Starting Warehouse Inspection ===")
print(f"Initial inventory logged to {logger.log_file}")
print("Detection progress will be logged to", logger.detection_file)

# Inspection parameters
current_waypoint_idx = 0
time_at_waypoint = 0
min_time_at_waypoint = 2 * env.PYB_FREQ  # Minimum steps to spend at each waypoint
last_scan_time = time.time()
SCAN_INTERVAL = 1.0  # Scan for packages every second

# Main inspection loop
for i in range(60000):  # Can adjust this number based on needed duration
    state = env._getDroneStateVector(0)
    current_pos = state[0:3]
    
    # Get current target waypoint
    target_pos = np.array(inspection_path[current_waypoint_idx])
    
    # Compute required yaw to face the nearest shelf
    target_yaw = np.arctan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0])
    
    # Check if we've reached the waypoint
    dist_to_target = np.linalg.norm(target_pos - current_pos)
    if dist_to_target < 0.2:  # 20cm threshold
        time_at_waypoint += 1
        if time_at_waypoint >= min_time_at_waypoint:
            # Move to next waypoint
            current_waypoint_idx = (current_waypoint_idx + 1) % len(inspection_path)
            time_at_waypoint = 0
            print(f"\nMoving to waypoint {current_waypoint_idx}/{len(inspection_path)}")
    
    # Scan for packages periodically
    current_time = time.time()
    if current_time - last_scan_time > SCAN_INTERVAL:
        detections = env.process_drone_view(0)
        for det in detections:
            logger.log_detection(
                item_id=det['item_id'],
                info=det['info'],
                drone_pos=current_pos,
                timestamp=current_time
            )
        last_scan_time = current_time
    
    # Compute control action
    action, _, _ = ctrl.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP,
        state=state,
        target_pos=target_pos,
        target_rpy=np.array([0, 0, target_yaw])
    )
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action.reshape(1,4))
    
    # Render environment occasionally to reduce output
    if i % 240 == 0:
        env.render()
        
        # Print progress
        elapsed_time = time.time() - START
        progress = (current_waypoint_idx / len(inspection_path)) * 100
        print(f"\rInspection Progress: {progress:.1f}% | "
              f"Time: {elapsed_time:.1f}s | "
              f"Current Waypoint: {current_waypoint_idx}/{len(inspection_path)}", 
              end="")
    
    # Sync to real time
    sync(i, START, env.CTRL_TIMESTEP)
    
    # Check if we've completed the inspection
    if current_waypoint_idx == len(inspection_path) - 1 and time_at_waypoint >= min_time_at_waypoint:
        print("\n\nInspection complete!")
        break

# Print final statistics
stats = logger.get_detection_stats()
print("\n=== Inspection Complete ===")
print(f"Total items detected: {stats['total_detected']}")
print("\nDetailed logs written to:")
print(f"Initial inventory: {logger.log_file}")
print(f"Detection log: {logger.detection_file}")

env.close()