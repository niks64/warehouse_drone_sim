import time

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

from inventory_logger import InventoryLogger
from warehouse_env import WarehouseAviary
from warehouse_path_planner import WarehousePathPlanner

INITIAL_XYZS = np.array([[-4, -6, 1.0]]) # Start in bottom left corner
INITIAL_RPYS = np.array([[0, 0, 0]])

WAREHOUSE_DIMS = (10.0, 14.0, 3.0)
SHELF_DIMS = (0.5, 2.0, 2.0)

AISLE_X_WIDTH = 2.0
AISLE_Y_WIDTH = 2.0

INSPECTION_HEIGHT = 1.0
SAFETY_MARGIN = 0.75
DETECTION_RANGE = 1.0

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
    aisle_y_width=AISLE_Y_WIDTH,
    detection_range=DETECTION_RANGE
)

# Initialize inventory logger
logger = InventoryLogger()
logger.log_initial_inventory(env.inventory_system.inventory)

# Initialize path planner with warehouse dimensions
path_planner = WarehousePathPlanner(
    warehouse_dims=WAREHOUSE_DIMS,
    shelf_dims=SHELF_DIMS,
    aisle_x_width=AISLE_X_WIDTH,
    aisle_y_width=AISLE_Y_WIDTH,
    inspection_height=INSPECTION_HEIGHT,
    safety_margin=SAFETY_MARGIN
)

# Initialize PID controller
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

# Flight parameters
current_waypoint_idx = 0
waypoint_thresh = 0.2  # Distance threshold to consider waypoint reached
time_at_waypoint = 0   # Time spent at current waypoint
min_time_at_waypoint = 2 * env.PYB_FREQ  # Minimum steps to spend at each waypoint

# Get full inspection path
inspection_path = path_planner._generate_inspection_points()

print(f"\nStarting inspection of {len(inspection_path)} waypoints...")
print(f"Inspection path: {inspection_path}")
print(f"Initial inventory logged to {logger.log_file}")
print("Detection progress will be logged to", logger.detection_file)
print("\nBeginning warehouse inspection...\n")

START = time.time()
last_scan_time = time.time()
SCAN_INTERVAL = 1.0  # Scan for packages every second

# Main control loop
for i in range(60000):
    state = env._getDroneStateVector(0)
    pos = state[0:3]
    
    # Current target
    target_pos = np.array(inspection_path[current_waypoint_idx])
    
    # Get required yaw to face the shelf
    target_yaw = path_planner.get_required_yaw(pos, target_pos)
    target_rpy = np.array([0, 0, target_yaw])
    
    # Check if we've reached the waypoint
    dist_to_target = np.linalg.norm(target_pos - pos)
    
    if dist_to_target < waypoint_thresh:
        time_at_waypoint += 1
        if time_at_waypoint >= min_time_at_waypoint:
            # Get next waypoint
            next_pos, next_idx = path_planner.get_next_waypoint(pos, current_waypoint_idx)
            current_waypoint_idx = next_idx
            time_at_waypoint = 0
    
    # Scan for packages periodically
    current_time = time.time()
    if current_time - last_scan_time > SCAN_INTERVAL:
        detections = env.process_drone_view(0)
        for det in detections:
            logger.log_detection(
                item_id=det['item_id'],
                info=det['info'],
                drone_pos=pos,
                timestamp=current_time
            )
        last_scan_time = current_time
    
    # Compute control action using PID
    action, _, _ = ctrl.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP,
        state=state,
        target_pos=target_pos,
        target_rpy=target_rpy
    )
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action.reshape(1,4))
    
    # Render but suppress output
    if i % 240 == 0:  # Only render every second
        env.render()
    
    # Sync to real time
    sync(i, START, env.CTRL_TIMESTEP)

# Print final statistics
stats = logger.get_detection_stats()
print("\n=== Inspection Complete ===")
print(f"Total items detected: {stats['total_detected']}")
print("\nDetailed logs written to:")
print(f"Initial inventory: {logger.log_file}")
print(f"Detection log: {logger.detection_file}")

env.close()