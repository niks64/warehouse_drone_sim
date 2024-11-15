import time
import numpy as np
from warehouse_env import WarehouseAviary
from warehouse_path_planner import WarehousePathPlanner
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

INITIAL_XYZS = np.array([[-4, -6, 1.0]]) # Start in bottom left corner
INITIAL_RPYS = np.array([[0, 0, 0]])

WAREHOUSE_DIMS = (10.0, 14.0, 3.0)
SHELF_DIMS = (0.5, 2.0, 2.0)

AISLE_X_WIDTH = 2.0
AISLE_Y_WIDTH = 2.0

INSPECTION_HEIGHT = 1.0
SAFETY_MARGIN = 0.75

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
TAKEOFF_HEIGHT = 1.0
TAKEOFF_STEPS = 480  # 2 seconds at 240Hz
current_waypoint_idx = 0
waypoint_thresh = 0.2  # Distance threshold to consider waypoint reached
time_at_waypoint = 0   # Time spent at current waypoint
min_time_at_waypoint = 2 * env.PYB_FREQ  # Minimum steps to spend at each waypoint

# Get full inspection path
inspection_path = path_planner._generate_inspection_points()

print(f"Total inspection waypoints: {len(inspection_path)}")
for i, wp in enumerate(inspection_path):
    print(f"Waypoint {i}: {wp}")

# exit()

START = time.time()

# Main control loop
for i in range(60000):  # Increased steps for longer flight
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
            print(f"Moving to waypoint {current_waypoint_idx}: {next_pos}")
    
    # Compute control action using PID
    action, _, _ = ctrl.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP,
        state=state,
        target_pos=target_pos,
        target_rpy=target_rpy
    )
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action.reshape(1,4))
    env.render()

    # Print info occasionally
    if i % 240 == 0:
        print(f"Drone position: {pos}")
        print(f"Target position: {target_pos}")
        print(f"Distance to target: {dist_to_target:.2f}")
        print(f"Current yaw: {state[9]:.2f}, Target yaw: {target_yaw:.2f}")
    
    # Sync to real time
    sync(i, START, env.CTRL_TIMESTEP)

env.close()