"""Script demonstrating package inspection with manual navigation.
"""
import time

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

from warehouse_env import WarehouseAviary

# Environment parameters
WAREHOUSE_DIMS = (10.0, 14.0, 3.0)  # width, length, height
SHELF_DIMS = (0.5, 2.0, 2.0)        # width, length, height
AISLE_X_WIDTH = 2.0
AISLE_Y_WIDTH = 2.0
INITIAL_XYZS = np.array([[-4, -6, 1.0]])  # Start in bottom left corner
INITIAL_RPYS = np.array([[0, 0, 0]])

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
    detection_range=1.0
)

# Initialize controller
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

# Flight parameters
current_waypoint_idx = 0
waypoint_thresh = 0.2  # Distance threshold to consider waypoint reached
time_at_waypoint = 0   # Time spent at current waypoint
min_time_at_waypoint = 2 * env.CTRL_FREQ  # Minimum steps to spend at each waypoint

# Get waypoints from path planner
inspection_path = env.path_planner.generate_waypoints()
print(f"Inspection path: {inspection_path}")
action = np.zeros((1,4))

print(f"\nStarting inspection of {len(inspection_path)} waypoints...")

START = time.time()
last_scan_time = time.time()
SCAN_INTERVAL = 1.0  # Scan for packages every second
detected_packages = set()

try:
    for i in range(0, int(600*env.CTRL_FREQ)):  # Run for 600 seconds
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        
        state = env._getDroneStateVector(0)
        
        # Current target
        target_pos = inspection_path[current_waypoint_idx]
        
        # Get required yaw to face shelf
        target_yaw = env.path_planner.get_required_yaw(state[0:3], target_pos)
        target_rpy = np.array([0, 0, target_yaw])
        
        # Check if we've reached the waypoint
        dist_to_target = np.linalg.norm(target_pos - state[0:3])
        
        if dist_to_target < waypoint_thresh:
            time_at_waypoint += 1
            if time_at_waypoint >= min_time_at_waypoint:
                # Move to next waypoint
                current_waypoint_idx = (current_waypoint_idx + 1) % len(inspection_path)
                time_at_waypoint = 0
                print(f"Moving to waypoint {current_waypoint_idx}/{len(inspection_path)}")
        
        # Scan for packages periodically
        current_time = time.time()
        if current_time - last_scan_time > SCAN_INTERVAL:
            detections = env.process_drone_view(0)
            for det in detections:
                if det['item_id'] not in detected_packages:
                    detected_packages.add(det['item_id'])
                    print(f"\nDetected new package: {det['info']['name']}")
                    print(f"Location: {det['info']['location']}")
                    print(f"Total packages found: {len(detected_packages)}\n")
            last_scan_time = current_time

        # Compute control action
        action[0], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target_pos,
            target_rpy=target_rpy
        )

        #### Print drone info #####################################
        if i % env.CTRL_FREQ == 0:
            env.render()
            
            pos = state[0:3]
            print(f"Position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
            print(f"Target: [{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}]")
            print(f"Distance: {dist_to_target:.2f}")

        #### Sync the simulation ###################################
        sync(i, START, env.CTRL_TIMESTEP)
            
except KeyboardInterrupt:
    pass

finally:
    env.close()
    
print("\nFlight completed!")
print(f"Total packages detected: {len(detected_packages)}")