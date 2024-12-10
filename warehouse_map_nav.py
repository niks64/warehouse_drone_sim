import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
from gymnasium import spaces

from utils import capture_frame, setup_camera_and_recorder
from warehouse_env import WarehouseAviary
from warehouse_mapper import WarehouseMapper


def run_warehouse_mapping(
    grid_resolution=0.05,
    grid_margin=1.0,
    gui=True,
    plot=True,
    record_video=False
):

    WAREHOUSE_DIMS = (10.0, 14.0, 3.0)
    length, width, height = WAREHOUSE_DIMS
    SHELF_DIMS = (0.5, 2.0, 2.0)
    AISLE_X_WIDTH = 2.0
    AISLE_Y_WIDTH = 2.0
    DETECTION_RANGE = 0.75
    initial_drone_pos = np.array([[-length/2 + grid_margin, -width/2 + grid_margin, height]])
    
    # Initialize warehouse 
    env = WarehouseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        neighbourhood_radius=np.inf,
        warehouse_dims=WAREHOUSE_DIMS,
        shelf_dims=SHELF_DIMS,
        aisle_x_width=AISLE_X_WIDTH,
        aisle_y_width=AISLE_Y_WIDTH,
        detection_range=DETECTION_RANGE,
        initial_xyzs=initial_drone_pos,
        initial_rpys=np.array([[0, 0, 0]]),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=240,
        gui=gui,
        record=False,
        obstacles=False, 
        user_debug_gui=True
    )
    
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=12.0,
            cameraYaw=0,
            cameraPitch=-60,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=env.CLIENT
        )
    
    # Initialize mapper
    mapper = WarehouseMapper(
        warehouse_dims=WAREHOUSE_DIMS,
        safety_margin=grid_margin,
        grid_resolution=grid_resolution,
        mapping_height=height,
        shelf_height_threshold=1.5
    )
    
    # Generate mapping path
    mapping_path = mapper.generate_mapping_path()
    print(f"Generated mapping path with {len(mapping_path)} waypoints")
    
    # Initialize controller
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    
    # Setup video recording if enabled
    if record_video:
        VIDEO_WIDTH = 640
        VIDEO_HEIGHT = 480
        video_writer = setup_camera_and_recorder(env, VIDEO_WIDTH, VIDEO_HEIGHT)
    else:
        video_writer = None
    
    # Initialize path tracking variables
    current_waypoint_idx = 0
    time_at_waypoint = 0
    waypoint_threshold = 0.1  # Distance threshold to consider waypoint reached
    min_time_at_waypoint = 10  # Minimum steps to stay at waypoint for stable raycast
    
    action = np.zeros((1, 4))
    START = time.time()
    
    print("\nStarting warehouse mapping...")
    
    try:
        for i in range(600000):  # Long enough to complete mapping
            # Get current state
            state = env._getDroneStateVector(0)
            pos = state[0:3]
            
            # Get current target waypoint
            target_pos = np.array(mapping_path[current_waypoint_idx])
            
            # Calculate yaw to face forward along path
            if current_waypoint_idx < len(mapping_path) - 1:
                next_pos = np.array(mapping_path[current_waypoint_idx + 1])
                path_direction = next_pos - target_pos
                target_yaw = np.arctan2(path_direction[1], path_direction[0])
            else:
                target_yaw = 0
            
            target_rpy = np.array([0, 0, target_yaw])
            
            # Check if reached current waypoint
            dist_to_target = np.linalg.norm(target_pos - pos)
            if dist_to_target < waypoint_threshold:
                time_at_waypoint += 1
                
                # Perform raycast after staying at waypoint
                if time_at_waypoint >= min_time_at_waypoint:
                    mapper.process_raycast(pos, env.CLIENT)
                    
                    # Move to next waypoint
                    if current_waypoint_idx < len(mapping_path) - 1:
                        current_waypoint_idx += 1
                        time_at_waypoint = 0
                    else:
                        print("\nMapping complete!")
                        break
            
            # Compute control action
            act, _, _ = ctrl.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=state,
                target_pos=target_pos,
                target_rpy=target_rpy
            )
            action[0] = act
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record video frame
            if record_video and i % 40 == 0:  # 6 fps at 240Hz
                capture_frame(env, video_writer, VIDEO_WIDTH, VIDEO_HEIGHT)
            
            if i % 240 == 0:  # Update GUI
                env.render()
                if gui:  # Draw current raycast line
                    p.addUserDebugLine(
                        pos,
                        [pos[0], pos[1], 0],
                        [1, 0, 0],  # Red color
                        1,  # Line width
                        lifeTime=0.5,  # Short lifetime
                        physicsClientId=env.CLIENT
                    )
            
            # Maintain desired freq
            sync(i, START, env.CTRL_TIMESTEP)
            
        # Save mapping results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mapper.save_occupancy_grid(f"warehouse_map_{timestamp}.pkl")
        
        # Plot occupancy grid
        if plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(mapper.occupancy_grid.T, origin='lower', cmap='gray')
            plt.colorbar(label='Occupancy (0: empty, 1: shelf)')
            plt.title('Warehouse Occupancy Grid')
            plt.xlabel('Grid X')
            plt.ylabel('Grid Y')
            plt.savefig(f"occupancy_grid_{timestamp}.png")
            if gui:
                plt.show()
        
        # Extract and print shelf positions
        shelf_positions = mapper.extract_shelf_positions()
        print(f"\nDetected {len(shelf_positions)} shelf positions:")
        for i, shelf in enumerate(shelf_positions):
            pos = shelf['position']
            dims = shelf['dimensions']
            print(f"Shelf {i}: position={pos}, dimensions={dims}")
            
    except KeyboardInterrupt:
        pass
    finally:
        if record_video and video_writer is not None:
            video_writer.release()
        env.close()
        
    return mapper

if __name__ == "__main__":
    mapper = run_warehouse_mapping(
        grid_resolution=0.2,
        gui=False,
        plot=True,
        record_video=False
    )