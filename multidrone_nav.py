import pickle
import time

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

from inventory_logger import InventoryLogger
from utils import capture_frame, setup_camera_and_recorder
from warehouse_env import WarehouseAviary
from warehouse_mapper import WarehouseMapper

# Warehouse configuration
WAREHOUSE_DIMS = (10.0, 14.0, 3.0)
SHELF_DIMS = (0.5, 2.0, 2.0)
AISLE_X_WIDTH = 2.0
AISLE_Y_WIDTH = 2.0
DETECTION_RANGE = 1.0
NUM_DRONES = 3

START_OFFSET = 0.5  # Distance between drones
INITIAL_XYZS = np.array([
    [-4 + i * START_OFFSET, -6, 1.0] for i in range(NUM_DRONES)
])
INITIAL_RPYS = np.array([[0, 0, 0] for _ in range(NUM_DRONES)])

MAP_FILE = "warehouse_map_20241210_012302.pkl"

def run_multi_drone_inspection():
    with open(MAP_FILE, 'rb') as f:
        map_data = pickle.load(f)

    mapper = WarehouseMapper.load_from_file(MAP_FILE)
    shelf_positions = mapper.extract_shelf_positions()
    # for i, shelf in enumerate(shelf_positions):
    #     pos = shelf['position']
    #     dims = shelf['dimensions']
    #     print(f"Shelf {i}: position={pos}, dimensions={dims}")
    map_data['shelf_positions'] = shelf_positions
    
    env = WarehouseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=INITIAL_XYZS,
        initial_rpys=INITIAL_RPYS,
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=240,
        gui=False,
        warehouse_dims=WAREHOUSE_DIMS,
        shelf_dims=SHELF_DIMS,
        aisle_x_width=AISLE_X_WIDTH,
        aisle_y_width=AISLE_Y_WIDTH,
        detection_range=DETECTION_RANGE,
        map_data=map_data
    )

    logger = InventoryLogger()
    logger.log_initial_inventory(env.inventory_system.inventory)

    start_positions = [pos for pos in INITIAL_XYZS]
    inspection_paths = env.path_planner.generate_inspection_paths(start_positions)

    for i in range(NUM_DRONES):
        print(f"Drone {i} path: {inspection_paths[i]}")

    controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(NUM_DRONES)]
    for ctrl in controllers:
        ctrl.P_COEFF_FOR = ctrl.P_COEFF_FOR * 0.5
        ctrl.P_COEFF_TOR = ctrl.P_COEFF_TOR * 0.5

    waypoint_indices = [0 for _ in range(NUM_DRONES)]
    time_at_waypoint = [0 for _ in range(NUM_DRONES)]
    waypoint_thresh = 0.2
    min_time_at_waypoint = 0

    print(f"\nStarting inspection with {NUM_DRONES} drones")
    for i in range(NUM_DRONES):
        print(f"Drone {i} path length: {len(inspection_paths[i])} waypoints")

    # VIDEO_WIDTH = 640
    # VIDEO_HEIGHT = 480
    # video_writer = setup_camera_and_recorder(env, VIDEO_WIDTH, VIDEO_HEIGHT)    

    START = time.time()
    last_scan_time = time.time()
    SCAN_INTERVAL = 1.0

    try:
        action = np.zeros((NUM_DRONES, 4))
        for i in range(60000):
            for drone_id in range(NUM_DRONES):
                state = env._getDroneStateVector(drone_id)
                pos = state[0:3]
                
                target_pos = np.array(inspection_paths[drone_id][waypoint_indices[drone_id]])
                target_yaw = env.path_planner.get_required_yaw(pos, target_pos)
                target_rpy = np.array([0, 0, target_yaw])
                
                dist_to_target = np.linalg.norm(target_pos - pos)
                
                if dist_to_target < waypoint_thresh:
                    time_at_waypoint[drone_id] += 1
                    if time_at_waypoint[drone_id] >= min_time_at_waypoint:
                        next_pos, next_idx = env.path_planner.get_next_waypoint(
                            drone_id=drone_id,
                            current_pos=pos,
                            current_waypoint_idx=waypoint_indices[drone_id]
                        )
                        waypoint_indices[drone_id] = next_idx
                        time_at_waypoint[drone_id] = 0
                
                act, _, _ = controllers[drone_id].computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=state,
                    target_pos=target_pos,
                    target_rpy=target_rpy
                )
                action[drone_id] = act
            
            if time.time() - last_scan_time > SCAN_INTERVAL:
                for drone_id in range(NUM_DRONES):
                    detections = env.process_drone_view(drone_id)
                    for det in detections:
                        logger.log_detection(
                            item_id=det['item_id'],
                            info=det['info'],
                            drone_pos=env._getDroneStateVector(drone_id)[0:3],
                            timestamp=time.time()
                        )
                last_scan_time = time.time()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # if i % 40 == 0:
            #     capture_frame(env, video_writer, VIDEO_WIDTH, VIDEO_HEIGHT)

            if i % 240 == 0:
                env.render()
            
            sync(i, START, env.CTRL_TIMESTEP)

            # Check if all drones have completed their paths
            if all([waypoint_indices[i] == len(inspection_paths[i]) - 1 for i in range(NUM_DRONES)]):
                break

            # Check if items have been detected
            stats = logger.get_detection_stats()
            print(f"\r[INFO] Total items detected: {stats['total_detected']}/{len(env.inventory_system.inventory)}", end="")
            if stats['total_detected'] == len(env.inventory_system.inventory):
                print("\n[INFO] All items detected!")
                break
        
    except KeyboardInterrupt:
        pass
    finally:
        # video_writer.release()
        env.close()

if __name__ == "__main__":
    run_multi_drone_inspection()