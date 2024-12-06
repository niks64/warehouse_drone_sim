import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gymnasium import spaces

from drone_vision_system import DroneVisionSystem
from inventory_system import InventorySystem
from warehouse_path_planner import WarehousePathPlanner


class WarehouseAviary(BaseAviary):
    """Custom warehouse environment for drone fleet inventory management."""
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 warehouse_dims=(10.0, 10.0, 3.0),
                 shelf_dims=(0.5, 2.0, 2.0),
                 aisle_x_width=1.5,
                 aisle_y_width=1.0,
                 detection_range=1.0,
                 qr_folder='qr_codes'
                 ):
        """Initialize warehouse environment."""
        super().__init__(drone_model=drone_model,
                        num_drones=num_drones,
                        neighbourhood_radius=neighbourhood_radius,
                        initial_xyzs=initial_xyzs,
                        initial_rpys=initial_rpys,
                        physics=physics,
                        pyb_freq=pyb_freq,
                        ctrl_freq=ctrl_freq,
                        gui=gui,
                        record=record,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui,
                        output_folder=output_folder
                        )
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=12.0,
                cameraYaw=0, 
                cameraPitch=-60,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.CLIENT
            )
            
            # Enable shadow mapping for better QR code visualization
            p.configureDebugVisualizer(
                p.COV_ENABLE_SHADOWS,
                1,
                physicsClientId=self.CLIENT
            )
        
        # Initialize warehouse dimensions
        self.WAREHOUSE_WIDTH = warehouse_dims[0]
        self.WAREHOUSE_LENGTH = warehouse_dims[1]
        self.WAREHOUSE_HEIGHT = warehouse_dims[2]
        self.SHELF_WIDTH = shelf_dims[0]
        self.SHELF_LENGTH = shelf_dims[1]
        self.SHELF_HEIGHT = shelf_dims[2]
        self.AISLE_X_WIDTH = aisle_x_width
        self.AISLE_Y_WIDTH = aisle_y_width
        self.gui = gui

        # Initialize systems
        self.inventory_system = InventorySystem(qr_folder=qr_folder)
        self.vision_system = DroneVisionSystem(
            detection_range=detection_range,
            image_width=640,  # Increased resolution for better QR detection
            image_height=480
        )

        self.path_planner = WarehousePathPlanner(
            warehouse_dims=warehouse_dims,
            shelf_dims=shelf_dims,
            aisle_x_width=aisle_x_width,
            aisle_y_width=aisle_y_width
        )
        self.path_planner.package_positions = []
        
        # Create warehouse structure
        self._create_warehouse(shelf_dims)
        
        # Store drone IDs in inventory system
        self.inventory_system.drone_id = self.DRONE_IDS[0]

    def _create_warehouse(self, shelf_dims):
        """Create warehouse structure with shelves and aisles."""
        
        # Floor
        p.loadURDF(
            "plane.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
        )
        
        # Create walls
        self._create_box(
            [0, self.WAREHOUSE_LENGTH/2, self.WAREHOUSE_HEIGHT/2],  # Back wall
            [self.WAREHOUSE_WIDTH, 0.1, self.WAREHOUSE_HEIGHT]
        )

        self._create_box(
            [0, -self.WAREHOUSE_LENGTH/2, self.WAREHOUSE_HEIGHT/2], # Front wall
            [self.WAREHOUSE_WIDTH, 0.1, self.WAREHOUSE_HEIGHT]
        )

        self._create_box(
            [self.WAREHOUSE_WIDTH/2, 0, self.WAREHOUSE_HEIGHT/2],   # Right wall
            [0.1, self.WAREHOUSE_LENGTH, self.WAREHOUSE_HEIGHT]    
        )

        self._create_box(
            [-self.WAREHOUSE_WIDTH/2, 0, self.WAREHOUSE_HEIGHT/2],  # Left wall
            [0.1, self.WAREHOUSE_LENGTH, self.WAREHOUSE_HEIGHT]
        )

        shelf_positions = self._generate_shelf_positions()
        for i, pos in enumerate(shelf_positions):
            self._create_box(pos[0], shelf_dims)
            self._add_packages_to_shelf(pos[0], pos[1])

    def _add_packages_to_shelf(self, shelf_pos, shelf_id):
        """Add packages to a shelf with QR codes
        
        Creates three rows of packages, with three packages per row.
        Each package has a QR code on its front face.
        """
        # Package positions relative to shelf center
        package_positions = [
            # Bottom row (row 1)
            [0.0, -0.8, 0.3],  # Left 
            [0.0, 0.0, 0.3],   # Center
            [0.0, 0.8, 0.3],   # Right
            
            # Middle row (row 2)
            [0.0, -0.8, 0.9],  # Left
            [0.0, 0.0, 0.9],   # Center
            [0.0, 0.8, 0.9],   # Right
            
            # Top row (row 3)
            [0.0, -0.8, 1.5],  # Left
            [0.0, 0.0, 1.5],   # Center
            [0.0, 0.8, 1.5],   # Right
        ]
        
        # Base colors for packages
        colors = [
            [1, 0, 0, 1],     # Red
            [0, 1, 0, 1],     # Green
            [0, 0, 1, 1],     # Blue
            [1, 1, 0, 1],     # Yellow
            [1, 0, 1, 1],     # Magenta
            [0, 1, 1, 1],     # Cyan
            [0.5, 0, 0, 1],   # Dark red
            [0, 0.5, 0, 1],   # Dark green
            [0, 0, 0.5, 1],   # Dark blue
        ]
        
        # Create packages with QR codes
        for i, ((x_offset, y_offset, z_offset), color) in enumerate(zip(package_positions, colors)):
            row_num = (i // 3) + 1
            pos_num = i % 3
            position_name = ["left", "center", "right"][pos_num]
            
            # Create unique item ID
            item_id = f"shelf_{shelf_id}_row_{row_num}_pos_{pos_num}"
            
            # Calculate absolute position
            package_pos = [
                shelf_pos[0] + x_offset,
                shelf_pos[1] + y_offset,
                z_offset
            ]
            
            # Add position to path planner
            self.path_planner.package_positions.append((item_id, package_pos))
            
            # Add item to inventory with detailed location information
            self.inventory_system.add_item(
                item_id=item_id,
                name=f"Package on shelf {shelf_id}, row {row_num} ({z_offset:.1f}m height), {position_name} position",
                quantity=1,
                location={
                    'position': package_pos,
                    'shelf_id': shelf_id,
                    'row': row_num,
                    'height': z_offset,
                    'position': position_name
                },
                shelf_id=shelf_id
            )
            
            # Create package with QR code texture
            size = [0.5, 0.3, 0.3]  # Smaller packages for better visibility
            box_color = [0.6, 0.4, 0.2, 1]  # Brown cardboard-like color
            self.inventory_system.create_package(
                item_id=item_id,
                position=package_pos,
                size=size,
                color=box_color,
                client_id=self.CLIENT
            )

    def process_drone_view(self, drone_idx: int) -> List[Dict]:
        """Process drone's view for QR code detection"""
        state = self._getDroneStateVector(drone_idx)
        pos = state[0:3]
        
        # Get drone's camera view and detect QR codes
        rgb, _, _ = self.vision_system.get_camera_image(
            drone_pos=pos,
            drone_orientation=state[3:7],
            client_id=self.CLIENT
        )
        
        # Detect and process QR codes
        detections = self.vision_system.detect_packages(
            drone_pos=pos,
            inventory_system=self.inventory_system,
            client_id=self.CLIENT
        )
        
        # Update inventory for detected items
        current_time = time.time()
        for detection in detections:
            self.inventory_system.update_inventory(
                item_id=detection['item_id'],
                timestamp=current_time
            )
            
        # Optionally save visualization for debugging
        if self.gui and len(detections) > 0:
            vis_img = self.vision_system.visualize_detections(rgb, detections)
            debug_path = os.path.join(self.inventory_system.qr_folder, 'debug')
            os.makedirs(debug_path, exist_ok=True)
            cv2.imwrite(
                os.path.join(debug_path, f"detection_{int(current_time)}.jpg"),
                cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            )
            
        return detections

    def _create_box(self, position, size):
        """Create a box obstacle in the environment.
        
        Args:
            position (list): [x, y, z] position of box center
            size (list): [width, length, height] of box
        """
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size[0]/2, size[1]/2, size[2]/2],
            physicsClientId=self.CLIENT
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size[0]/2, size[1]/2, size[2]/2],
            rgbaColor=[0.7, 0.7, 0.7, 1],
            physicsClientId=self.CLIENT
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.CLIENT
        )

    def _generate_shelf_positions(self):
        """Generate shelf positions and IDs."""
        positions = []
        shelf_id = 0
        x_start = -self.WAREHOUSE_WIDTH/2 + self.AISLE_X_WIDTH
        y_start = -self.WAREHOUSE_LENGTH/2 + self.AISLE_Y_WIDTH
        
        for x in np.arange(
            x_start, self.WAREHOUSE_WIDTH/2 - self.AISLE_X_WIDTH - self.SHELF_WIDTH,
            self.AISLE_X_WIDTH + self.SHELF_WIDTH
        ):
            for y in np.arange(
                y_start, self.WAREHOUSE_LENGTH/2,
                self.AISLE_Y_WIDTH + self.SHELF_LENGTH
            ):
                position = [
                    x + self.SHELF_WIDTH/2,
                    y + self.SHELF_LENGTH/2,
                    self.SHELF_HEIGHT/2
                ]
                positions.append((position, f"shelf_{shelf_id}"))
                shelf_id += 1
        
        return positions
    
    def get_drone_camera_view(self, drone_idx: int):
        """Get camera view from specified drone."""
        state = self._getDroneStateVector(drone_idx)
        pos = state[0:3]
        quat = state[3:7]
        
        return self.vision_system.get_camera_image(
            drone_pos=pos,
            drone_orientation=quat,
            client_id=self.CLIENT
        )

    def _actionSpace(self):
        """Returns the action space of the environment."""
        #### Action vector ### [RPM_1, RPM_2, RPM_3, RPM_4]
        act_lower_bound = np.array([[0, 0, 0, 0] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(
            low=act_lower_bound,
            high=act_upper_bound,
            dtype=np.float32
        )

    def _observationSpace(self):
        """Returns the observation space of the environment."""
        #### Observation vector ### [X, Y, Z, Q1, Q2, Q3, Q4, R, P, Y, VX, VY, VZ, WX, WY, WZ, P0, P1, P2, P3]
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0, -1, -1, -1, -1, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf, np.inf, np.inf, 1, 1, 1, 1, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(
            low=obs_lower_bound,
            high=obs_upper_bound,
            dtype=np.float32
        )

    def _computeObs(self):
        """Return the current observation."""
        obs = np.zeros((self.NUM_DRONES, 20))
        for i in range(self.NUM_DRONES):
            obs[i] = self._getDroneStateVector(i)
        return obs

    def _preprocessAction(self, action):
        """Pre-process the action for feeding to the simulator."""
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _computeReward(self):
        """Compute reward (placeholder for now)."""
        return -1

    def _computeTerminated(self):
        """Compute terminated signal."""
        return False
        
    def _computeTruncated(self):
        """Compute truncated signal."""
        return False

    def _computeInfo(self):
        """Compute info dict."""
        return {"answer": 42}