from typing import Dict, List, Tuple

import cv2
import numpy as np
import pybullet as p

from inventory_system import InventorySystem


class DroneVisionSystem:
    """Manages drone camera and package detection"""
    
    def __init__(self,
                 image_width: int = 640,
                 image_height: int = 480,
                 fov: float = 60.0,
                 near_val: float = 0.1,
                 far_val: float = 3.0,
                 detection_range: float = 1.0
                 ):
        """Initialize vision system
        
        Args:
            image_width: Width of camera image
            image_height: Height of camera image
            fov: Field of view in degrees
            near_val: Near plane distance
            far_val: Far plane distance
            detection_range: Maximum package detection range
        """
        self.width = image_width
        self.height = image_height
        self.fov = fov
        self.near = near_val
        self.far = far_val
        self.detection_range = detection_range
        
        # Camera matrices
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far
        )
        
    def get_camera_image(self,
                        drone_pos: np.ndarray,
                        drone_orientation: np.ndarray,
                        client_id: int
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get camera image from drone's perspective
        
        Args:
            drone_pos: [x,y,z] drone position
            drone_orientation: Quaternion orientation
            client_id: PyBullet client ID
            
        Returns:
            rgb: RGB image array
            depth: Depth image array 
            seg: Segmentation image array
        """
        # Calculate camera view matrix
        rot_matrix = p.getMatrixFromQuaternion(drone_orientation)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        # Camera faces forward along x-axis
        forward = rot_matrix.dot(np.array([1, 0, 0]))
        up = rot_matrix.dot(np.array([0, 0, 1]))
        
        target = drone_pos + forward
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=drone_pos,
            cameraTargetPosition=target,
            cameraUpVector=up,
            physicsClientId=client_id
        )
        
        # Get camera image
        width, height, rgb, depth, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.proj_matrix,
            physicsClientId=client_id
        )
        
        rgb = np.array(rgb).reshape(height, width, 4)[:,:,:3]  # Remove alpha
        depth = np.array(depth).reshape(height, width)
        seg = np.array(seg).reshape(height, width)
        
        return rgb, depth, seg
    
    def detect_packages(self,
                       drone_pos: np.ndarray,
                       inventory_system : InventorySystem,
                       client_id: int
                       ) -> List[Dict]:
        """Detect packages near drone
        
        Args:
            drone_pos: Drone position
            inventory_system: InventorySystem instance
            client_id: PyBullet client ID
            
        Returns:
            detections: List of detected packages and their info
        """
        detected_items = inventory_system.scan_area(
            drone_pos=drone_pos,
            scan_radius=self.detection_range
        )
        
        results = []
        for item_id in detected_items:
            item_info = inventory_system.get_item_info(item_id)
            if item_info:
                results.append({
                    'item_id': item_id,
                    'info': item_info
                })
                
        return results