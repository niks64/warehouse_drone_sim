import ast
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pybullet as p

from inventory_system import InventorySystem


class DroneVisionSystem:
    """Manages drone camera and QR code detection"""
    
    def __init__(self,
                 image_width: int = 640,
                 image_height: int = 480,
                 fov: float = 60.0,
                 near_val: float = 0.1,
                 far_val: float = 3.0,
                 detection_range: float = 1.0,
                 min_qr_size: int = 30  # Minimum QR code size in pixels
                 ):
        """Initialize vision system
        
        Args:
            image_width: Width of camera image
            image_height: Height of camera image
            fov: Field of view in degrees
            near_val: Near plane distance
            far_val: Far plane distance
            detection_range: Maximum detection range
            min_qr_size: Minimum QR code size for detection
        """
        self.width = image_width
        self.height = image_height
        self.fov = fov
        self.near = near_val
        self.far = far_val
        self.detection_range = detection_range
        self.min_qr_size = min_qr_size
        
        # Initialize QR code detector
        self.qr_detector = cv2.QRCodeDetector()
        
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
        """Get camera image from drone's perspective"""
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
            physicsClientId=client_id,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to correct format for OpenCV
        rgb = np.reshape(rgb, (height, width, 4))
        rgb = rgb.astype(np.uint8)
        rgb = rgb[:,:,:3]  # Remove alpha
        
        return rgb, depth, seg
    
    def detect_qr_codes(self, image: np.ndarray) -> List[Dict]:
        """Detect and decode QR codes in image
        
        Args:
            image: RGB image array
            
        Returns:
            detections: List of detected QR codes and their data
        """
        # Convert to grayscale for QR detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect and decode QR codes
        detected = []
        found, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(gray)
        
        if found:
            for info, corners in zip(decoded_info, points):
                try:
                    # Calculate QR code size in pixels
                    width = np.linalg.norm(corners[0] - corners[1])
                    height = np.linalg.norm(corners[1] - corners[2])
                    qr_size = min(width, height)
                    
                    # Only process if QR code is large enough
                    if qr_size >= self.min_qr_size:
                        # Convert string representation of dictionary back to dict
                        qr_data = ast.literal_eval(info)
                        
                        detected.append({
                            'data': qr_data,
                            'corners': corners,
                            'size': qr_size
                        })
                except (ValueError, SyntaxError) as e:
                    print(f"Error decoding QR data: {e}")
                    continue
                    
        return detected
    
    def detect_packages(self,
                       drone_pos: np.ndarray,
                       inventory_system: InventorySystem,
                       client_id: int
                       ) -> List[Dict]:
        """Detect packages using QR codes
        
        Args:
            drone_pos: Drone position
            inventory_system: InventorySystem instance
            client_id: PyBullet client ID
            
        Returns:
            detections: List of detected packages and their info
        """
        # Get camera image
        rgb, _, _ = self.get_camera_image(
            drone_pos=drone_pos,
            drone_orientation=p.getBasePositionAndOrientation(
                inventory_system.drone_id, 
                physicsClientId=client_id
            )[1],
            client_id=client_id
        )
        
        # Detect QR codes
        qr_detections = self.detect_qr_codes(rgb)
        
        results = []
        for detection in qr_detections:
            item_id = detection['data']['item_id']
            item_info = inventory_system.get_item_info(item_id)
            
            if item_info:
                results.append({
                    'item_id': item_id,
                    'info': item_info,
                    'qr_data': detection['data'],
                    'qr_corners': detection['corners']
                })
                
        return results
    
    def visualize_detections(self, 
                           image: np.ndarray,
                           detections: List[Dict]
                           ) -> np.ndarray:
        """Draw detected QR codes on image
        
        Args:
            image: RGB image array
            detections: List of QR code detections
            
        Returns:
            image: Annotated image array
        """
        vis_img = image.copy()
        
        for det in detections:
            corners = det['qr_corners']
            
            # Draw QR code boundary
            points = corners.astype(np.int32)
            cv2.polylines(vis_img, [points], True, (0, 255, 0), 2)
            
            # Draw item ID
            cv2.putText(vis_img,
                       det['data']['item_id'],
                       tuple(points[0].astype(np.int32)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8,
                       (255, 0, 0),
                       2)
            
        return vis_img