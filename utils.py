from datetime import datetime

import cv2
import numpy as np
import pybullet as p


# Add this function to test.py
def setup_camera_and_recorder(env, width=1920, height=1080):
    """Setup fixed camera view and video recorder
    
    Args:
        env: WarehouseAviary environment
        width: Video width
        height: Video height
    
    Returns:
        video_writer: OpenCV VideoWriter object
    """
    # Set camera parameters for a side view of warehouse
    camera_distance = 8
    camera_yaw = 0
    camera_pitch = -60
    camera_target = [-2.5, -2, 0]
    
    # Set the camera view in PyBullet
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=camera_target,
        physicsClientId=env.CLIENT
    )
    
    # Create video writer
    output_path = f'warehouse_inspection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = 30
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return video_writer

def capture_frame(env, video_writer, width=640, height=480):
    """Capture current frame and add to video"""
    # Get image from PyBullet
    _, _, rgb, _, _ = p.getCameraImage(
        width=width,
        height=height,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_NO_SEGMENTATION_MASK,
        shadow=1,  # Enable shadows
        lightDirection=[1, 1, 1],  # Better lighting
        physicsClientId=env.CLIENT
    )
    
    # Convert the RGB array to the correct format
    rgb_array = np.reshape(rgb, (height, width, 4))
    rgb_array = rgb_array.astype(np.uint8)
    
    # Convert RGBA to BGR
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
    video_writer.write(bgr)