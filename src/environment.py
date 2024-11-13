import pybullet as p
import pybullet_data
import numpy as np
import time
import math

class WarehouseEnv:
    def __init__(self, gui=True):
        self.gui = gui
        self.physics_client = None
        
    def setup(self):
        # Connect to physics server
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Configure debug visualizer
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        
        # Set up physics properties
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create basic warehouse walls (we'll replace these with proper models later)
        self.create_warehouse_boundaries()
        
    def create_warehouse_boundaries(self):
        """Create simple boundary walls for the warehouse"""
        wall_height = 3
        wall_thickness = 0.1
        room_length = 10
        room_width = 10
        
        # Create collision shape for walls
        wall_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[room_length/2, wall_thickness/2, wall_height/2]
        )
        
        # Create walls
        wall_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[room_length/2, wall_thickness/2, wall_height/2],
            rgbaColor=[0.7, 0.7, 0.7, 1]
        )
        
        # Create four walls
        self.walls = []
        wall_positions = [
            [0, room_width/2, wall_height/2],  # Front wall
            [0, -room_width/2, wall_height/2], # Back wall
            [room_length/2, 0, wall_height/2], # Right wall
            [-room_length/2, 0, wall_height/2] # Left wall
        ]
        wall_orientations = [
            p.getQuaternionFromEuler([0, 0, 0]),
            p.getQuaternionFromEuler([0, 0, 0]),
            p.getQuaternionFromEuler([0, 0, math.pi/2]),
            p.getQuaternionFromEuler([0, 0, math.pi/2])
        ]
        
        for pos, orn in zip(wall_positions, wall_orientations):
            wall = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_shape,
                baseVisualShapeIndex=wall_visual,
                basePosition=pos,
                baseOrientation=orn
            )
            self.walls.append(wall)
    
    def reset(self):
        """Reset the environment"""
        p.resetSimulation()
        self.setup()
    
    def close(self):
        """Disconnect from physics server"""
        if self.physics_client is not None:
            p.disconnect()