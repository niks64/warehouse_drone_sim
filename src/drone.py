import pybullet as p
import numpy as np
import math

class Drone:
    def __init__(self, client, start_pos=[0, 0, 1]):
        self.client = client
        
        # Create a simple drone shape (we'll replace this with a proper URDF later)
        self.base_radius = 0.2
        self.base_height = 0.1
        
        # Create collision shape
        self.collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.base_radius,
            height=self.base_height
        )
        
        # Create visual shape
        self.visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.base_radius,
            length=self.base_height,
            rgbaColor=[0.7, 0.7, 0.7, 1]
        )
        
        # Create drone body
        self.drone_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=self.collision_shape,
            baseVisualShapeIndex=self.visual_shape,
            basePosition=start_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Set damping for stability
        p.changeDynamics(
            self.drone_id,
            -1,
            linearDamping=0.9,
            angularDamping=0.9
        )
        
        # Parameters for control
        self.max_force = 10
        self.target_pos = start_pos
        self.target_orn = [0, 0, 0]
        
    def set_target_position(self, position):
        """Set target position for the drone"""
        self.target_pos = position
    
    def apply_control(self):
        """Apply basic position control"""
        current_pos, current_orn = p.getBasePositionAndOrientation(self.drone_id)
        current_orn = p.getEulerFromQuaternion(current_orn)
        
        # Simple P controller
        kp = 10.0  # Position gain
        kd = 1.0   # Damping gain
        
        # Get current velocity
        current_vel, _ = p.getBaseVelocity(self.drone_id)
        
        # Calculate force based on position error and velocity damping
        force = [0, 0, 0]
        for i in range(3):
            pos_error = self.target_pos[i] - current_pos[i]
            vel_damping = -current_vel[i]
            force[i] = kp * pos_error + kd * vel_damping
        
        # Apply force at center of mass
        p.applyExternalForce(
            self.drone_id,
            -1,  # -1 for base link
            force,
            current_pos,
            p.WORLD_FRAME
        )
        
        # Gravity compensation
        p.applyExternalForce(
            self.drone_id,
            -1,
            [0, 0, 9.81],  # Compensate for gravity
            current_pos,
            p.WORLD_FRAME
        )
    
    def get_position(self):
        """Get current drone position"""
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        return pos
    
    def get_orientation(self):
        """Get current drone orientation"""
        _, orn = p.getBasePositionAndOrientation(self.drone_id)
        return p.getEulerFromQuaternion(orn)