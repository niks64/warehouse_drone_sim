import numpy as np
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gymnasium import spaces
import pybullet as p

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
                 obstacles=True,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        """Initialize warehouse environment.
        
        Args:
            standard args from BaseAviary plus warehouse-specific params
        """
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
        # Warehouse dimensions
        self.WAREHOUSE_WIDTH = 10.0
        self.WAREHOUSE_LENGTH = 10.0
        self.WAREHOUSE_HEIGHT = 3.0
        self.SHELF_WIDTH = 0.5
        self.AISLE_WIDTH = 1.5
        
        # Create warehouse structure
        self._create_warehouse()

    def _create_warehouse(self):
        """Create warehouse structure with shelves and aisles."""
        # Floor
        p.loadURDF("plane.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT)
        
        # Create walls
        self._create_box([0, self.WAREHOUSE_LENGTH/2, self.WAREHOUSE_HEIGHT/2],  # Back wall
                        [self.WAREHOUSE_WIDTH, 0.1, self.WAREHOUSE_HEIGHT])
        self._create_box([0, -self.WAREHOUSE_LENGTH/2, self.WAREHOUSE_HEIGHT/2], # Front wall
                        [self.WAREHOUSE_WIDTH, 0.1, self.WAREHOUSE_HEIGHT])
        self._create_box([self.WAREHOUSE_WIDTH/2, 0, self.WAREHOUSE_HEIGHT/2],   # Right wall
                        [0.1, self.WAREHOUSE_LENGTH, self.WAREHOUSE_HEIGHT])
        self._create_box([-self.WAREHOUSE_WIDTH/2, 0, self.WAREHOUSE_HEIGHT/2],  # Left wall
                        [0.1, self.WAREHOUSE_LENGTH, self.WAREHOUSE_HEIGHT])
        
        # Create shelves and packages
        shelf_positions = self._generate_shelf_positions()
        for pos in shelf_positions:
            self._create_box(pos, [self.SHELF_WIDTH, 2.0, 2.0])
            # Add packages on each shelf
            self._add_packages_to_shelf(pos)

    def _add_packages_to_shelf(self, shelf_pos):
        """Add packages to a specific shelf."""
        # Add packages at different positions on the shelf
        package_positions = [
            [0.0, -0.8],  # Left side
            [0.0, 0.0],   # Center
            [0.0, 0.8],   # Right side
        ]
        colors = [
            [1, 0, 0, 1],     # Red
            [0, 1, 0, 1],     # Green
            [0, 0, 1, 1],     # Blue
        ]
        
        for (x_offset, y_offset), color in zip(package_positions, colors):
            size = [0.5, 0.4, 0.4]  # Slightly larger packages
            collision_box = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[s/2 for s in size],
                physicsClientId=self.CLIENT
            )
            visual_box = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[s/2 for s in size],
                rgbaColor=color,
                physicsClientId=self.CLIENT
            )
            # Position package on shelf
            package_pos = [
                shelf_pos[0] + x_offset,  # Offset from shelf center
                shelf_pos[1] + y_offset,  # Different positions along shelf
                shelf_pos[2] + 0.3        # Slightly above shelf
            ]
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_box,
                baseVisualShapeIndex=visual_box,
                basePosition=package_pos,
                physicsClientId=self.CLIENT
            )

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
        """Generate positions for warehouse shelves in a grid pattern."""
        positions = []
        x_start = -self.WAREHOUSE_WIDTH/2 + 2.0
        y_start = -self.WAREHOUSE_LENGTH/2 + 2.0
        
        for x in np.arange(x_start, self.WAREHOUSE_WIDTH/2-1, self.AISLE_WIDTH + self.SHELF_WIDTH):
            for y in np.arange(y_start, self.WAREHOUSE_LENGTH/2-1, 3.0):
                positions.append([x, y, 1.0])
        
        return positions

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