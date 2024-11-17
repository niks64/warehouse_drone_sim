import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pybullet as p
import qrcode


class InventorySystem:
    """Manages warehouse inventory and package tracking"""
    
    def __init__(self):
        """Initialize inventory system"""
        self.inventory = {}  # Dict to track all items
        self.packages = {}  # Track package positions and info
        
    def add_item(self, 
                 item_id: str,
                 name: str,
                 quantity: int,
                 location: Tuple[float, float, float],
                 shelf_id: str
                 ) -> None:
        """Add a new item to inventory
        
        Args:
            item_id: Unique identifier for item
            name: Item name/description
            quantity: Number of items
            location: (x,y,z) position in warehouse
            shelf_id: ID of shelf containing item
        """
        self.inventory[item_id] = {
            'name': name,
            'quantity': quantity,
            'location': location,
            'shelf_id': shelf_id,
            'last_checked': None
        }
        
    def create_package(self,
                      item_id: str,
                      position: List[float],
                      size: List[float],
                      color: List[float],
                      client_id: int
                      ) -> int:
        """Create a package in PyBullet
        
        Args:
            item_id: Item ID for package
            position: [x,y,z] position
            size: [width, length, height] 
            color: RGBA color
            client_id: PyBullet client ID
            
        Returns:
            body_id: PyBullet body ID of package
        """
        # Create collision and visual shapes
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[s/2 for s in size],
            physicsClientId=client_id
        )
        
        visual_id = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[s/2 for s in size],
            rgbaColor=color,
            physicsClientId=client_id
        )
        
        # Create package body
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            physicsClientId=client_id
        )
        
        # Store package info
        self.packages[body_id] = {
            'item_id': item_id,
            'position': position
        }
        
        print(f"Created package {item_id} at position {position}")
        return body_id
    
    def scan_area(self, 
                  drone_pos: np.ndarray,
                  scan_radius: float = 0.5
                  ) -> List[str]:
        """Simulate scanning of packages in an area
        
        Args:
            drone_pos: Current drone position
            scan_radius: Radius within which to detect packages
            
        Returns:
            detected_items: List of detected item IDs
        """
        detected_items = []
        
        # Check packages within scan radius
        for body_id, package_info in self.packages.items():
            package_pos = np.array(package_info['position'])
            distance = np.linalg.norm(package_pos - drone_pos)
            
            if distance <= scan_radius:
                detected_items.append(package_info['item_id'])
                
        return detected_items
    
    def update_inventory(self, item_id: str, timestamp: float) -> None:
        """Update last checked timestamp for an item
        
        Args:
            item_id: Item ID that was scanned
            timestamp: Time of scan
        """
        if item_id in self.inventory:
            self.inventory[item_id]['last_checked'] = timestamp
            print(f"Updated inventory for {item_id} at time {timestamp}")
            
    def get_item_info(self, item_id: str) -> Dict:
        """Get information about an inventory item
        
        Args:
            item_id: Item ID to look up
            
        Returns:
            info: Dict of item information
        """
        return self.inventory.get(item_id, None)
    
    def get_unchecked_items(self, threshold: float) -> List[str]:
        """Get items not checked within time threshold
        
        Args:
            threshold: Time threshold in seconds
            
        Returns:
            items: List of item IDs needing check
        """
        unchecked = []
        for item_id, info in self.inventory.items():
            if info['last_checked'] is None or \
               (time.time() - info['last_checked']) > threshold:
                unchecked.append(item_id)
        return unchecked