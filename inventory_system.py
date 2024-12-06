import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pybullet as p
import qrcode
from PIL import Image


class InventorySystem:
    """Manages warehouse inventory and package tracking"""
    
    def __init__(self, qr_folder: str = "qr_codes"):
        """Initialize inventory system
        
        Args:
            qr_folder: Directory to store generated QR codes
        """
        self.inventory = {}  # Dict to track all items
        self.packages = {}  # Track package positions and info
        
        # Create QR code directory if it doesn't exist
        self.qr_folder = qr_folder
        if not os.path.exists(qr_folder):
            os.makedirs(qr_folder)
        
    def add_item(self, 
                 item_id: str,
                 name: str,
                 quantity: int,
                 location: dict,
                 shelf_id: str
                 ) -> None:
        """Add a new item to inventory
        
        Args:
            item_id: Unique identifier for item
            name: Item name/description
            quantity: Number of items
            location: Dictionary containing position and location metadata
            shelf_id: ID of shelf containing item
        """
        item_info = {
            'name': name,
            'quantity': quantity,
            'location': location,
            'shelf_id': shelf_id,
            'last_checked': None
        }
        
        # Generate QR code for item
        qr_path = self.generate_qr_code(item_id, item_info)
        item_info['qr_path'] = qr_path
        
        self.inventory[item_id] = item_info
        
    def create_package(self,
                    item_id: str,
                    position: List[float],
                    size: List[float],
                    color: List[float],
                    client_id: int
                    ) -> int:
        """Create a package in PyBullet with QR codes on each face
        
        Args:
            item_id: Item ID for package
            position: [x,y,z] position
            size: [width, length, height] 
            color: RGBA color
            client_id: PyBullet client ID
            
        Returns:
            body_id: PyBullet body ID of package
        """
        # Get QR code path
        qr_path = self.inventory[item_id]['qr_path']
        
        # Create collision shape
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[s/2 for s in size],
            physicsClientId=client_id
        )

        # Create visual shape first
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

        # Load and process QR code image
        qr_img = cv2.imread(qr_path)
        if qr_img is None:
            print(f"Failed to load QR code image: {qr_path}")
            return body_id

        # Create texture for each face
        texture_size = 128
        margin = 32  # Margin around QR code
        qr_size = texture_size - (2 * margin)  # QR code size with margins

        # Create base texture with brown color
        base_texture = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * np.array([255, 255, 255], dtype=np.uint8)  # Brown color

        # Resize QR code to fit within margins
        qr_img_resized = cv2.resize(qr_img, (qr_size, qr_size))

        # Place QR code in center of texture
        base_texture[margin:margin+qr_size, margin:margin+qr_size] = qr_img_resized

        # Save temporary texture
        temp_path = qr_path.replace('.png', '_temp.bmp')
        cv2.imwrite(temp_path, base_texture)
        
        # Load texture
        texture_id = p.loadTexture(temp_path, physicsClientId=client_id)
        
        # Clean up temporary file
        # if os.path.exists(temp_path):
        #     os.remove(temp_path)

        # Apply texture to each face of the box
        for i in range(6):  # 6 faces of the box
            p.changeVisualShape(
                body_id,
                -1,  # base link
                textureUniqueId=texture_id,
                rgbaColor=color,
                specularColor=[0.2, 0.2, 0.2],
                flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
                physicsClientId=client_id
            )

        print(f"Created package {item_id} at position {position} with QR codes on all faces")
        return body_id

    def generate_qr_code(self, item_id: str, info: Dict) -> str:
        """Generate QR code for a package with larger size and better quality
        
        Args:
            item_id: Unique identifier for item
            info: Item information to encode
            
        Returns:
            path: Path to generated QR code image
        """
        # Create QR code with better settings for visibility
        qr = qrcode.QRCode(
            version=3,  # Smaller version for cleaner appearance
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # Higher error correction
            box_size=12,  # Larger boxes for better visibility
            border=2,  # Smaller border since we'll add margin in texture
        )
        
        # Add data to encode
        qr_data = {
            'item_id': item_id,
            'name': info['name'],
            'shelf_id': info['shelf_id']
        }
        qr.add_data(str(qr_data))
        qr.make(fit=True)
        
        # Create image with pure black and white
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Save QR code with high quality
        img_path = os.path.join(self.qr_folder, f"{item_id}.png")
        qr_img = qr_img.convert('RGB')
        qr_img.save(img_path, 'PNG', quality=100)
        
        return img_path
    
    def scan_area(self, 
                  drone_pos: np.ndarray,
                  scan_radius: float = 0.5
                  ) -> List[str]:
        """Simulate scanning of packages in an area
        Note: This is now deprecated in favor of QR code scanning
        """
        detected_items = []
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
            shelf_id = self.inventory[item_id]['location']['shelf_id']
            row = self.inventory[item_id]['location']['row']
            position = self.inventory[item_id]['location']['position']
            print(f"Updated inventory for item on shelf {shelf_id}, row {row}, position {position} at time {timestamp}")
            
    def get_item_info(self, item_id: str) -> Dict:
        """Get information about an inventory item
        
        Args:
            item_id: Item ID to look up
            
        Returns:
            info: Dict of item information
        """
        return self.inventory.get(item_id, None)
    
    def get_items_by_row(self, row_number: int) -> List[Dict]:
        """Get all items in a specific row"""
        row_items = []
        for item_id, info in self.inventory.items():
            if info['location']['row'] == row_number:
                row_items.append({
                    'item_id': item_id,
                    'info': info
                })
        return row_items
    
    def get_items_by_shelf(self, shelf_id: str) -> List[Dict]:
        """Get all items on a specific shelf"""
        shelf_items = []
        for item_id, info in self.inventory.items():
            if info['location']['shelf_id'] == shelf_id:
                shelf_items.append({
                    'item_id': item_id,
                    'info': info
                })
        return shelf_items
    
    def get_unchecked_items(self, threshold: float) -> List[str]:
        """Get items not checked within time threshold"""
        unchecked = []
        current_time = time.time()
        for item_id, info in self.inventory.items():
            if info['last_checked'] is None or \
               (current_time - info['last_checked']) > threshold:
                unchecked.append(item_id)
        return unchecked