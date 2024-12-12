import os
from datetime import datetime
from typing import Dict, List


class InventoryLogger:
    """Handles inventory logging and tracking"""
    
    def __init__(self, output_folder: str = "inventory_logs"):
        """Initialize inventory logger
        
        Args:
            output_folder: Directory for log files
        """
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_folder, f"inventory_log_{timestamp}.txt")
        self.detection_file = os.path.join(output_folder, f"detection_log_{timestamp}.txt")
        
        self.detected_items = set()
        
    def log_initial_inventory(self, inventory: Dict) -> None:
        """Log initial state of all items in inventory
        
        Args:
            inventory: Dictionary of all inventory items
        """
        with open(self.log_file, 'w') as f:
            f.write("=== Initial Inventory State ===\n\n")
            
            sorted_items = sorted(inventory.items(), 
                                key=lambda x: (x[1]['shelf_id'], x[0]))
            
            current_shelf = None
            for item_id, info in sorted_items:
                if info['shelf_id'] != current_shelf:
                    current_shelf = info['shelf_id']
                    f.write(f"\nShelf {current_shelf}:\n")
                    f.write("-" * 40 + "\n")
                
                f.write(f"Item ID: {item_id}\n")
                f.write(f"  Name: {info['name']}\n")
                f.write(f"  Location: {info['location']}\n")
                f.write(f"  Quantity: {info['quantity']}\n\n")
                
    def log_detection(self, 
                     item_id: str,
                     info: Dict,
                     drone_pos: List[float],
                     timestamp: float
                     ) -> None:
        """Log detection of an item
        
        Args:
            item_id: ID of detected item
            info: Item information
            drone_pos: Current drone position
            timestamp: Detection timestamp
        """
        if item_id not in self.detected_items:
            self.detected_items.add(item_id)
            
            formatted_time = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
            
            with open(self.detection_file, 'a') as f:
                f.write(f"=== New Item Detected ===\n")
                f.write(f"Time: {formatted_time}\n")
                f.write(f"Item ID: {item_id}\n")
                f.write(f"Name: {info['name']}\n")
                f.write(f"Shelf: {info['shelf_id']}\n")
                f.write(f"Drone Position: {[round(x, 2) for x in drone_pos]}\n")
                f.write("-" * 40 + "\n\n")
                
            print(f"\n[DETECTION] {formatted_time} - Found {info['name']} on {info['shelf_id']}")
    
    def get_detection_stats(self) -> Dict:
        """Get statistics about detected items
        
        Returns:
            stats: Dictionary of detection statistics
        """
        return {
            'total_detected': len(self.detected_items),
            'detected_items': sorted(list(self.detected_items))
        }