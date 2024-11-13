import math
import time
from src.environment import WarehouseEnv
import pybullet as p

def main():
    # Create and setup environment
    env = WarehouseEnv(gui=True)
    env.setup()
    
    try:
        while True:
            # Step simulation
            p.stepSimulation()
            time.sleep(1./240.)  # Run at approximately 240 FPS
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        env.close()

if __name__ == "__main__":
    main()