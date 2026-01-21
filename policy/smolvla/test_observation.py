
import sys
import os
import numpy as np
import time
import cv2
from datetime import datetime

# Calculate root path: Double_Piper_Teleop
# Current file is in Double_Piper_Teleop/policy/smolvla/
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(workspace_root)

print(f"Workspace Root: {workspace_root}")

try:
    from my_robot.agilex_piper_single_base import PiperSingle
except ImportError as e:
    print(f"Error importing PiperSingle: {e}")
    print("Please make sure you are in the correct environment and the path is correct.")
    sys.exit(1)

def input_transform(data):
    """
    Process robot data for SmolVLA model
    Same logic as in example/deploy/piper_single_on_smolvla.py
    """
    # 1. State processing (Proprioception)
    left_joint = np.array(data[0]["left_arm"]["joint"]).reshape(-1)
    left_gripper = np.array(data[0]["left_arm"]["gripper"]).reshape(-1)
    
    # Construct state vector
    state = np.concatenate([left_joint, left_gripper])
    
    # 2. Image processing
    img_head = data[1]["cam_head"]["color"]
    img_wrist = data[1]["cam_wrist"]["color"]
    
    return (img_head, img_wrist), state

def main():
    # Setup save directory
    save_base_dir = os.path.join(workspace_root, "test", "check_obs")
    if not os.path.exists(save_base_dir):
        os.makedirs(save_base_dir)
        print(f"Created directory: {save_base_dir}")
    else:
        print(f"Saving to existing directory: {save_base_dir}")

    # Initialize Robot
    print("Initializing Robot...")
    try:
        robot = PiperSingle()
        robot.set_up()
        robot.reset()
        print("Robot initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize robot: {e}")
        return

    # Capture loop
    num_frames = 5
    print(f"Starting capture of {num_frames} frames...")
    print("Keep the robot still or move it slowly to check for observation consistency.")
    
    time.sleep(1) # Warm up

    for i in range(num_frames):
        start_time = time.time()
        
        # Get Observation
        try:
            data = robot.get()
            if data is None:
                print(f"Frame {i}: robot.get() returned None!")
                continue
                
            img_arr, state = input_transform(data)
            img_head, img_wrist = img_arr
            
            # Print state for immediate feedback
            print(f"Frame {i} State: {state}")
            
            # Save format: frame_001_head.jpg, frame_001_state.npy/txt
            timestamp = datetime.now().strftime("%H%M%S_%f")
            
            # Save Images (Convert RGB to BGR for OpenCV)
            if img_head is not None:
                img_head_bgr = cv2.cvtColor(img_head, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_base_dir, f"frame_{i:03d}_{timestamp}_head.jpg"), img_head_bgr)
            
            if img_wrist is not None:
                img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_base_dir, f"frame_{i:03d}_{timestamp}_wrist.jpg"), img_wrist_bgr)
                
            # Save State
            np.save(os.path.join(save_base_dir, f"frame_{i:03d}_{timestamp}_state.npy"), state)
            with open(os.path.join(save_base_dir, f"frame_{i:03d}_{timestamp}_state.txt"), "w") as f:
                f.write(str(state.tolist()))
                
        except Exception as e:
            print(f"Error checking frame {i}: {e}")
            import traceback
            traceback.print_exc()
        
        # Sleep to simulate control loop freq (e.g. 10Hz - 50Hz)
        time.sleep(0.1)

    print("Test finished. Please check the 'Double_Piper_Teleop/test/check_obs' directory.")
    
    # Optional: robot cleanup if needed
    # robot.close() 

if __name__ == "__main__":
    main()
