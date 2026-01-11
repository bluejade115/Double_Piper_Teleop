import sys
sys.path.append("./")
import os
import numpy as np
import time
import math
import pdb
from my_robot.agilex_piper_single_base import PiperSingle
from policy.smolvla.inference_model import SMOLVLA
from utils.data_handler import is_enter_pressed

def input_transform(data):
    """
    Process robot data for SmolVLA model
    """
    # 1. State processing (Proprioception)
    # PiperSingle typically has 'left_arm'
    # We construct a 7-DOF state: 6 joints + 1 gripper
    
    left_joint = np.array(data[0]["left_arm"]["joint"]).reshape(-1)
    left_gripper = np.array(data[0]["left_arm"]["gripper"]).reshape(-1)
    
    # Construct state vector (7 dim for single arm)
    # Note: If your model was trained with dual arm state padding, adjust here.
    # Assuming the refined smolvla (finetuned on piper) expects just the relevant arm state 
    # OR standard OpenVLA which primarily uses vision.
    # We pass the state just in case the model wrapper uses it or for logging.
    state = np.concatenate([left_joint, left_gripper])
    
    # 2. Image processing
    # PiperSingle returns: data[1]["cam_head"]["color"], data[1]["cam_wrist"]["color"]
    img_head = data[1]["cam_head"]["color"]
    img_wrist = data[1]["cam_wrist"]["color"]
    
    return (img_head, img_wrist), state

def output_transform(action):
    """
    Process model output action to robot command
    """
    # Action expected: [Joint1, ..., Joint6, Gripper] (7 dims)
    # Or Chunked [T, 7]
    
    # Take first step if chunked
    if len(action.shape) > 1:
        action = action[0]
        
    # Joint Limits (Safety) - Piper specific
    # Radian limits from Piper documentation/code
    joint_limits_rad = [
        (math.radians(-150), math.radians(150)),   # joint1
        (math.radians(0), math.radians(180)),      # joint2
        (math.radians(-170), math.radians(0)),     # joint3
        (math.radians(-100), math.radians(100)),   # joint4
        (math.radians(-70), math.radians(70)),     # joint5
        (math.radians(-120), math.radians(120))    # joint6
    ]

    def clamp(value, min_val, max_val):
        return max(min_val, min(value, max_val))
    
    # Map Action to Joints
    # Assuming model output corresponds to [J1...J6, Gripper]
    target_joints = [
        clamp(action[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    target_gripper = action[6]
    
    # Construct Move Command
    move_data = {
        "arm": {
            "left_arm": {
                "joint": target_joints,
                "gripper": target_gripper,
            }
        }
    }
    return move_data

def main():
    # 1. Configuration
    CKPT_PATH = "path/to/your/smolvla/checkpoint" # TODO: Update this
    DATASET_REPO_ID = "miku112/piper_pick_place_banana" # TODO: Ensure this matches your training dataset for correct normalization
    TASK_INSTRUCTION = "pick up the banana and put it into the container" # TODO: Update this
    
    # 2. Initialize Robot
    print("Initializing Robot...")
    robot = PiperSingle()
    robot.set_up()
    robot.reset()
    
    # 3. Initialize Model
    print(f"Loading Model from {CKPT_PATH}...")
    try:
        model = SMOLVLA(model_path=CKPT_PATH, dataset_repo_id=DATASET_REPO_ID)
        model.random_set_language(TASK_INSTRUCTION)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Inference Loop
    max_step = 2000
    num_episode = 10
    
    for i in range(num_episode):
        print(f"--- Episode {i+1}/{num_episode} ---")
        
        # Reset
        robot.reset()
        model.reset_obsrvationwindows()
        model.random_set_language(TASK_INSTRUCTION) # Reset instruction just in case
        
        # Wait for start
        print("Press ENTER to start inference...")
        is_start = False
        while not is_start:
            if is_enter_pressed():
                is_start = True
            time.sleep(0.1)
            
        print("Inference Started!")
        
        step = 0
        while step < max_step:
            loop_start = time.time()
            
            # Get Observation
            data = robot.get()
            img_arr, state = input_transform(data)
            
            # Update Model
            model.update_observation_window(img_arr, state)
            
            # Predict
            try:
                action = model.get_action()
            except Exception as e:
                print(f"Inference Failed: {e}")
                break
                
            # Execute
            move_data = output_transform(action)
            robot.move(move_data)
            
            step += 1
            
            # Frequency Control
            # elapsed = time.time() - loop_start
            # target_dt = 1.0 / robot.condition["save_freq"] # e.g. 1/30
            # if elapsed < target_dt:
            #     time.sleep(target_dt - elapsed)
            
            print(f"Step {step}: Action Executed")

        print(f"Episode {i+1} Finished.")
        robot.reset()

if __name__ == "__main__":
    main()
