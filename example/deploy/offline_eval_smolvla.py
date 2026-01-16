import sys
sys.path.append("./")
import os
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import torch

from utils.data_handler import debug_print, hdf5_groups_to_dict
from my_robot.base_robot import dict_to_list

# Import SmolVLA model wrapper we just created
from policy.smolvla.inference_model import SMOLVLA

# Import helper functions from offline_eval (or redefine them if importing is messy)
# For simplicity and isolation, I'll redefine the relevant transforms here based on offline_eval.py content

### ========= CONFIG =========
MODEL_CHUNK_SIZE = 1 # SmolVLA usually predicts next action or chunk. Let's assume 1 for now unless specified.
SKIP_FRAME = 20 # Evaluation interval
CKPT_PATH = "path/to/your/smolvla/checkpoint" # Placeholder
TASK_NAME = "pick_banana" # Placeholder
### ==========================


def input_transform(data):
    """
    Transforms robot data into model input format.
    PiperSingle returns:
    data[0]: arm state (joint, gripper)
    data[1]: images (cam_head, cam_wrist)
    """
    # Create valid state vector as per offline_eval logic
    # offline_eval/input_transform expects:
    # left_joint (6), left_gripper (1), right_joint (6), right_gripper (1)
    
    # We need to robustly handle missing arms just like in piper_single_on_ACT
    # But offline_eval seems to assume specific structure.
    # Let's look at `piper_single_on_ACT.py` input_transform for robustness.
    
    # Actually, for offline eval, we are reading from collected HDF5 data.
    # HDF5 data structure from `offline_eval.py`:
    # data[0] is state dict, data[1] is images dict.
    
    left_joint = np.array(data[0].get("left_arm", {}).get("joint", [0]*6)).reshape(-1)
    left_gripper = np.array(data[0].get("left_arm", {}).get("gripper", [0])).reshape(-1)
    
    # Fill dummy right if missing (since PiperSingle is single arm)
    right_joint = np.zeros(6)
    right_gripper = np.zeros(1)
    
    state = np.concatenate([
        left_joint, left_gripper, right_joint, right_gripper
    ])

    # Images: offline_eval expects: cam_head, cam_right_wrist, cam_left_wrist
    # But PiperSingle has: cam_head, cam_wrist
    # Let's map cam_wrist to appropriate side. Assuming left arm robot -> left wrist.
    
    cam_head = data[1].get("cam_head", {}).get("color", None)
    cam_wrist = data[1].get("cam_wrist", {}).get("color", None)
    
    # Return in format expected by SMOLVLA update_observation_window
    # which we defined as [cam_head, cam_wrist]
    return (cam_head, cam_wrist), state

def compare_transform(data_chunk):
    """
    Transforms ground truth data chunk into action format for comparison.
    """
    actions = []
    for data in data_chunk[0]:
        # Extract Left Arm Action
        left_joint = np.array(data.get("left_arm", {}).get("joint", [0]*6)).reshape(-1)
        left_gripper = np.array(data.get("left_arm", {}).get("gripper", [0])).reshape(-1)
        
        # We only care about the active arm (Left) for PiperSingle
        action = np.concatenate([left_joint, left_gripper])
        actions.append(action)

    return np.stack(actions)

class Replay:
    def __init__(self, hdf5_path) -> None:
        self.ptr = 0
        self.episode = dict_to_list(hdf5_groups_to_dict(hdf5_path))
    
    def get_data(self):
        try:
            if self.ptr >= len(self.episode):
                return None, None, None
            
            data = self.episode[self.ptr], self.episode[self.ptr]
            
            # For chunking (if model predicts chunks)
            # SmolVLA standard is usually single step, but openvla can be configured.
            # We'll grab just 1 step for now or MODEL_CHUNK_SIZE
            data_chunk_end_ptr = min(len(self.episode), self.ptr + MODEL_CHUNK_SIZE)
            data_chunk = self.episode[self.ptr:data_chunk_end_ptr], self.episode[self.ptr:data_chunk_end_ptr]
            
            # Step forward
            current_ptr = self.ptr
            self.ptr += SKIP_FRAME
            
            return data, data_chunk, (current_ptr, data_chunk_end_ptr)
        except Exception as e:
            print(f"Error in Replay: {e}")
            return None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH, help="Path to SmolVLA checkpoint")
    parser.add_argument("--task", type=str, default=TASK_NAME, help="Task name")
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 dataset file for evaluation")
    args = parser.parse_args()

    # Initialize Model
    try:
        model = SMOLVLA(args.ckpt, args.task)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Initialize Dataset Replay
    replay = Replay(args.data)
    
    print(f"Starting Offline Evaluation on {args.data}")
    
    errors = []
    
    step_count = 0
    while True:
        data, data_chunk, _ = replay.get_data()
        if data is None:
            break
            
        # Preprocess Input
        img_arr, state = input_transform(data)
        
        # Update Model State
        # You might need to set specific instruction manually if not "task_name" based
        model.update_observation_window(img_arr, state)
        # model.random_set_language("pick up the object") # Optional: Override instruction if needed
        
        # Inference
        try:
            pred_action = model.get_action() # Expected shape depends on model. Usually [7] for 7-dof or formatted.
        except Exception as e:
            print(f"Inference error: {e}")
            break

        # Ground Truth
        real_action_chunk = compare_transform(data_chunk)
        # real_action_chunk shape: [CHUNK_SIZE, 7] (6 joint + 1 gripper)

        # Handle Predictions
        # If pred_action is single step [7], expand to [1, 7]
        if len(pred_action.shape) == 1:
            pred_action = pred_action.reshape(1, -1)
            
        # Compare
        # Ensure dimensions match (truncate if needed)
        min_len = min(pred_action.shape[0], real_action_chunk.shape[0])
        pred_comp = pred_action[:min_len]
        real_comp = real_action_chunk[:min_len]
        
        # Compute Error (L2 Norm)
        error = np.linalg.norm(pred_comp - real_comp)
        errors.append(error)
        
        step_count += 1
        print(f"Step {step_count}: Error = {error:.4f}")

    if errors:
        print("="*30)
        print(f"Average Prediction Error: {np.mean(errors):.4f}")
        print("="*30)
        
        plt.figure()
        plt.plot(errors, label='Prediction Error (L2)')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Error')
        plt.title('Offline Evaluation - SmolVLA')
        plt.legend()
        plt.savefig('smolvla_offline_eval.png')
        print("Saved error plot to smolvla_offline_eval.png")

if __name__ == "__main__":
    main()
