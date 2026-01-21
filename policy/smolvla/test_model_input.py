
import sys
import os
import numpy as np
import time
import cv2
import torch
from datetime import datetime

# Calculate root path: Double_Piper_Teleop
# Current file is in Double_Piper_Teleop/policy/smolvla/
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(workspace_root)

print(f"Workspace Root: {workspace_root}")

try:
    from my_robot.agilex_piper_single_base import PiperSingle
    from policy.smolvla.inference_model import SMOLVLA
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configuration (Matches example/deploy/piper_single_on_smolvla.py)
CKPT_PATH = "/home/charles/workspaces/lerobot/weights/train/piper-pick-banana-50/checkpoints/009000/pretrained_model"
DATASET_REPO_ID = "miku112/piper-pick-banana-50"
DATASET_ROOT = "/home/charles/workspaces/lerobot/datasets/miku112/piper-pick-banana-50"
TASK_INSTRUCTION = "pick up the banana and put it into the bowl"
ACTION_CHUNK_SIZE = 50 

SAVE_DIR = os.path.join(workspace_root, "test", "check_input")

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def input_transform(data):
    # Same as deploy script
    left_joint = np.array(data[0]["left_arm"]["joint"]).reshape(-1)
    left_gripper = np.array(data[0]["left_arm"]["gripper"]).reshape(-1)
    state = np.concatenate([left_joint, left_gripper])
    img_head = data[1]["cam_head"]["color"]
    img_wrist = data[1]["cam_wrist"]["color"]
    return (img_head, img_wrist), state

def save_batch_debug(batch, step_idx):
    """
    Saves the processed batch content to disk.
    Batch is expected to be a dictionary of tensors (likely on GPU).
    """
    ensure_dir(SAVE_DIR)
    timestamp = datetime.now().strftime("%H%M%S_%f")
    prefix = f"step_{step_idx:03d}_{timestamp}"

    print(f"[{step_idx}] Debugging Preprocessed Batch:")
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # Move to CPU numpy
            val_np = value.detach().cpu().numpy()
            
            print(f"  Key: {key}, Shape: {val_np.shape}, Range: [{val_np.min():.3f}, {val_np.max():.3f}]")
            
            # Save raw tensor data
            np_filename = os.path.join(SAVE_DIR, f"{prefix}_{key.replace('.','_')}.npy")
            np.save(np_filename, val_np)
            
            # If it looks like an image, try to save a visualization
            # LeRobot images are usually [B, C, H, W]
            if "image" in key and val_np.ndim == 4:
                # Take first batch item: [C, H, W]
                img_chw = val_np[0]
                # Transpose to [H, W, C]
                img_hwc = np.transpose(img_chw, (1, 2, 0))
                
                # Normalize for visualization: map min-max to 0-255
                # Note: The model input is likely normalized by mean/std, so it might not look like a normal photo.
                v_min, v_max = img_hwc.min(), img_hwc.max()
                if v_max - v_min > 1e-6:
                    img_vis = (img_hwc - v_min) / (v_max - v_min) * 255.0
                else:
                    img_vis = img_hwc * 0
                
                img_vis = img_vis.astype(np.uint8)
                # Convert RGB to BGR for OpenCV
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                
                img_filename = os.path.join(SAVE_DIR, f"{prefix}_VIS_{key.replace('.','_')}.jpg")
                cv2.imwrite(img_filename, img_vis)
        else:
            print(f"  Key: {key}, Type: {type(value)}")

def main():
    ensure_dir(SAVE_DIR)
    
    # 1. Initialize Robot
    print("Initializing Robot...")
    try:
        robot = PiperSingle()
        robot.set_up()
        robot.reset()
    except Exception as e:
        print(f"Robot initialization failed: {e}")
        return

    # 2. Initialize Model
    print(f"Loading Model from {CKPT_PATH}...")
    try:
        model = SMOLVLA(
            model_path=CKPT_PATH, 
            dataset_repo_id=DATASET_REPO_ID, 
            dataset_root=DATASET_ROOT,
            action_chunk_size=ACTION_CHUNK_SIZE
        )
        model.random_set_language(TASK_INSTRUCTION)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    # 3. Hook the Preprocessor
    print("Hooking model preprocessor...")
    original_preprocessor = model.preprocessor
    
    # We use a closure or a class based wrapper to keep track of steps or just pass it through
    # For simplicity, we assume single threaded simple execution
    global step_counter
    step_counter = 0

    def hooked_preprocessor(batch):
        global step_counter
        # 1. Call original logic
        processed_batch = original_preprocessor(batch)
        
        # 2. Inspect results
        try:
            save_batch_debug(processed_batch, step_counter)
        except Exception as e:
            print(f"Error in hook: {e}")
        
        step_counter += 1
        return processed_batch

    # Replace the method on the instance
    model.preprocessor = hooked_preprocessor

    # 4. Run Test Loop
    print("Starting Inference Test Loop (5 steps)...")
    model.reset_obsrvationwindows()
    
    # Warmup
    time.sleep(1)
    
    for i in range(5):
        print(f"--- Test Step {i} ---")
        
        # Get Obs
        data = robot.get()
        if data is None:
            print("Failed to get robot data")
            continue

        img_arr, state = input_transform(data)
        
        # Update Model
        model.update_observation_window(img_arr, state)
        
        # Predict (This triggers the hook)
        try:
            action = model.get_action_chunk()
            print(f"Action shape: {action.shape}")
        except Exception as e:
            print(f"Inference failed or interrupted: {e}")
            break
            
        time.sleep(0.5)

    print(f"Test finished. Check results in {SAVE_DIR}")

if __name__ == "__main__":
    main()
