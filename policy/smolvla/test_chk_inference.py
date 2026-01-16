import sys
import os
import torch
import numpy as np

# Add project root to sys.path to allow imports
# Assuming script is run from project root or checks relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from policy.smolvla.inference_model import SMOLVLA
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("LeRobot not installed or found in path.")

def test_chunked_inference():
    # 1. Configuration
    # Use paths similar to your openloop_smolvla_eval.py or deploy script
    # Update these if your paths are different
    CKPT_PATH = "../lerobot/weights/train/my_smolvla/checkpoints/015000/pretrained_model" 
    DATASET_REPO_ID = "miku112/piper-pick-place-banana-v2"
    DATASET_ROOT = "../lerobot/datasets/miku112/piper-pick-place-banana-v2"
    
    CHUNK_SIZE = 50 # We want to verify this chunk size
    
    print("="*50)
    print(f"Testing Chunked Inference (Size: {CHUNK_SIZE})")
    print(f"Ckpt: {CKPT_PATH}")
    print("="*50)

    # 2. Get a sample frame from dataset
    print(f"Loading single sample from dataset: {DATASET_REPO_ID}")
    try:
        dataset = LeRobotDataset(repo_id=DATASET_REPO_ID, root=DATASET_ROOT)
        sample = dataset[0] # Get index 0
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check DATASET_ROOT and ensure lerobot is installed.")
        return

    # 3. Simulate Robot Input (Convert Dataset output to Real-world input format)
    # LeRobotDataset usually returns CHW tensors (Float). 
    # Real Robot (Realsense) returns HWC numpy arrays (Uint8).
    # Inference wrapper expects HWC numpy.
    
    print("simulating robot observation...")
    
    # Helper to convert tensor (C,H,W) -> numpy (H,W,C)
    def to_robot_img(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.permute(1, 2, 0).numpy()
        return tensor

    img_head = to_robot_img(sample["observation.images.image"])
    img_wrist = to_robot_img(sample["observation.images.wrist_image"])
    
    # State: 1D array
    if isinstance(sample["observation.state"], torch.Tensor):
        state = sample["observation.state"].numpy()
    else:
        state = sample["observation.state"]

    print(f"Input Head Image Shape: {img_head.shape}")
    print(f"Input Wrist Image Shape: {img_wrist.shape}")
    print(f"Input State Shape: {state.shape}")

    # 4. Initialize Model with Chunk Override
    print("\nInitializing Model with action_chunk_size override...")
    try:
        model = SMOLVLA(
            model_path=CKPT_PATH, 
            dataset_repo_id=DATASET_REPO_ID, 
            dataset_root=DATASET_ROOT,
            action_chunk_size=CHUNK_SIZE # This is the key param to test
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # 5. Run Inference
    print("\nUpdating observation and predicting...")
    try:
        model.update_observation_window((img_head, img_wrist), state)
        action = model.get_action()
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    # 6. Verify Output
    print("\n" + "="*20 + " RESULTS " + "="*20)
    print(f"Output Action Type: {type(action)}")
    print(f"Output Action Shape: {action.shape}")
    
    expected_shape = (CHUNK_SIZE, 7) # Assuming 7 DoF (6 joint + gripper)
    
    if action.shape == expected_shape:
        print(f"\n[SUCCESS] Model successfully returned a chunk of {CHUNK_SIZE} actions.")
    else:
        print(f"\n[FAILURE] Model returned shape {action.shape}, expected {expected_shape}.")

    print("Sample of first 3 actions in chunk:")
    print(action[:3])

if __name__ == "__main__":
    test_chunked_inference()
