
import sys
import os
import torch
from safetensors.torch import load_file
import json

# Path to the checkpoint
ckpt_path = "/home/charles/workspaces/lerobot/weights/train/piper-pick-banana-50/checkpoints/009000/pretrained_model"

unnorm_file = os.path.join(ckpt_path, "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
norm_file = os.path.join(ckpt_path, "policy_preprocessor_step_5_normalizer_processor.safetensors")
config_file = os.path.join(ckpt_path, "train_config.json")

def print_stats(file_path, title):
    print(f"\n{'='*20} {title} {'='*20}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        tensors = load_file(file_path)
        for key, value in tensors.items():
            print(f"\nKey: {key}")
            print(f"Shape: {value.shape}")
            print(f"Values: {value}")
            
            # Basic analysis
            if "mean" in key or "std" in key:
                 print(f"  Min: {value.min()}, Max: {value.max()}")
                 if "std" in key and (value < 1e-6).any():
                     print("  WARNING: Extremely small std deviation found!")

    except Exception as e:
        print(f"Error loading {file_path}: {e}")

def main():
    print(f"Inspecting checkpoint at: {ckpt_path}")
    
    # Check Config for context
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            cfg = json.load(f)
            print("\n--- Training Config Summary ---")
            print(f"Repo ID: {cfg.get('dataset', {}).get('repo_id')}")
            print(f"Normalization: {cfg.get('policy', {}).get('normalization_mapping')}")
            print(f"n_action_steps: {cfg.get('policy', {}).get('n_action_steps')}")

    print_stats(unnorm_file, "Post-Processor (Action Un-normalization)")
    print_stats(norm_file, "Pre-Processor (Observation Normalization)")

if __name__ == "__main__":
    main()
