
import tensorflow as tf
import os
import numpy as np
from typing import Dict, Any

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Path to the Piper dataset TFRecord
TFRECORD_PATH = '/home/charles/workspaces/Double_Piper_Teleop/datasets_rlds/piper_dataset/pick_banana_50/1.0.0/piper_dataset-train.tfrecord-00000-of-00004'

def piper_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform for Piper dataset to load pre-calculated EEF Delta actions and format state.
    Expected Input:
      'action_eef_delta': [T, 7] (Pre-calculated in builder)
      'observation': { 'state': [T, 7] (EEF), ... }
    """
    
    # 1. Load Pre-calculated Action (EEF Delta)
    # The builder already computed: [Delta_XYZ (3), Delta_RPY (3), Target_Gripper_Abs (1)]
    if 'action_eef_delta' in trajectory:
        trajectory['action'] = trajectory['action_eef_delta']
    else:
        raise ValueError("Feature 'action_eef_delta' not found in trajectory! Ensure dataset was regenerated with new builder.")
    
    # 2. Format Observation
    state = trajectory['observation']['state']
    
    # OpenVLA often expects 'proprio' padded to 8 dims.
    # POS_EULER = [XYZ (3), RPY (3), PAD (1), Gripper (1)]
    # Piper state is [XYZ (3), RPY (3), Gripper (1)].
    
    obs_pose = state[:, :6]
    obs_gripper = state[:, 6:]
    padding = tf.zeros([tf.shape(state)[0], 1], dtype=state.dtype)

    # Insert padding before gripper
    proprio = tf.concat([obs_pose, padding, obs_gripper], axis=-1)
    
    trajectory['observation']['proprio'] = proprio
    
    # Libero-style splits for specific adapters
    trajectory['observation']['EEF_state'] = obs_pose
    trajectory['observation']['gripper_state'] = obs_gripper
    
    return trajectory

def parse_tfrecord_example(example_proto):
    """
    Manually parses the specific structure of the Piper TFRecord into a trajectory dict.
    """
    feature_description = {
        'steps/observation/state': tf.io.VarLenFeature(tf.float32),
        'steps/observation/image': tf.io.VarLenFeature(tf.string), 
        'steps/action': tf.io.VarLenFeature(tf.float32),          # Original Joint Action
        'steps/action_eef_delta': tf.io.VarLenFeature(tf.float32), # New EEF Delta Action
        'steps/is_last': tf.io.VarLenFeature(tf.int64),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode sparse tensors
    state_flat = tf.sparse.to_dense(parsed['steps/observation/state'])
    action_flat = tf.sparse.to_dense(parsed['steps/action'])
    is_last = tf.sparse.to_dense(parsed['steps/is_last'])
    
    # Try to decode new field
    if parsed['steps/action_eef_delta'].values.shape[0] > 0:
        action_eef_delta_flat = tf.sparse.to_dense(parsed['steps/action_eef_delta'])
    else:
        action_eef_delta_flat = None

    T = tf.shape(is_last)[0]
    
    # Reshape
    state = tf.reshape(state_flat, (T, 7))
    action = tf.reshape(action_flat, (T, 7))
    
    trajectory = {
        'action': action,
        'observation': {
            'state': state
        }
    }
    
    if action_eef_delta_flat is not None:
         action_eef_delta = tf.reshape(action_eef_delta_flat, (T, 7))
         trajectory['action_eef_delta'] = action_eef_delta
    
    return trajectory

def test_transform():
    if not os.path.exists(TFRECORD_PATH):
        print(f"File not found: {TFRECORD_PATH}")
        return

    print("Reading TFRecord...")
    raw_dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
    
    for raw_record in raw_dataset.take(1):
        # 1. Parse
        trajectory = parse_tfrecord_example(raw_record)
        
        print("\n--- Before Transform ---")
        print("Keys:", trajectory.keys())
        if 'action_eef_delta' in trajectory:
             print("Found 'action_eef_delta' feature.")
             print("Shape:", trajectory['action_eef_delta'].shape)
             print("First 3 values:", trajectory['action_eef_delta'][:3].numpy())
        else:
             print("ERROR: 'action_eef_delta' NOT found in TFRecord.")
             return

        # 2. Transform
        new_trajectory = piper_dataset_transform(trajectory)
        
        print("\n--- After Transform ---")
        print("New Action (assigned from delta):")
        print(new_trajectory['action'][:3].numpy())
        
        # Verify correctness: Is it consistent with runtime calculation?
        # Delta = State[1] - State[0]
        # Note: The Builder logic used (State[t+1] - State[t]) for pose, and State[t+1] for gripper
        calc_delta_pose_0 = trajectory['observation']['state'][1, :6] - trajectory['observation']['state'][0, :6]
        stored_action_0 = new_trajectory['action'][0]
        
        print("\nVerifying Stored vs Runtime Calc:")
        print("Runtime Calc Delta (Pose):", calc_delta_pose_0.numpy())
        print("Stored Action (Pose):", stored_action_0[:6].numpy())
        
        diff = np.abs(calc_delta_pose_0.numpy() - stored_action_0[:6].numpy())
        if np.all(diff < 1e-4):
            print("\nSUCCESS: Stored Action matches expected Delta logic.")
        else:
            print("\nFAILURE: Stored Action differs from runtime calc (Did you regenerate dataset?).")
            
        print("\nProprio Shape:", new_trajectory['observation']['proprio'].shape)

if __name__ == "__main__":
    test_transform()
