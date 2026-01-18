
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
import tensorflow as tf
import numpy as np

# Path to the TFRecord file
file_path = '/home/charles/workspaces/Double_Piper_Teleop/test/liber_o10-train.tfrecord-00000-of-00032'

def inspect_tfrecord(file_path):
    print(f"Inspecting: {file_path}")
    if not os.path.exists(file_path):
        print("File not found.")
        return

    # Create a feature description dictionary to parse the features
    # This must match roughly what we expect, but for raw inspection we can parse partially
    # Or just use the iterator to check values.
    
    # Since raw parsing is complex without the full schema, we will reconstruct 
    # based on the counts we saw earlier (which confirmed T=116, Dim=7, etc.)
    
    raw_dataset = tf.data.TFRecordDataset(file_path)
    
    count = 0
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print("\n--- Detailed Data Inspection (First Episode) ---")
        
        # Helper to get numpy array from feature
        def get_data(key, dtype=float):
            feat = example.features.feature[key]
            kind = feat.WhichOneof('kind')
            if kind == 'float_list':
                return np.array(feat.float_list.value, dtype=dtype)
            elif kind == 'int64_list':
                return np.array(feat.int64_list.value, dtype=dtype)
            return None

        # Number of steps estimation
        is_first = get_data('steps/is_first', int)
        steps = len(is_first)
        print(f"Total Steps in Episode: {steps}")

        # Check Action
        action = get_data('steps/action', np.float32)
        if action is not None:
            # Reshape based on known dim 7
            action = action.reshape(steps, 7)
            print(f"\n[Feature: steps/action]")
            print(f"  Shape: {action.shape} | Dtype: {action.dtype}")
            print(f"  Range: min={action.min():.4f}, max={action.max():.4f}, mean={action.mean():.4f}")
            print(f"  First 3 samples:\n{action[:3]}")
        
        # Check State
        state = get_data('steps/observation/state', np.float32)
        if state is not None:
            state = state.reshape(steps, 7)
            print(f"\n[Feature: steps/observation/state] (EEF + Gripper)")
            print(f"  Shape: {state.shape} | Dtype: {state.dtype}")
            print(f"  Range: min={state.min():.4f}, max={state.max():.4f}")
            print(f"  First 3 samples:\n{state[:3]}")
            
            # Check gripper specifically (last dim)
            gripper = state[:, -1]
            print(f"  Gripper Range: min={gripper.min():.4f}, max={gripper.max():.4f}")

        # Check Joint State
        joint_state = get_data('steps/observation/joint_state', np.float32)
        if joint_state is not None:
            joint_state = joint_state.reshape(steps, 7)
            print(f"\n[Feature: steps/observation/joint_state]")
            print(f"  Shape: {joint_state.shape} | Dtype: {joint_state.dtype}")
            print(f"  Range: min={joint_state.min():.4f}, max={joint_state.max():.4f}")
            
        # Check Language Embedding
        lang_emb = get_data('steps/language_embedding', np.float32)
        if lang_emb is not None:
            lang_emb = lang_emb.reshape(steps, 512)
            print(f"\n[Feature: steps/language_embedding]")
            print(f"  Shape: {lang_emb.shape} | Dtype: {lang_emb.dtype}")
            print(f"  Range: min={lang_emb.min():.4f}, max={lang_emb.max():.4f}")

        # Check Images (Decode one to check dtype/range)
        feat_img = example.features.feature['steps/observation/image']
        img_bytes = feat_img.bytes_list.value[0] # Take first frame
        img_decoded = tf.io.decode_image(img_bytes)
        print(f"\n[Feature: steps/observation/image] (Checking first frame)")
        print(f"  Shape: {img_decoded.shape} | Dtype: {img_decoded.dtype}")
        print(f"  Range: min={np.min(img_decoded)}, max={np.max(img_decoded)}")

        # Check Instructions
        feat_instr = example.features.feature['steps/language_instruction']
        instr = feat_instr.bytes_list.value[0].decode('utf-8')
        print(f"\n[Feature: steps/language_instruction]")
        print(f"  Sample: '{instr}'")

if __name__ == "__main__":
    inspect_tfrecord(file_path)
