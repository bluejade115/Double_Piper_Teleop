import tensorflow as tf
import os
import numpy as np

# Disable GPU to avoid errors if CUDA is not configured
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

TFRECORD_PATH = '/home/charles/workspaces/Double_Piper_Teleop/test/liber_o10-train.tfrecord-00000-of-00032'

def inspect_tfrecord(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Inspecting: {file_path}")
    
    # Create dataset
    # Note: We don't know the exact schema, so we parse it generically first if possible,
    # or assume a standard RLDS structure. 
    # Since RLDS tfrecords are standard Example protos, we can parse them.
    
    raw_dataset = tf.data.TFRecordDataset(file_path)

    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print("\n--- Features in Example ---")
        feature_keys = sorted(example.features.feature.keys())
        for key in feature_keys:
            feat = example.features.feature[key]
            kind = feat.WhichOneof('kind')
            
            # Helper to get value
            val = "N/A"
            length = 0
            if kind == 'bytes_list':
                length = len(feat.bytes_list.value)
                if length == 1:
                    # Check if it looks like a tensor storage (numpy bytes)
                    v = feat.bytes_list.value[0]
                    val = f"<Bytes len={len(v)}>"
            elif kind == 'float_list':
                length = len(feat.float_list.value)
                if length < 10:
                    val = feat.float_list.value
                else:
                    val = f"[Float list len={length}]"
            elif kind == 'int64_list':
                length = len(feat.int64_list.value)
                if length < 10:
                    val = feat.int64_list.value
                else:
                    val = f"[Int64 list len={length}]"
            
            print(f"Key: {key} | Type: {kind} | Length: {length} | Val: {val}")

    print("\n--- Decoding 'steps/action' and 'steps/observation/state' if possible ---")
    
    # Simple decoder for common RLDS features
    # We assume 'steps/action' is a float list or bytes
    
    for i, raw_record in enumerate(raw_dataset.take(3)):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print(f"\nEpisode {i} Analysis:")
        
        # Try to find standard fields
        # Note: In standard RLDS/TFDS serialization, tensors are often flattened.
        features = example.features.feature
        
        # Check Action
        if 'steps/action' in features:
            f = features['steps/action']
            if f.float_list.value:
                vals = np.array(f.float_list.value)
                print(f"  steps/action (Flat): {vals.shape}, Total Elements: {vals.size}")
                # Try to guess shape based on length. Usually [T, D]
                # If we know T (e.g. from is_last or checking other fields)
            else:
                print("  steps/action: Not a float_list")
        
        # Check State
        if 'steps/observation/state' in features:
            f = features['steps/observation/state']
            if f.float_list.value:
                vals = np.array(f.float_list.value)
                print(f"  steps/observation/state (Flat): {vals.shape}, Total Elements: {vals.size}")
                
        # Check Image
        if 'steps/observation/image' in features:
             f = features['steps/observation/image']
             print("  steps/observation/image: Present (likely encoded images)")

        # Determine Episode Length using 'steps/is_last' or similar
        if 'steps/is_last' in features:
            is_last = features['steps/is_last'].int64_list.value
            print(f"  steps/is_last count: {len(is_last)}")
            T = len(is_last)
            
            # Now try to reshape inputs
            if 'steps/action' in features and features['steps/action'].float_list.value:
                act_data = np.array(features['steps/action'].float_list.value)
                if act_data.size % T == 0:
                    dim = act_data.size // T
                    print(f"  -> Inferred Action Shape: ({T}, {dim})")
                    print(f"  -> First Action: {act_data[:dim]}")
                else:
                    print(f"  -> Action size {act_data.size} not divisible by steps {T}")

            if 'steps/observation/state' in features and features['steps/observation/state'].float_list.value:
                obs_data = np.array(features['steps/observation/state'].float_list.value)
                if obs_data.size % T == 0:
                    dim = obs_data.size // T
                    print(f"  -> Inferred State Shape: ({T}, {dim})")
                    print(f"  -> First State: {obs_data[:dim]}")


if __name__ == "__main__":
    inspect_tfrecord(TFRECORD_PATH)
