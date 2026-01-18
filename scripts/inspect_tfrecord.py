
import tensorflow as tf
import os
import json

file_path = '/home/charles/workspaces/Double_Piper_Teleop/test/liber_o10-train.tfrecord-00000-of-00032'

def print_example(record):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    print(example)

# Read the first record
if os.path.exists(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    for raw_record in raw_dataset.take(1):
        print_example(raw_record)
else:
    print(f"File not found: {file_path}")

# Check for JSON files
json_files = [f for f in os.listdir('/home/charles/workspaces/Double_Piper_Teleop/test') if f.endswith('.json')]
print("\nJSON Files found:")
for jf in json_files:
    print(f"- {jf}")
    try:
         with open(os.path.join('/home/charles/workspaces/Double_Piper_Teleop/test', jf), 'r') as f:
             print(json.dumps(json.load(f), indent=2))
    except Exception as e:
        print(f"Error reading {jf}: {e}")
