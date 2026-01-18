
import h5py
import os

file_path = '/home/charles/workspaces/Double_Piper_Teleop/datasets/pick_banana_50/0.hdf5'

def print_structure(name, obj):
    print(name)
    if isinstance(obj, h5py.Dataset):
        print(f"  Shape: {obj.shape}, Type: {obj.dtype}")
        # Print a few attributes if any
        for key, val in obj.attrs.items():
            print(f"  Attr {key}: {val}")

if os.path.exists(file_path):
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_structure)
        # Check attrs of root
        print("Root attributes:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")
else:
    print(f"File not found: {file_path}")
