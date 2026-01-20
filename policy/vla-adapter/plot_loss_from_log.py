''''
Example:
    cd VLA-Adapter && \
    python plot_loss_from_log.py --log_file_path ./logs/VLA_4GPU_20260118_151755.log

'''

import re
import matplotlib.pyplot as plt
import os
from pathlib import Path   
from dataclasses import dataclass
import draccus
@dataclass
class PlotLossConfig:
    log_file_path: str = './logs/VLA_4GPU_20260118_151755.log'  # Path to the log file
@draccus.wrap() 
def plot_loss(cfg: PlotLossConfig):
    log_path = cfg.log_file_path  # Relative path assuming script is in the same dir
    
    output_image = Path(cfg.log_file_path).with_suffix('.png')
    # If script is run from root, adjust path
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found at {log_path}")

    curr_losses = []
    pattern = re.compile(r"curr:\s*(\d+\.\d+)")

    try:
        with open(log_path, 'r') as f:
            for line in f:
                matches = pattern.findall(line)
                for match in matches:
                    curr_losses.append(float(match))

        if not curr_losses:
            print("No 'curr' loss values found in the log file.")
        else:
            print(f"Found {len(curr_losses)} loss values.")
            
            plt.figure(figsize=(10, 6))
            plt.plot(curr_losses, label='curr loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Curr Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(output_image)
            print(f"Loss curve saved to {output_image}")

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_loss()