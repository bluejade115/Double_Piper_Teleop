import sys
import os
import json
import time
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ====== Path Setup ======
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOUBLE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
VLA_ROOT = os.path.join(REPO_ROOT, "VLA-Adapter")

sys.path.insert(0, DOUBLE_ROOT)
sys.path.insert(0, VLA_ROOT)

from my_robot.agilex_piper_single_base import PiperSingle
from utils.data_handler import is_enter_pressed

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    get_model,
    get_action,
    set_seed_everywhere,
)


# ====== State / Action Encoding ======
class StateEncoding(IntEnum):
    # fmt: off
    NONE = -1               # No Proprioceptive State
    POS_EULER = 1           # EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
    POS_QUAT = 2            # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    JOINT = 3               # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    JOINT_BIMANUAL = 4      # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
    # fmt: on


class ActionEncoding(IntEnum):
    # fmt: off
    EEF_POS = 1             # EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)
    JOINT_POS = 2           # Joint Delta Position (7) + Gripper Open/Close (1)
    JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
    EEF_R6 = 4              # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
    # fmt: on


@dataclass
class DeployConfig:
    # Model parameters
    pretrained_checkpoint: str = (
        "/home/lxx/repo/VLA-Adapter/outputs/"
        "configs+pick_banana_50+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--train1_4gpu--25000_chkpt"
    )
    model_family: str = "openvla"

    # Input parameters
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 1
    # Match dataset_statistics key in checkpoint (see dataset_statistics.json)
    unnorm_key: str = "pick_banana_50"

    # Action head / adapter settings
    use_l1_regression: bool = True
    use_minivlm: bool = True
    use_pro_version: bool = True
    use_film: bool = False

    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Action chunk execution
    execute_chunk_steps: int = 10


# ====== Input / Output Helpers ======

def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    return np.ascontiguousarray(image)


def build_proprio(controller_data: Dict[str, Any]) -> np.ndarray:
    # StateEncoding.POS_EULER: XYZ (3) + RPY (3) + PAD (1) + Gripper (1)
    eef = np.array(controller_data["left_arm"]["qpos"], dtype=np.float32).reshape(-1)
    gripper = np.array([controller_data["left_arm"]["gripper"]], dtype=np.float32).reshape(-1)
    if eef.shape[0] != 6:
        raise ValueError(f"Expected EEF state dim=6, got {eef.shape[0]}")
    pad = np.zeros((1,), dtype=np.float32)
    proprio = np.concatenate([eef, pad, gripper], axis=0)
    return proprio


def build_obs(sensor_data: Dict[str, Any], proprio: np.ndarray) -> Dict[str, Any]:
    img_head = to_uint8_rgb(sensor_data["cam_head"]["color"])
    img_wrist = to_uint8_rgb(sensor_data["cam_wrist"]["color"])

    obs = {
        "full_image": img_head,
        "state": proprio,
        "image_wrist": img_wrist,
    }
    return obs


def clamp(value: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(value, lo), hi)


def action_to_move(
    action: np.ndarray,
    current_qpos: np.ndarray,
    max_delta_pos: float = 0.05,
    max_delta_rpy: float = 0.2,
) -> Dict[str, Any]:
    # ActionEncoding.EEF_POS: delta XYZ (3) + delta RPY (3) + gripper (1)
    if action.ndim > 1:
        action = action[0]
    if action.shape[-1] < 7:
        raise ValueError(f"Expected action dim=7, got {action.shape[-1]}")

    delta_pos = clamp(action[:3], -max_delta_pos, max_delta_pos)
    delta_rpy = clamp(action[3:6], -max_delta_rpy, max_delta_rpy)

    target_qpos = current_qpos.copy()
    target_qpos[:3] += delta_pos
    target_qpos[3:6] += delta_rpy

    gripper = float(np.clip(action[6], 0.0, 1.0))

    return {
        "arm": {
            "left_arm": {
                "qpos": target_qpos.tolist(),
                "gripper": gripper,
            }
        }
    }


def ensure_unnorm_key(model: Any, cfg: DeployConfig) -> None:
    if not hasattr(model, "norm_stats"):
        return
    if cfg.unnorm_key in model.norm_stats:
        return
    keys = list(model.norm_stats.keys())
    if keys:
        print(f"[WARN] unnorm_key {cfg.unnorm_key} not found, use {keys[0]}")
        cfg.unnorm_key = keys[0]


def main():
    cfg = DeployConfig()
    set_seed_everywhere(0)

    print("Initializing Robot...")
    robot = PiperSingle()
    robot.set_up()
    robot.reset()

    # Configure Piper motion mode (EEF position control)
    try:
        piper_ctrl = robot.controllers["arm"]["left_arm"].controller
        # ctrl_mode=0x01 (CAN cmd), move_mode=0x00 (MOVE P), move_spd_rate=50, is_mit_mode=0x00
        piper_ctrl.MotionCtrl_2(0x01, 0x00, 50, 0x00)
    except Exception as e:
        print(f"[WARN] Failed to set MotionCtrl_2: {e}")

    print(f"Loading VLA-Adapter model from {cfg.pretrained_checkpoint}...")
    model = get_model(cfg)
    processor = get_processor(cfg)

    # Load action head + proprio projector (continuous action & proprio)
    try:
        llm_dim = model.config.text_config.hidden_size
    except Exception:
        llm_dim = 4096

    action_head = get_action_head(cfg, llm_dim)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, llm_dim, proprio_dim=8)

    ensure_unnorm_key(model, cfg)

    task_instruction = "pick up the banana"
    max_steps = 2000

    print("Press ENTER to start inference...")
    while True:
        if is_enter_pressed():
            break
        time.sleep(0.1)

    print("Inference Started!")
    step = 0
    while step < max_steps:
        data = robot.get()
        controller_data, sensor_data = data[0], data[1]

        proprio = build_proprio(controller_data)
        obs = build_obs(sensor_data, proprio)

        try:
            actions = get_action(
                cfg,
                model,
                obs,
                task_instruction,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                use_minivlm=cfg.use_minivlm,
                use_film=cfg.use_film,
            )
        except Exception as e:
            print(f"Inference failed: {e}")
            break

        # Normalize action container to a list for chunk execution
        if isinstance(actions, list):
            action_list = actions
        elif isinstance(actions, np.ndarray) and actions.ndim > 1:
            action_list = [actions[i] for i in range(actions.shape[0])]
        else:
            action_list = [actions]
        
        exec_steps = min(cfg.execute_chunk_steps, len(action_list))
        for i in range(exec_steps):
            if step >= max_steps:
                break
            action = action_list[i]
            current_qpos = np.array(controller_data["left_arm"]["qpos"], dtype=np.float32).reshape(-1)
            move_data = action_to_move(action, current_qpos)
            robot.move(move_data)
            step += 1
            time.sleep(0.02)

    print("Finished.")
    robot.reset()


if __name__ == "__main__":
    main()
