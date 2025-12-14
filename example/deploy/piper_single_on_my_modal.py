import sys

sys.path.append("./")
import os
import numpy as np
import torch
import math
from my_robot.agilex_piper_single_base import PiperSingle
# from policy.ACT.inference_model import MYACT
from utils.data_handler import is_enter_pressed
import time
import pdb
#
# {
# 'arm': {'left_arm': { 
#                       'joint': [0.0, 0.85220935, -0.68542569, 0.0, 0.78588684, -0.05256932], 
#                       'qpos': 0.0},
#                       'gripper': 0.0}
#                     },
# 'image': {'cam_head': {'color': array, 'depth': array}, 'cam_wrist': {'color': array, 'depth': array}}
# }
# 
# map = {
#     "observation.images.cam_high": "observation.cam_head.color",
#     "observation.images.cam_left_wrist": "observation.cam_left_wrist.color",
#     "observation.images.cam_right_wrist": "observation.cam_right_wrist.color",
#     "observation.state": [
#         "observation.left_arm.joint",
#         "observation.left_arm.gripper",
#         "observation.right_arm.joint",
#         "observation.right_arm.gripper"
#     ],
# }
def input_transform(data):
    has_left_arm = "left_arm" in data[0]
    has_right_arm = "right_arm" in data[0]
    
    if has_left_arm and not has_right_arm:
        left_joint_dim = len(data[0]["left_arm"]["joint"])
        left_gripper_dim = 1
        
        data[0]["right_arm"] = {
            "joint": [0.0] * left_joint_dim,
            "gripper": [0.0] * left_gripper_dim
        }
        has_right_arm = True
    
    elif has_right_arm and not has_left_arm:
        right_joint_dim = len(data[0]["right_arm"]["joint"])
        right_gripper_dim = 1
        
        # fill left_arm data
        data[0]["left_arm"] = {
            "joint": [0.0] * right_joint_dim,
            "gripper": [0.0] * right_gripper_dim
        }
        has_left_arm = True
    
    elif not has_left_arm and not has_right_arm:
        default_joint_dim = 6
        
        data[0]["left_arm"] = {
            "joint": [0.0] * default_joint_dim,
            "gripper": 0.0
        }
        data[0]["right_arm"] = {
            "joint": [0.0] * default_joint_dim,
            "gripper": 0.0
        }
        has_left_arm = True
        has_right_arm = True
    
    # state = np.concatenate([
    #     np.array(data[0]["left_arm"]["joint"]).reshape(-1),
    #     np.array(data[0]["left_arm"]["gripper"]).reshape(-1),
    #     np.array(data[0]["right_arm"]["joint"]).reshape(-1),
    #     np.array(data[0]["right_arm"]["gripper"]).reshape(-1)
    # ])
    # # print(state)
    # img_arr = data[1]["cam_head"]["color"], data[1]["cam_wrist"]["color"]

    map = {
    # "images.head": data[1]["cam_head"]["color"],
    # "images.wrist": data[1]["cam_wrist"]["color"],
    "state": [
        np.array(data[0]["left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["left_arm"]["gripper"]).reshape(-1),
        np.array(data[0]["right_arm"]["joint"]).reshape(-1),
        np.array(data[0]["right_arm"]["gripper"]).reshape(-1)
    ]
    }
    return map

def output_transform(data):
    joint_limits_rad = [
        (math.radians(-150), math.radians(150)),   # joint1
        (math.radians(0), math.radians(180)),    # joint2
        (math.radians(-170), math.radians(0)),   # joint3
        (math.radians(-100), math.radians(100)),   # joint4
        (math.radians(-70), math.radians(70)),   # joint5
        (math.radians(-120), math.radians(120))    # joint6
        ]
    def clamp(value, min_val, max_val):
        """将值限制在[min_val, max_val]范围内"""
        return max(min_val, min(value, max_val))
    left_joints = [
        clamp(data[0][i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ] #(1,6)
    left_gripper = data[0][6] #(1,1)
    
    move_data = {
        "left_arm":{
            "joint": left_joints,
            "gripper": left_gripper,
        }
    }
    return move_data

if __name__ == "__main__":
    os.environ["INFO_LEVEL"] = "INFO"
    robot = PiperSingle()
    robot.set_up()
    #load model
    #for example:/home/usr/policy/ACT/act_ckpt/act-pick_place_cup/50
    # model = MYACT("/path/your/ckpt/path","act-pick_place_cup")
    max_step = 2000
    num_episode = 10
    for i in range(num_episode):
        step = 0
        # 重置所有信息
        robot.reset()
        # model.reset_obsrvationwindows()
        # model.random_set_language()
        
        # 等待允许执行推理指令, 按enter开始
        is_start = False
        while not is_start:
            if is_enter_pressed():
                is_start = True
                print("start to inference...")
            else:
                print("waiting for start command...")
                time.sleep(1)

        # 开始逐条推理运行
        while step < max_step:
            data = robot.get()
            observation = input_transform(data)
            print(observation)
            # model.update_observation_window(img_arr, state)
            # action = model.get_action()
            action = np.zeros((1,7))  # for test
            move_data = output_transform(action)
            # robot.move({"arm": 
            #                 move_data
            #             })
            step += 1
            # pdb.set_trace()
            time.sleep(1/robot.condition["save_freq"])
            print(f"Episode {i}, Step {step}/{max_step} completed.")

        robot.reset()
        print("finish episode", i)
    robot.reset()
    robot.reset()


