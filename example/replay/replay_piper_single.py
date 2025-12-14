import argparse
import time
import os
import h5py
import numpy as np

from my_robot.agilex_piper_single_base import PiperSingle

def print_h5_tree(h5obj, prefix=""):
    for k in h5obj:
        obj = h5obj[k]
        path = f"{prefix}/{k}" if prefix else k
        if isinstance(obj, h5py.Dataset):
            print(f"D: {path}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"G: {path}")
            print_h5_tree(obj, path)

def find_datasets(h5f):
    ds_paths = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            ds_paths.append(name)
    h5f.visititems(visitor)
    joint = None
    gripper = None
    time_ds = None
    # prefer datasets that include 'joint' and 'gripper' in name
    for name in ds_paths:
        lname = name.lower()
        if "joint" in lname and joint is None:
            joint = name
        if "gripper" in lname and gripper is None:
            gripper = name
        if ("time" in lname or "timestamp" in lname) and time_ds is None:
            time_ds = name
    return joint, gripper, time_ds, ds_paths

def load_array(h5f, path):
    arr = h5f[path][()]
    # ensure numpy array
    return np.asarray(arr)

def replay_file(hdf5_path, can_name="piper_slave", rate=30.0, compare=False):
    # init controller
    p = PiperSingle()
    p.set_can_name(can_name)
    p.set_up()
    p.reset()

    with h5py.File(hdf5_path, "r") as f:
        print("HDF5 content:")
        print_h5_tree(f)
        joint_ds, grip_ds, time_ds, all_ds = find_datasets(f)
        if joint_ds is None or grip_ds is None:
            raise RuntimeError(f"未找到 joint/gripper 数据集。候选数据集: {all_ds}")
        joints = load_array(f, joint_ds)
        grippers = load_array(f, grip_ds)

        # try to align first dimension
        n = min(getattr(joints, "shape", (len(joints),))[0], getattr(grippers, "shape", (len(grippers),))[0])
        joints = joints[:n]
        grippers = grippers[:n]

        period = 1.0 / float(rate) if rate > 0 else 0.0
        print(f"Replaying {n} frames at ~{rate}Hz (period {period:.4f}s)")

        try:
            for i in range(n):
                joint_frame = np.asarray(joints[i]).astype(float).flatten()
                grip_val = float(np.asarray(grippers[i]).squeeze())

                # # 安全限制：每个 joint 绝对值不得超过 0.1（临时限制）
                # clipped = np.clip(joint_frame, -0.1, 0.1)
                # if not np.allclose(clipped, joint_frame):
                #     print(f"[WARN] frame {i}: joint values clipped to ±0.1")
                # joint_frame = clipped

                move = {"joint": [float(x) for x in joint_frame], "gripper": float(grip_val)}

                # 根据 example/teleop 中的调用格式，发送 move 到 left_arm
                p.move({"arm": {"left_arm": move}})

                if compare:
                    try:
                        sensor = p.get()
                        # 打印简要对比信息
                        print(f"[{i}] sent joint={np.round(joint_frame,3).tolist()} gripper={grip_val}")
                        print(f"     sensor sample: {sensor}")
                    except Exception as e:
                        print(f"[{i}] sent (compare failed: {e})")
                # sleep 到下一帧
                if period > 0:
                    time.sleep(period)
        except KeyboardInterrupt:
            print("Replay interrupted by user.")
    print("Replay finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 hdf5 回放动作到 PiperSingle")
    parser.add_argument("hdf5", nargs="?", default=os.path.join(os.path.dirname(__file__), "../../datasets/test/0.hdf5"),
                        help="hdf5 文件路径（默认 datasets/test/0.hdf5）")
    parser.add_argument("--can", default="can0", help="CAN 名称，传给 PiperSingle.set_can_name")
    parser.add_argument("--rate", type=float, default=2.0, help="回放频率 (Hz)")
    parser.add_argument("--compare", action="store_true", help="同时读取当前控制器返回的传感器数据并打印，便于对比")
    args = parser.parse_args()

    replay_file(os.path.abspath(args.hdf5), can_name=args.can, rate=args.rate, compare=args.compare)