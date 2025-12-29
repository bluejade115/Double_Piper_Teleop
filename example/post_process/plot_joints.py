# ...existing code...
import os
import sys
import argparse
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

def open_h5_with_retry_impl(path, mode="r+", retries=10, wait=0.5):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return h5py.File(path, mode)
        except OSError as e:
            last_exc = e
            msg = str(e).lower()
            errno_val = getattr(e, "errno", None)
            if errno_val == 11 or "unable to lock" in msg or "resource temporarily unavailable" in msg:
                if attempt < retries:
                    print(f"  File locked, retry {attempt}/{retries} after {wait}s...")
                    time.sleep(wait)
                    continue
            raise
    raise last_exc

def find_datasets_info(hf):
    info = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            info[name] = obj.shape
    hf.visititems(visitor)
    return info

def detect_joint_dataset(hf, debug=False):
    """
    返回 (dataset_name, ndarray_of_shape_(N,J))
    更宽松：检测任意 dataset，尝试把含 6 或 7 的轴移动到最后并 reshape 为 (N,J)。
    优先选择名字末尾为 joint/joints/qpos 的 dataset。
    """
    preferred = {"joint", "joints", "qpos", "joint_positions", "joint_pos"}
    candidates = []

    def visitor(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        try:
            raw = np.asarray(obj[()])
        except Exception:
            return
        arr = np.squeeze(raw)
        if arr.size == 0:
            return

        # if already (N,J)
        if arr.ndim == 2:
            candidates.append((name, arr))
            return
        # if 1D -> treat as (1, L)
        if arr.ndim == 1:
            candidates.append((name, arr.reshape(1, -1)))
            return
        # try move any axis==6 or ==7 to last and flatten front axes
        for idx, s in enumerate(arr.shape):
            if s in (6, 7):
                try:
                    arr2 = np.moveaxis(arr, idx, -1).reshape(-1, s)
                    candidates.append((name, arr2))
                    return
                except Exception:
                    continue
        # fallback: try reshape to (-1, last)
        try:
            arr2 = arr.reshape(-1, arr.shape[-1])
            if arr2.shape[1] >= 1:
                candidates.append((name, arr2))
        except Exception:
            pass

    hf.visititems(visitor)

    if debug:
        print("  detect candidates:")
        for n, a in candidates:
            print(f"    {n} -> inferred {a.shape}")

    # prefer name match with >=2 timesteps
    for name, arr in candidates:
        last = name.split("/")[-1].lower()
        if last in preferred and arr.shape[0] >= 2:
            return name, arr
    # prefer any candidate with >=2 timesteps
    for name, arr in candidates:
        if arr.shape[0] >= 2 and arr.shape[1] >= 1:
            return name, arr
    # fallback first candidate
    if candidates:
        return candidates[0][0], candidates[0][1]
    return None, None

def choose_time_dataset(hf, joints_len, joint_name=None, debug=False):
    preferred = {"timestamp", "timestamps", "time", "t"}
    info = find_datasets_info(hf)
    parent = joint_name.rsplit("/", 1)[0] if joint_name and "/" in joint_name else ""
    for name, shape in info.items():
        last = name.split("/")[-1].lower()
        same_group = (parent == "" and "/" not in name) or (parent != "" and name.startswith(parent + "/"))
        if same_group and last in preferred and len(shape) == 1 and shape[0] == joints_len:
            return name
    for name, shape in info.items():
        last = name.split("/")[-1].lower()
        if last in preferred and len(shape) == 1 and shape[0] == joints_len:
            return name
    for name, shape in info.items():
        if len(shape) == 1 and shape[0] == joints_len:
            return name
    return None

def time_to_seconds(tarr):
    t = np.asarray(tarr, dtype=float)
    if t.size == 0:
        return None
    mx = t.max()
    # nanoseconds
    if mx > 1e12:
        return t / 1e9
    # microseconds
    if mx > 1e9:
        return t / 1e6
    return t

def compute_derivatives(joints, time=None):
    """
    joints: (N, J)
    time: 1D seconds array or None
    返回 vel, acc (N, J)
    """
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 2 or joints.shape[0] < 2:
        raise ValueError("joints must be (N, J) with N>=2")
    if time is None:
        vel = np.gradient(joints, axis=0)
        acc = np.gradient(vel, axis=0)
    else:
        t = np.asarray(time, dtype=float)
        if t.ndim != 1 or t.shape[0] != joints.shape[0]:
            raise ValueError("time length must match joints length")
        vel = np.gradient(joints, t, axis=0)
        acc = np.gradient(vel, t, axis=0)
    return vel, acc

def load_datasets(path, dataset=None, recompute=False, debug=False):
    """
    从 hdf5 读取 joint (必需)，优先读取并返回文件内 joint_vel/joint_acc，
    当 recompute=True 或者文件内缺少时，在内存用 numpy.gradient 重新计算。
    返回 joints, vel, acc, time_seconds_or_None, dataset_name
    """
    with open_h5_with_retry_impl(path, "r") as hf:
        if dataset is None:
            dataset, _ = detect_joint_dataset(hf, debug=debug)
            if dataset is None:
                raise RuntimeError("无法自动找到 joint dataset，使用 --dataset 指定")
        joints = np.asarray(hf[dataset][()]).squeeze()
        if joints.ndim == 1:
            joints = joints.reshape(1, -1)
        n, j = joints.shape

        base_parent = dataset.rsplit("/",1)[0] if "/" in dataset else ""
        def sibling(name):
            return f"{base_parent}/{name}" if base_parent else name

        vel = None
        acc = None
        if not recompute and sibling("joint_vel") in hf:
            try:
                vel = np.asarray(hf[sibling("joint_vel")][()]).squeeze()
            except Exception:
                vel = None
        if not recompute and sibling("joint_acc") in hf:
            try:
                acc = np.asarray(hf[sibling("joint_acc")][()]).squeeze()
            except Exception:
                acc = None

        time = None
        time_name = choose_time_dataset(hf, n, joint_name=dataset, debug=debug)
        if time_name:
            try:
                raw_t = np.asarray(hf[time_name][()])
                tsec = time_to_seconds(raw_t)
                if tsec is not None and tsec.shape[0] == n:
                    time = tsec
                else:
                    if debug:
                        print("  time found but length mismatch, ignoring")
            except Exception:
                if debug:
                    print("  failed to read time dataset")

    # ensure shapes: if vel/acc are 1D or single-sample expand to (N,J)
    if vel is not None:
        vel = np.asarray(vel, dtype=float)
        if vel.ndim == 1 and j == 1:
            vel = vel.reshape(-1, 1)
        if vel.ndim == 1 and vel.shape[0] == j:
            # single-sample vector -> broadcast
            vel = np.tile(vel.reshape(1, -1), (n, 1))
    if acc is not None:
        acc = np.asarray(acc, dtype=float)
        if acc.ndim == 1 and j == 1:
            acc = acc.reshape(-1, 1)
        if acc.ndim == 1 and acc.shape[0] == j:
            acc = np.tile(acc.reshape(1, -1), (n, 1))

    if vel is None or acc is None or recompute:
        vel_calc, acc_calc = compute_derivatives(joints, time=time)
        if vel is None or recompute:
            vel = vel_calc
        if acc is None or recompute:
            acc = acc_calc

    return joints, vel, acc, time, dataset

def plot_all(joints, vel, acc, time=None, dataset_name="joint", save=None, show=True):
    n, j = joints.shape
    x = time if (time is not None and len(time) == n) else np.arange(n)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    cmap = plt.cm.get_cmap("tab10")
    for k in range(j):
        c = cmap(k % 10)
        axes[0].plot(x, joints[:, k], label=f"q{k}", color=c)
        axes[1].plot(x, vel[:, k], label=f"q{k}", color=c)
        axes[2].plot(x, acc[:, k], label=f"q{k}", color=c)
    axes[0].set_ylabel("position")
    axes[1].set_ylabel("velocity")
    axes[2].set_ylabel("acceleration")
    axes[2].set_xlabel("time (s)" if time is not None else "frame")
    axes[0].legend(ncol=min(j,6), fontsize="small", loc="upper right")
    axes[1].legend(ncol=min(j,6), fontsize="small", loc="upper right")
    axes[2].legend(ncol=min(j,6), fontsize="small", loc="upper right")
    fig.suptitle(f"Joint traces: {dataset_name}")
    fig.tight_layout(rect=[0,0,1,0.96])
    if save:
        fig.savefig(save, dpi=200)
        print("Saved plot to", save)
    if show:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot joint / joint_vel / joint_acc from HDF5")
    parser.add_argument("file", help="hdf5 文件路径")
    parser.add_argument("--dataset", help="指定 joint dataset 路径（例如 left_arm/joint）")
    parser.add_argument("--save", help="保存图像到文件")
    parser.add_argument("--no-show", action="store_true", help="不弹窗显示")
    parser.add_argument("--debug", action="store_true", help="调试信息")
    parser.add_argument("--recompute", action="store_true", help="忽略文件内已有 joint_vel/joint_acc，重新计算并绘图")
    parser.add_argument("--retries", type=int, default=3, help="打开 hdf5 时重试次数")
    parser.add_argument("--wait", type=float, default=0.2, help="重试间隔秒")
    args = parser.parse_args()

    joints, vel, acc, time_arr, dsname = load_datasets(args.file, dataset=args.dataset, recompute=args.recompute, debug=args.debug)
    plot_all(joints, vel, acc, time=time_arr, dataset_name=dsname, save=args.save, show=not args.no_show)

if __name__ == "__main__":
    main()
# ...existing code...