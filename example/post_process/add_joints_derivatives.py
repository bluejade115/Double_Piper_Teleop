import os
import sys
import argparse
import time
import h5py
import numpy as np

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
    更宽松：检测任意 dataset，尝试把含 7 或 6 的轴移动到最后并 reshape 为 (N,J)。
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

        # if already (N,J) and J>=1
        if arr.ndim == 2:
            candidates.append((name, arr))
            return
        # if 1D and length==J (single sample) -> (1,J)
        if arr.ndim == 1:
            candidates.append((name, arr.reshape(1, -1)))
            return
        # higher dim: try move any axis==6 or ==7 to last then reshape
        for idx, s in enumerate(arr.shape):
            if s in (6, 7):
                try:
                    arr2 = np.moveaxis(arr, idx, -1).reshape(-1, s)
                    candidates.append((name, arr2))
                    return
                except Exception:
                    continue
        # otherwise try flatten to (N,-1) if last axis >1
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
    # prefer any candidate with >=2 timesteps and reasonable joints (>=2 cols)
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
    # Prefer same-group timestamp
    parent = joint_name.rsplit("/", 1)[0] if joint_name and "/" in joint_name else ""
    for name, shape in info.items():
        last = name.split("/")[-1].lower()
        same_group = (parent == "" and "/" not in name) or (parent != "" and name.startswith(parent + "/"))
        if same_group and last in preferred and len(shape) == 1 and shape[0] == joints_len:
            return name
    # global preferred
    for name, shape in info.items():
        last = name.split("/")[-1].lower()
        if last in preferred and len(shape) == 1 and shape[0] == joints_len:
            return name
    # fallback any 1D matching length
    for name, shape in info.items():
        if len(shape) == 1 and shape[0] == joints_len:
            return name
    return None

def time_to_seconds(tarr):
    t = np.asarray(tarr, dtype=float)
    if t.size == 0:
        return None
    mx = t.max()
    # likely nanoseconds if very large (>1e12)
    if mx > 1e12:
        return t / 1e9
    # likely microseconds
    if mx > 1e9:
        return t / 1e6
    # if large but not that large, still divide by 1e9 conservatively when values look like ns strings
    return t

def compute_derivatives(joints, time=None):
    """
    joints: (N, J)
    time: 1D array length N in seconds, or None (assume uniform spacing)
    返回 vel, acc shape (N, J)
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

def ensure_group(hf, parent):
    if parent in ("", "/"):
        return hf["/"]
    return hf.require_group(parent)

def write_sibling_dataset(hf, joint_name, base_name, data, attrs, overwrite=False, dry_run=False):
    parent = joint_name.rsplit("/", 1)[0] if "/" in joint_name else ""
    if parent:
        grp = ensure_group(hf, parent)
        exists = base_name in grp
        full = f"{parent}/{base_name}"
    else:
        grp = hf
        exists = base_name in hf
        full = base_name

    if exists:
        if not overwrite:
            print(f"  Dataset {full} exists, skipping (use --overwrite to replace).")
            return
        if dry_run:
            print(f"  Would overwrite {full}")
            return
        del grp[base_name]
        ds = grp.create_dataset(base_name, data=data, compression="gzip")
    else:
        if dry_run:
            print(f"  Would create {full}")
            return
        ds = grp.create_dataset(base_name, data=data, compression="gzip")
    for k, v in attrs.items():
        ds.attrs[k] = v
    print(f"  Wrote dataset: {full}")

def process_file(path, overwrite=False, dry_run=False, retries=10, wait=0.5, debug=False):
    print("Processing:", path)
    try:
        with open_h5_with_retry_impl(path, "r", retries=retries, wait=wait) as hf:
            joint_name, joints = detect_joint_dataset(hf, debug=debug)
            if joint_name is None or joints is None:
                print("  No joint-like dataset found, skip.")
                if debug:
                    for n, s in find_datasets_info(hf).items():
                        print(f"    available: {n} -> {s}")
                return
            joints = np.asarray(joints, dtype=float)
            N, J = joints.shape
            print(f"  Detected joint dataset: {joint_name} shape={joints.shape}")

            if N < 2:
                print("  Too few timesteps, skip.")
                return

            time_name = choose_time_dataset(hf, N, joint_name=joint_name, debug=debug)
            time_arr = None
            if time_name:
                try:
                    raw_t = np.asarray(hf[time_name][()])
                    tsec = time_to_seconds(raw_t)
                    if tsec is not None and tsec.shape[0] == N:
                        time_arr = tsec
                        print(f"  Using time dataset: {time_name} (converted to seconds)")
                    else:
                        print("  Found time dataset but length mismatch, ignoring.")
                except Exception as e:
                    print("  Failed to read time dataset:", e)

        # compute derivatives in memory
        try:
            vel, acc = compute_derivatives(joints, time=time_arr)
        except Exception as e:
            print("  Error computing derivatives:", e)
            return

        # write back to file as siblings
        with open_h5_with_retry_impl(path, "r+", retries=retries, wait=wait) as hf:
            attrs_vel = {"units": "rad/s", "description": "angular velocity per joint", "source": joint_name}
            attrs_acc = {"units": "rad/s^2", "description": "angular acceleration per joint", "source": joint_name}
            write_sibling_dataset(hf, joint_name, "joint_vel", vel, attrs_vel, overwrite=overwrite, dry_run=dry_run)
            write_sibling_dataset(hf, joint_name, "joint_acc", acc, attrs_acc, overwrite=overwrite, dry_run=dry_run)

    except OSError as e:
        print("  Failed to open file:", e)
        print("  如果文件被 HDFView 或其他程序占用，请先关闭它们或增加 --retries/--wait 再试。")

def walk_folder(folder, overwrite=False, dry_run=False, retries=10, wait=0.5, debug=False):
    if os.path.isfile(folder):
        process_file(folder, overwrite=overwrite, dry_run=dry_run, retries=retries, wait=wait, debug=debug)
        return
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith((".h5", ".hdf5", ".hdf")):
                process_file(os.path.join(root, f), overwrite=overwrite, dry_run=dry_run, retries=retries, wait=wait, debug=debug)

def main():
    parser = argparse.ArgumentParser(description="Append per-joint velocity/acceleration to HDF5 (sibling of joint dataset)")
    parser.add_argument("folder", help="hdf5 文件夹或单文件路径")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 joint_vel/joint_acc")
    parser.add_argument("--dry-run", action="store_true", help="不写文件，仅显示操作")
    parser.add_argument("--retries", type=int, default=10, help="打开 hdf5 时重试次数")
    parser.add_argument("--wait", type=float, default=0.5, help="重试间隔秒")
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
    args = parser.parse_args()

    walk_folder(args.folder, overwrite=args.overwrite, dry_run=args.dry_run,
                retries=args.retries, wait=args.wait, debug=args.debug)

if __name__ == "__main__":
    main()