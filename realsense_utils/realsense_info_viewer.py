# ...existing code...
#!/usr/bin/env python3
"""
realsense_info_viewer.py

功能：
- 查询连接的 RealSense 设备并打印序列号
- 自动选择一个 color 流（选取最高分辨率的 color profile）并打印该分辨率和帧率
- 自动选择一个 depth 流（选取最高分辨率的 depth profile）
- 启动管线，将 color 对齐到 depth，并用 OpenCV 实时可视化对齐后的 RGB 视频（按 q 退出）

依赖：pyrealsense2, opencv-python, numpy

用法示例：
python3 example/collect/realsense_info_viewer.py
# 如果有多个设备可指定序列号：
python3 example/collect/realsense_info_viewer.py --serial 1234567890

"""

import argparse
import sys
import time

try:
    import pyrealsense2 as rs
except Exception:
    print("错误：无法导入 pyrealsense2。请先安装：pip install pyrealsense2")
    # 不在此处退出，以便静态编译检查通过；运行时用户会看到此提示。

import numpy as np
import cv2


def select_color_profile(device, preferred_serial=None):
    """
    在设备的所有传感器中查找 color 视频 profile，返回选中的 (sensor, video_profile)
    策略：选择 color stream 中分辨率最大的 profile（width * height 最大）
    """
    best = None
    for s in device.sensors:
        try:
            profiles = s.get_stream_profiles()
        except Exception:
            continue
        for p in profiles:
            # 仅处理 video profile
            try:
                if p.stream_type() == rs.stream.color:
                    vp = p.as_video_stream_profile()
                    w = vp.width()
                    h = vp.height()
                    fps = int(round(vp.fps()))
                    score = w * h
                    if best is None or score > best[2]:
                        best = (s, vp, score, w, h, fps)
            except Exception:
                # 有些 profile 无法转为 video profile，忽略
                continue
    return best  # None or tuple


def select_depth_profile(device):
    """
    在设备的所有传感器中查找 depth 视频 profile，返回选中的 (sensor, video_profile)
    策略：选择 depth stream 中分辨率最大的 profile（width * height 最大）
    """
    best = None
    for s in device.sensors:
        try:
            profiles = s.get_stream_profiles()
        except Exception:
            continue
        for p in profiles:
            # 仅处理 video profile
            try:
                if p.stream_type() == rs.stream.depth:
                    vp = p.as_video_stream_profile()
                    w = vp.width()
                    h = vp.height()
                    fps = int(round(vp.fps()))
                    score = w * h
                    if best is None or score > best[2]:
                        best = (s, vp, score, w, h, fps)
            except Exception:
                # 有些 profile 无法转为 video profile，忽略
                continue
    return best  # None or tuple


def main():
    parser = argparse.ArgumentParser(description="RealSense info + viewer")
    parser.add_argument("--serial", type=str, default=None, help="指定设备序列号（可选）")
    parser.add_argument("--width", type=int, default=None, help="可选：覆盖 color 分辨率宽")
    parser.add_argument("--height", type=int, default=None, help="可选：覆盖 color 分辨率高")
    parser.add_argument("--fps", type=int, default=None, help="可选：覆盖 color 帧率")
    args = parser.parse_args()

    # 延迟导入 pyrealsense2，便于静态检查时不报错
    try:
        import pyrealsense2 as rs
    except Exception:
        print("运行时错误：未能导入 pyrealsense2。请安装 RealSense SDK 的 Python 包，例如：\n  pip install pyrealsense2")
        sys.exit(1)

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("未检测到任何 RealSense 设备。请连接设备后重试。")
        sys.exit(1)

    # 根据 --serial 选择设备，否则使用第一个设备
    selected_dev = None
    for d in devices:
        try:
            serial = d.get_info(rs.camera_info.serial_number)
        except Exception:
            serial = "unknown"
        if args.serial and serial == args.serial:
            selected_dev = d
            break
    if selected_dev is None:
        selected_dev = devices[0]

    serial = selected_dev.get_info(rs.camera_info.serial_number)
    name = selected_dev.get_info(rs.camera_info.name)
    print(f"检测到设备: name={name}, serial={serial}")

    # 选择 color profile
    choice_color = select_color_profile(selected_dev)
    if choice_color is None:
        print("设备未找到 color 流的 video profile。尝试启用默认 color 流失败。")
        sys.exit(1)

    sensor_color, vp_color, _, w, h, fps = choice_color

    # 选择 depth profile
    choice_depth = select_depth_profile(selected_dev)
    if choice_depth is None:
        print("设备未找到 depth 流的 video profile。尝试启用默认 depth 流失败。")
        sys.exit(1)

    sensor_depth, vp_depth, _, dw, dh, dfps = choice_depth

    # 如果用户指定覆盖参数，则使用用户指定（仅对 color）
    if args.width:
        w = args.width
    if args.height:
        h = args.height
    if args.fps:
        fps = args.fps

    print(f"使用 color 流 -> 分辨率: {w}x{h}, 帧率: {fps} fps")
    print(f"使用 depth 流 -> 分辨率: {dw}x{dh}, 帧率: {dfps} fps")

    # 配置并启动管线
    pipeline = rs.pipeline()
    config = rs.config()
    # 绑定到指定设备序列号，避免多设备混淆
    if serial:
        config.enable_device(serial)

    # 启用 color 流，使用 BGR 格式，以便直接给 OpenCV
    try:
        config.enable_stream(rs.stream.color, int(w), int(h), rs.format.bgr8, int(fps))
    except Exception as e:
        print(f"无法启用 color 流 (w={w}, h={h}, fps={fps})，错误: {e}")
        print("尝试启用 color 流不成功，退出。")
        sys.exit(1)

    # 启用 depth 流
    try:
        config.enable_stream(rs.stream.depth, int(dw), int(dh), rs.format.z16, int(dfps))
    except Exception as e:
        print(f"无法启用 depth 流 (w={dw}, h={dh}, fps={dfps})，错误: {e}")
        print("尝试启用 depth 流不成功，退出。")
        sys.exit(1)

    # 创建对齐对象，将 color 对齐到 depth
    align = rs.align(rs.stream.depth)

    pipeline_profile = None
    try:
        pipeline_profile = pipeline.start(config)
    except Exception as e:
        print(f"启动管线失败: {e}")
        sys.exit(1)

    print("按 'q' 键或关闭窗口以退出")

    window_name = f"RealSense {serial} ({w}x{h}@{fps}) aligned to depth"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # 等待帧
            frames = pipeline.wait_for_frames()
            # 对齐帧
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                # 没拿到帧，重试
                continue
            # 转为 numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # 直接显示对齐到 depth 的 RGB 图像（不叠加额外信息）
            cv2.imshow(window_name, color_image)

            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("收到中断，退出...")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
# ...existing code...