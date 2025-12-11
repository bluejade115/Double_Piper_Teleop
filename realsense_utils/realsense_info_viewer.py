#!/usr/bin/env python3
"""
realsense_info_viewer.py

功能：
- 查询连接的 RealSense 设备并打印序列号
- 自动选择一个 color 流（选取最高分辨率的 color profile）并打印该分辨率和帧率
- 启动管线并用 OpenCV 实时可视化视频（按 q 退出）

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


def main():
    parser = argparse.ArgumentParser(description="RealSense info + viewer")
    parser.add_argument("--serial", type=str, default=None, help="指定设备序列号（可选）")
    parser.add_argument("--width", type=int, default=None, help="可选：覆盖分辨率宽")
    parser.add_argument("--height", type=int, default=None, help="可选：覆盖分辨率高")
    parser.add_argument("--fps", type=int, default=None, help="可选：覆盖帧率")
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
    choice = select_color_profile(selected_dev)
    if choice is None:
        print("设备未找到 color 流的 video profile。尝试启用默认 color 流失败。")
        sys.exit(1)

    sensor, vp, _, w, h, fps = choice

    # 如果用户指定覆盖参数，则使用用户指定
    if args.width:
        w = args.width
    if args.height:
        h = args.height
    if args.fps:
        fps = args.fps

    print(f"使用 color 流 -> 分辨率: {w}x{h}, 帧率: {fps} fps")

    # 配置并启动管线
    pipeline = rs.pipeline()
    config = rs.config()
    # 绑定到指定设备序列号，避免多设备混淆
    if serial:
        config.enable_device(serial)

    # 使用 BGR 格式，以便直接给 OpenCV
    try:
        config.enable_stream(rs.stream.color, int(w), int(h), rs.format.bgr8, int(fps))
    except Exception as e:
        print(f"无法启用 color 流 (w={w}, h={h}, fps={fps})，错误: {e}")
        print("尝试启用 color 流不成功，退出。")
        sys.exit(1)

    pipeline_profile = None
    try:
        pipeline_profile = pipeline.start(config)
    except Exception as e:
        print(f"启动管线失败: {e}")
        sys.exit(1)

    print("按 'q' 键或关闭窗口以退出")

    window_name = f"RealSense {serial} ({w}x{h}@{fps})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # 等待帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                # 没拿到帧，重试
                continue
            # 转为 numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # 在图像上显示一些信息
            overlay = color_image.copy()
            text = f"serial: {serial}  {w}x{h} @{fps}fps"
            cv2.putText(overlay, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, overlay)

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
