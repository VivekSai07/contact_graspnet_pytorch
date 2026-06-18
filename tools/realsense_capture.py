#!/usr/bin/env python3
"""Capture an aligned RGB-D frame from an Intel RealSense D455(F) and save it
in the .npy dict format expected by contact_graspnet_pytorch's
`load_available_input_data` (keys: 'rgb', 'depth', 'K', and optionally 'seg').

Segmentation (optional, --segment) is a simple, model-free table-plane
removal: the dominant plane (the table) is RANSAC-fit in the point cloud,
everything above it is treated as foreground, and connected components in
the foreground mask become individual object segments (ids 1..N). This is
enough to exercise --local_regions --filter_grasps on a real scene without
pulling in a segmentation network.

Usage
-----
python tools/realsense_capture.py --out test_data/realsense_scene.npy
python tools/realsense_capture.py --out test_data/realsense_scene.npy --segment
"""
import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def capture_frame(width=640, height=480, fps=30, warmup_frames=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    try:
        for _ in range(warmup_frames):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        rgb = np.asanyarray(color_frame.get_data())  # BGR8, matches load_available_input_data's cv2 path

        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                       [0, intr.fy, intr.ppy],
                       [0, 0, 1]], dtype=np.float32)
    finally:
        pipeline.stop()

    return rgb, depth, K


def segment_table_top(depth, K, z_range=(0.2, 1.5), table_thresh=0.012, min_area=400):
    """Model-free segmentation: RANSAC table plane removal + connected components."""
    import open3d as o3d

    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    valid = (z > z_range[0]) & (z < z_range[1])

    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[valid].reshape(-1, 3))

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=table_thresh, ransac_n=3, num_iterations=1000)

    a, b, c, d = plane_model
    dist = np.abs(a * pts[..., 0] + b * pts[..., 1] + c * pts[..., 2] + d)
    above_table = valid & (dist > table_thresh)

    mask = (above_table.astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    seg = np.zeros((h, w), dtype=np.uint8)
    next_id = 1
    for label in range(1, num_labels):  # 0 = background
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            seg[labels == label] = next_id
            next_id += 1

    print(f'Found table plane {plane_model.round(3)}, {next_id - 1} object segment(s)')
    return seg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='test_data/realsense_scene.npy')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--segment', action='store_true',
                         help='Add a model-free table-plane-removal segmentation map')
    args = parser.parse_args()

    print('Connecting to RealSense D455F...')
    rgb, depth, K = capture_frame(args.width, args.height)
    print(f'Captured rgb {rgb.shape}, depth {depth.shape}, K=\n{K}')

    data = {'rgb': rgb, 'depth': depth, 'K': K}

    if args.segment:
        data['seg'] = segment_table_top(depth, K)

    np.save(args.out, data)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
