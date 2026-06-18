"""Headless smoke test for contact_graspnet_pytorch on a bundled test scene.

Runs the same pipeline as inference.py but skips the blocking GUI windows.
Results are saved to results/ for later visualization with visualize_saved_scene.py.
"""
import os
import sys
import time
import argparse

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.append(os.path.join(REPO, 'contact_graspnet_pytorch'))

from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO
from contact_graspnet_pytorch.data import load_available_input_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--np_path', default='test_data/7.npy')
    parser.add_argument('--forward_passes', type=int, default=1)
    parser.add_argument('--arg_configs', nargs='*', type=str, default=[])
    args = parser.parse_args()

    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU ONLY"}')

    global_config = config_utils.load_config(
        'checkpoints/contact_graspnet', batch_size=args.forward_passes,
        arg_configs=args.arg_configs)

    t0 = time.time()
    estimator = GraspEstimator(global_config)
    checkpoint_io = CheckpointIO(
        checkpoint_dir='checkpoints/contact_graspnet/checkpoints',
        model=estimator.model)
    checkpoint_io.load('model.pt')
    print(f'Model loaded in {time.time() - t0:.1f}s')

    segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(
        args.np_path, K=None)
    print(f'Scene: depth {depth.shape}, segments: {np.unique(segmap).astype(int)}')

    t0 = time.time()
    pc_full, pc_segments, pc_colors = estimator.extract_point_clouds(
        depth, cam_K, segmap=segmap, rgb=rgb, z_range=[0.2, 1.8])
    print(f'Point cloud: {pc_full.shape[0]} points ({time.time() - t0:.1f}s)')

    t0 = time.time()
    pred_grasps_cam, scores, contact_pts, _ = estimator.predict_scene_grasps(
        pc_full, pc_segments=pc_segments, local_regions=True,
        filter_grasps=True, forward_passes=args.forward_passes)
    infer_time = time.time() - t0

    print(f'\n=== Inference done in {infer_time:.1f}s ===')
    total = 0
    for seg_id in sorted(pred_grasps_cam.keys()):
        n = len(pred_grasps_cam[seg_id])
        total += n
        best = scores[seg_id].max() if n else float('nan')
        print(f'  segment {int(seg_id):2d}: {n:4d} grasps, best score {best:.3f}')
    print(f'  TOTAL: {total} grasps')

    if torch.cuda.is_available():
        print(f'Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')

    os.makedirs('results', exist_ok=True)
    out = os.path.join('results', 'predictions_' + os.path.basename(args.np_path).replace('npy', 'npz'))
    np.savez(out, pc_full=pc_full, pred_grasps_cam=pred_grasps_cam,
             scores=scores, contact_pts=contact_pts, pc_colors=pc_colors)
    print(f'Saved: {out}')

    # Show best grasp pose for sanity (4x4 transform, camera frame)
    seg_best = max(pred_grasps_cam, key=lambda k: scores[k].max() if len(scores[k]) else -1)
    idx = scores[seg_best].argmax()
    print(f'\nBest grasp (segment {int(seg_best)}, score {scores[seg_best][idx]:.3f}):')
    print(np.array_str(pred_grasps_cam[seg_best][idx], precision=3, suppress_small=True))


if __name__ == '__main__':
    main()
