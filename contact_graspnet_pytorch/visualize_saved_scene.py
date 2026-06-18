import argparse

import open3d as o3d
import numpy as np
from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps, show_image

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', default='results/predictions_7.npz',
                    help='Path to a predictions .npz saved by inference')
args = parser.parse_args()

# allow_pickle is required: inference saves per-segment dicts inside the .npz.
# Safe here — these files are generated locally by our own inference scripts.
data = np.load(args.results_path, allow_pickle=True)
pred_grasps_cam = data['pred_grasps_cam'].item()
scores = data['scores'].item()
pc_full = data['pc_full']
pc_colors = data['pc_colors']

visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
