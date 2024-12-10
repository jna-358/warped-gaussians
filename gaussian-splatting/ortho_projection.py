import numpy as np
import cv2
import glob
import os
from plyfile import PlyData

DATA_DIR = "/data/blender-cycles/scenes/lego"
VIEW_ID = 25

# Load image and metadata
image_paths = glob.glob(os.path.join(DATA_DIR, "image", "*.png"))
image_paths.sort()
image_path = image_paths[VIEW_ID]
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
metadata = np.load(os.path.join(DATA_DIR, "metadata", os.path.basename(image_path).replace(".png", ".npz")))
ortho_scale = metadata["ortho_scale"] / 10.0

# Load extrinsics
R_corr = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
extrinsic_matrix = metadata["camera_matrix"]
extrinsic_matrix[:3, :3] = extrinsic_matrix[:3, :3] @ R_corr
extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
R = extrinsic_matrix[:3, :3].T
T = extrinsic_matrix[:3, 3]

# Load point cloud
pc_path = os.path.join(DATA_DIR, "sparse.ply")
plydata = PlyData.read(pc_path)
vertices = plydata['vertex']
positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

# Create intrinsic matrix for resolution 512x512 and fov 90
resolution = 1024
fov = 45
focal_length = 0.5 * resolution / np.tan(np.radians(fov) / 2)
intrinsic_matrix = np.array([[focal_length, 0, resolution/2], [0, focal_length, resolution/2], [0, 0, 1]])

# Transform points according to camera extrinsics
positions = positions @ R + T

# Check that all points are in front of the camera
assert np.all(positions[:, 2] > 0), "Some points are behind the camera"

# # Project points to image plane
# positions = positions @ intrinsic_matrix.T
# positions = positions / positions[:, 2, None]
# positions = positions[:, :2]

# Orthographic projection
positions = positions[:, :2] * (resolution / ortho_scale) + resolution / 2

# Filter points outside image plane
valid = (positions[:, 0] >= 0) & (positions[:, 0] < resolution) & (positions[:, 1] >= 0) & (positions[:, 1] < resolution)
positions = positions[valid]

# Draw points on image
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
for i in range(positions.shape[0]):
    x, y = positions[i].astype(int)
    image_rgb[y, x] = [0, 255, 0]

# Display image
cv2.imwrite("output.png", image_rgb)



