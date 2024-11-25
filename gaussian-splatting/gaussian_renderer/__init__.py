#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getProjectionMatrix
import numpy as np
import scipy.spatial.transform
import configparser
from utils.diff_utils import jacobian_all
import itertools
from utils.general_utils import build_convariance_matrix, gram_schmidt, gram_schmidt_ordered

def load_extrinsics_from_config(config):
    translation = np.array([float(config["extrinsics"]["tx"]), float(config["extrinsics"]["ty"]), float(config["extrinsics"]["tz"])])
    angles = np.array([float(config["extrinsics"]["rx"]), float(config["extrinsics"]["ry"]), float(config["extrinsics"]["rz"])])
    R = scipy.spatial.transform.Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R
    extrinsics[3, :3] = translation
    extrinsics = np.linalg.inv(extrinsics)
    return torch.from_numpy(extrinsics).float().cuda()

# Function to convert a 3x3 rotation matrix to a quaternion (w, x, y, z)
def rotation_matrix_to_quaternion(R):
    # Ensure R is a proper rotation matrix of shape (3, 3)
    if R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix.")

    # Allocate space for the quaternion
    q = torch.empty(4, device=R.device, dtype=R.dtype)

    # Compute the trace of the matrix
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = torch.sqrt(tr + 1.0) * 2  # S=4*qw
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S

    return q

# Function to convert a 3x3 rotation matrix to a quaternion (w, x, y, z) (batched)
def rotation_matrix_to_quaternion_batched(rotation_matrices):
    """
    Converts a batch of 3x3 rotation matrices to quaternions.

    :param rotation_matrices: Tensor of shape (N, 3, 3) representing N 3x3 rotation matrices.
    :return: Tensor of shape (N, 4) representing N quaternions (w, x, y, z).
    """
    assert rotation_matrices.shape[-2:] == (3, 3), "Input should be a batch of 3x3 matrices."
    
    # Ensure all calculations happen on the same device as the input tensor
    device = rotation_matrices.device
    dtype = rotation_matrices.dtype
    
    # Pre-allocate quaternion tensor on the correct device and dtype
    N = rotation_matrices.shape[0]
    quaternions = torch.zeros((N, 4), device=device, dtype=dtype)

    # Extract rotation matrix elements
    R = rotation_matrices
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Compute the trace-based branch
    cond = t > 0
    S = torch.sqrt(t[cond] + 1.0).to(device) * 2  # S = 4 * qw
    quaternions[cond, 0] = 0.25 * S
    quaternions[cond, 1] = (R[cond, 2, 1] - R[cond, 1, 2]) / S
    quaternions[cond, 2] = (R[cond, 0, 2] - R[cond, 2, 0]) / S
    quaternions[cond, 3] = (R[cond, 1, 0] - R[cond, 0, 1]) / S

    # Compute the largest diagonal element branch
    cond = ~cond
    r_max = torch.argmax(R[cond].diagonal(dim1=-2, dim2=-1), dim=-1)
    S_max = torch.sqrt(1.0 + 2.0 * R[cond, r_max, r_max] - t[cond]).to(device) * 2
    idx = torch.arange(N, device=device)[cond]
    
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        is_i = (r_max == i)
        quaternions[idx[is_i], i+1] = 0.25 * S_max[is_i]
        quaternions[idx[is_i], 0] = (R[idx[is_i], k, j] - R[idx[is_i], j, k]) / S_max[is_i]
        quaternions[idx[is_i], j+1] = (R[idx[is_i], j, i] + R[idx[is_i], i, j]) / S_max[is_i]
        quaternions[idx[is_i], k+1] = (R[idx[is_i], k, i] + R[idx[is_i], i, k]) / S_max[is_i]

    return quaternions
    

# Function to convert a quaternion (w, x, y, z) to a 3x3 rotation matrix (batched)
def quaternion_to_rotation_matrix(qs):
    w, x, y, z = qs[:, 0], qs[:, 1], qs[:, 2], qs[:, 3]
    N = qs.shape[0]

    # Compute the rotation matrices
    rotms = torch.zeros((N, 3, 3), device=qs.device, dtype=qs.dtype)
    rotms[:, 0, 0] = 1 - 2*y*y - 2*z*z
    rotms[:, 0, 1] = 2*x*y - 2*z*w
    rotms[:, 0, 2] = 2*x*z + 2*y*w
    rotms[:, 1, 0] = 2*x*y + 2*z*w
    rotms[:, 1, 1] = 1 - 2*x*x - 2*z*z
    rotms[:, 1, 2] = 2*y*z - 2*x*w
    rotms[:, 2, 0] = 2*x*z - 2*y*w
    rotms[:, 2, 1] = 2*y*z + 2*x*w
    rotms[:, 2, 2] = 1 - 2*x*x - 2*y*y

    return rotms


# Compute the product of a set of quaternions with a single quaternion (w, x, y, z)
def quaternion_multiply(q1, q2, left_multiply=False):
    if left_multiply:
        # Left multiplication q1 * q2
        w = q1[:, 0]*q2[0] - q1[:, 1]*q2[1] - q1[:, 2]*q2[2] - q1[:, 3]*q2[3]
        x = q1[:, 0]*q2[1] + q1[:, 1]*q2[0] + q1[:, 2]*q2[3] - q1[:, 3]*q2[2]
        y = q1[:, 0]*q2[2] - q1[:, 1]*q2[3] + q1[:, 2]*q2[0] + q1[:, 3]*q2[1]
        z = q1[:, 0]*q2[3] + q1[:, 1]*q2[2] - q1[:, 2]*q2[1] + q1[:, 3]*q2[0]
    else:
        # Right multiplication q2 * q1
        w = q2[0]*q1[:, 0] - q2[1]*q1[:, 1] - q2[2]*q1[:, 2] - q2[3]*q1[:, 3]
        x = q2[0]*q1[:, 1] + q2[1]*q1[:, 0] + q2[2]*q1[:, 3] - q2[3]*q1[:, 2]
        y = q2[0]*q1[:, 2] - q2[1]*q1[:, 3] + q2[2]*q1[:, 0] + q2[3]*q1[:, 1]
        z = q2[0]*q1[:, 3] + q2[1]*q1[:, 2] - q2[2]*q1[:, 1] + q2[3]*q1[:, 0]

    return torch.stack([w, x, y, z], dim=1)

# Compute the product of a two sets of quaternions
def quaternion_multiply_batch(q1, q2):
    w = q1[:, 0]*q2[:, 0] - q1[:, 1]*q2[:, 1] - q1[:, 2]*q2[:, 2] - q1[:, 3]*q2[:, 3]
    x = q1[:, 0]*q2[:, 1] + q1[:, 1]*q2[:, 0] + q1[:, 2]*q2[:, 3] - q1[:, 3]*q2[:, 2]
    y = q1[:, 0]*q2[:, 2] - q1[:, 1]*q2[:, 3] + q1[:, 2]*q2[:, 0] + q1[:, 3]*q2[:, 1]
    z = q1[:, 0]*q2[:, 3] + q1[:, 1]*q2[:, 2] - q1[:, 2]*q2[:, 1] + q1[:, 3]*q2[:, 0]

def get_axis_pc_np(axis=0, resolution=21, size=11, scaling=0.1, z_offset=0.0):
    # Create properties of the point cloud
    num_points = (12 * resolution)
    means3D = np.zeros((num_points, 3))
    scales = np.ones((num_points, 3)) * 0.005
    rotations = np.zeros((num_points, 4))
    rotations[:, 0] = 1.0
    shs = np.zeros((num_points, 16, 3))
    opacity = np.ones((num_points, 1))
    
    # Top edge (y_min)
    means3D[:resolution, 0] = np.linspace(-size/2, size/2, resolution)
    means3D[:resolution, 1] = -size/2
    scales[:resolution, 0] = scaling
    shs[:resolution, 0, 0] = 2.0

    # Right edge (x_max)
    means3D[resolution:2*resolution, 1] = np.linspace(-size/2, size/2, resolution)
    means3D[resolution:2*resolution, 0] = size/2
    scales[resolution:2*resolution, 1] = scaling
    shs[resolution:2*resolution, 0, 1] = 2.0

    # Bottom edge (y_max)
    means3D[2*resolution:3*resolution, 0] = np.linspace(size/2, -size/2, resolution)
    means3D[2*resolution:3*resolution, 1] = size/2
    scales[2*resolution:3*resolution, 0] = scaling
    shs[2*resolution:3*resolution, 0, 0] = 2.0

    # Left edge (x_min)
    means3D[3*resolution:4*resolution, 1] = np.linspace(size/2, -size/2, resolution)
    means3D[3*resolution:4*resolution, 0] = -size/2
    scales[3*resolution:4*resolution, 1] = scaling
    shs[3*resolution:4*resolution, 0, 1] = 2.0

    # Top edge (y_min) (back)
    means3D[4*resolution:5*resolution, 0] = np.linspace(-size/2, size/2, resolution)
    means3D[4*resolution:5*resolution, 1] = -size/2
    means3D[4*resolution:5*resolution, 2] = size
    scales[4*resolution:5*resolution, 0] = scaling
    shs[4*resolution:5*resolution, 0, 0] = 2.0

    # Right edge (x_max) (back)
    means3D[5*resolution:6*resolution, 1] = np.linspace(-size/2, size/2, resolution)
    means3D[5*resolution:6*resolution, 0] = size/2
    means3D[5*resolution:6*resolution, 2] = size
    scales[5*resolution:6*resolution, 1] = scaling
    shs[5*resolution:6*resolution, 0, 1] = 2.0

    # Bottom edge (y_max) (back)
    means3D[6*resolution:7*resolution, 0] = np.linspace(size/2, -size/2, resolution)
    means3D[6*resolution:7*resolution, 1] = size/2
    means3D[6*resolution:7*resolution, 2] = size
    scales[6*resolution:7*resolution, 0] = scaling
    shs[6*resolution:7*resolution, 0, 0] = 2.0

    # Left edge (x_min) (back)
    means3D[7*resolution:8*resolution, 1] = np.linspace(size/2, -size/2, resolution)
    means3D[7*resolution:8*resolution, 0] = -size/2
    means3D[7*resolution:8*resolution, 2] = size
    scales[7*resolution:8*resolution, 1] = scaling
    shs[7*resolution:8*resolution, 0, 1] = 2.0

    # Top left edge (x_min, y_min)
    means3D[8*resolution:9*resolution, 0] = -size/2
    means3D[8*resolution:9*resolution, 1] = -size/2
    means3D[8*resolution:9*resolution, 2] = np.linspace(0, size, resolution)
    scales[8*resolution:9*resolution, 2] = scaling
    shs[8*resolution:9*resolution, 0, 2] = 2.0

    # Top right edge (x_max, y_min)
    means3D[9*resolution:10*resolution, 0] = size/2
    means3D[9*resolution:10*resolution, 1] = -size/2
    means3D[9*resolution:10*resolution, 2] = np.linspace(0, size, resolution)
    scales[9*resolution:10*resolution, 2] = scaling
    shs[9*resolution:10*resolution, 0, 2] = 2.0

    # Bottom right edge (x_max, y_max)
    means3D[10*resolution:11*resolution, 0] = size/2
    means3D[10*resolution:11*resolution, 1] = size/2
    means3D[10*resolution:11*resolution, 2] = np.linspace(0, size, resolution)
    scales[10*resolution:11*resolution, 2] = scaling
    shs[10*resolution:11*resolution, 0, 2] = 2.0

    # Bottom left edge (x_min, y_max)
    means3D[11*resolution:, 0] = -size/2
    means3D[11*resolution:, 1] = size/2
    means3D[11*resolution:, 2] = np.linspace(0, size, resolution)
    scales[11*resolution:, 2] = scaling
    shs[11*resolution:, 0, 2] = 2.0

    # Apply z offset
    means3D[:, 2] += z_offset

    return means3D, opacity, scales, rotations, shs

def get_square_pc_np(resolution=100, size=2):
     # Create properties of the point cloud
    num_points = (4 * resolution) - 4
    means3D = np.zeros((num_points, 3))
    scales = np.ones((num_points, 3)) * 0.005
    rotations = np.zeros((num_points, 4))
    rotations[:, 0] = 1.0
    shs = np.zeros((num_points, 16, 3))
    shs[:, 0, 2] = 2.0
    opacity = np.ones((num_points, 1))

    # Top edge (y_min)
    means3D[:resolution-1, 0] = np.linspace(-size/2, size/2, resolution-1, endpoint=False)
    means3D[:resolution-1, 1] = -size/2

    # Right edge (x_max)
    means3D[resolution-1:2*resolution-2, 1] = np.linspace(-size/2, size/2, resolution-1, endpoint=False)
    means3D[resolution-1:2*resolution-2, 0] = size/2

    # Bottom edge (y_max)
    means3D[2*resolution-2:3*resolution-3, 0] = np.linspace(size/2, -size/2, resolution-1, endpoint=False)
    means3D[2*resolution-2:3*resolution-3, 1] = size/2

    # Left edge (x_min)
    means3D[3*resolution-3:, 1] = np.linspace(size/2, -size/2, resolution-1, endpoint=False)
    means3D[3*resolution-3:, 0] = -size/2

    return means3D, opacity, scales, rotations, shs

def get_cube_pc_np(resolution=100, size=2):
     # Create properties of the point cloud
    num_points = (12 * resolution)
    means3D = np.zeros((num_points, 3))
    scales = np.ones((num_points, 3)) * 0.005
    rotations = np.zeros((num_points, 4))
    rotations[:, 0] = 1.0
    shs = np.zeros((num_points, 16, 3))
    shs[:, 0, :] = 2.0
    opacity = np.ones((num_points, 1)) * 0.2

    edges = [(main_axis, s1, s2) for main_axis in range(3) for s1 in [-1, 1] for s2 in [-1, 1]]

    for iEdge, (main_axis, s1, s2) in enumerate(edges):
        other_axes = [o for o in range(3) if o != main_axis]
        means3D[iEdge*resolution:(iEdge+1)*resolution, main_axis] = np.linspace(-size/2, size/2, resolution)
        means3D[iEdge*resolution:(iEdge+1)*resolution, other_axes[0]] = s1 * size/2
        means3D[iEdge*resolution:(iEdge+1)*resolution, other_axes[1]] = s2 * size/2

    return means3D, opacity, scales, rotations, shs

def np_to_torch(args):
    return (torch.from_numpy(arg).float().cuda() for arg in args)

def merge_pc(*pointclouds):
    num_props = len(pointclouds[0])
    concatted = [np.concatenate([pc[i] for pc in pointclouds], axis=0) for i in range(num_props)]
    return tuple(concatted)


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
           scaling_modifier = 1.0, render_coords = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Read config
    while True:
        config = configparser.ConfigParser()
        config.read("control.ini")
        if len(config.keys()) > 1:
            break


    # White background for coordinate rendering
    if render_coords:
        bg_color = torch.ones(3).float().cuda()
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if not render_coords:
        means3D = pc.get_xyz
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
    else:
        pc_edges = get_axis_pc_np(axis=0, resolution=int(config["mods"]["edge_count"]), size=2, scaling=float(config["mods"]["edge_length"]), z_offset=-1)
        pc_cube = get_cube_pc_np(resolution=100, size=2)
        means3D, opacity, scales, rotations, shs = np_to_torch(merge_pc(pc_edges, pc_cube))
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points

    # Set up rasterization configuration
    FOVX = viewpoint_camera.FoVx
    FOVY = viewpoint_camera.FoVy
    tanfovx = math.tan(FOVX * 0.5)
    tanfovy = math.tan(FOVY * 0.5)

    
    RESOLUTION_X = viewpoint_camera.image_width
    RESOLUTION_Y = viewpoint_camera.image_height

    K = getProjectionMatrix(
        znear=float(config["render-distance"]["znear"]), 
        zfar=float(config["render-distance"]["zfar"]), 
        fovX=FOVX, fovY=FOVY).transpose(0,1).cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=RESOLUTION_Y,
        image_width=RESOLUTION_X,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=torch.eye(4).cuda(),
        projmatrix=K,
        sh_degree=pc.active_sh_degree,
        campos=torch.zeros(3).cuda(),
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Apply camera pose
    if config["extrinsics"]["override"] == "true":
        camera_extrinsics = load_extrinsics_from_config(config)
    else:
        camera_extrinsics = viewpoint_camera.world_view_transform.float().cuda()

    R = camera_extrinsics[:3, :3].T
    t = camera_extrinsics[3, :3]

    world_view_quat = rotation_matrix_to_quaternion(R)

    means3D = (means3D @ R.T) + t
    rotations = quaternion_multiply(rotations, world_view_quat, left_multiply=False)

    # Geometric distortion based on fisheye lens

    poly_coeffs = viewpoint_camera.distortion_params
    
    if not pipe.jacobians_off:
        # Compute jacobians
        jacobians = jacobian_all(means3D, poly_coeffs)

        # Check for NaN and inf values
        assert not torch.any(torch.isnan(jacobians)), "NaN values in jacobians"
        assert not torch.any(torch.isinf(jacobians)), "Inf values in jacobians"

    # Convert to spherical coordinates
    radius = torch.norm(means3D, dim=1)
    theta = torch.arccos(means3D[:, 2] / radius)
    phi = torch.atan2(means3D[:, 1], means3D[:, 0])

    # Remove points outside the field of view
    fov_fisheye = viewpoint_camera.fisheye_fov

    if fov_fisheye:
        removal_mask = (theta < fov_fisheye / 2)
        opacity = opacity * removal_mask[:, None].float()

    # Apply distortion 
    theta_mod = torch.zeros_like(theta, device=theta.device)
    for i, coeff in enumerate(poly_coeffs):
        theta_mod += coeff * theta ** i

    # Convert back to cartesian coordinates
    x_res = torch.sin(theta_mod) * torch.cos(phi) * radius
    y_res = torch.sin(theta_mod) * torch.sin(phi) * radius
    z_res = torch.cos(theta_mod) * radius
    means3D = torch.stack([x_res, y_res, z_res], dim=1)

    if not pipe.jacobians_off:
        # Compute initial axes
        rotations_mat = quaternion_to_rotation_matrix(rotations)
        axes = torch.diag_embed(scales)
        axes = rotations_mat @ axes

        # Distort axes using jacobians
        axes_distorted = jacobians @ axes

        # Orthogonalize via Gram-Schmidt
        axes_ortho = gram_schmidt_ordered(axes_distorted)

        # Seperate into scale and rotation
        scales_distorted = torch.norm(axes_ortho, dim=1)
        scales_mat_inv = torch.diag_embed(1. / scales_distorted)
        rotmat = axes_ortho @ (scales_mat_inv)

        # Convert back to quaternion
        rot_quat = rotation_matrix_to_quaternion_batched(rotmat)

        # Update scales and rotations
        scales = scales_distorted
        rotations = rot_quat

    
    # Precompute covariance matrices
    cov3D = build_convariance_matrix(rotations, scales, scaling_modifier=scaling_modifier)

    kwargs = {
        "means3D": means3D,
        "means2D": means2D,
        "shs": shs,
        "colors_precomp": None,
        "opacities": opacity,
        "scales": None,
        "rotations": None,
        "cov3D_precomp": cov3D
    }


    for key in kwargs.keys():
        if kwargs[key] is not None:
            assert not torch.any(torch.isnan(kwargs[key])), f"NaN values in {key}"

    rendered_image, radii, depth = rasterizer(**kwargs)

    if viewpoint_camera.lens_mask is not None:
        rendered_image *= viewpoint_camera.lens_mask[None, ...]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}