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
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def gram_schmidt_ordered(axes_distorted):
    norms = torch.linalg.norm(axes_distorted, dim=1)
    order = torch.argsort(norms, dim=1, descending=True)
    order_reversed = torch.argsort(order, dim=1)
    axes_ordered = torch.gather(axes_distorted, 2, order[:, None, :].expand(-1, 3, -1))
    axes_ordered_ortho = gram_schmidt(axes_ordered)
    axes_ortho = torch.gather(axes_ordered_ortho, 2, order_reversed[:, None, :].expand(-1, 3, -1))
    return axes_ortho


def gram_schmidt(axes_distorted):
    a1, a2, a3 = axes_distorted[:, :, 0], axes_distorted[:, :, 1], axes_distorted[:, :, 2]
    b1 = a1
    proj_a2_b1 = ((a2[:, None, :] @ b1[:, :, None]) / (b1[:, None, :] @ b1[:, :, None]))[..., 0] * b1
    b2 = a2 - proj_a2_b1
    proj_a3_b1 = ((a3[:, None, :] @ b1[:, :, None]) / (b1[:, None, :] @ b1[:, :, None]))[..., 0] * b1
    proj_a3_b2 = ((a3[:, None, :] @ b2[:, :, None]) / (b2[:, None, :] @ b2[:, :, None]))[..., 0] * b2
    b3 = a3 - proj_a3_b1 - proj_a3_b2
    axes_ortho = torch.stack([b1, b2, b3], dim=2)
    return axes_ortho

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

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

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_convariance_matrix(rotation, scaling, scaling_modifier=1.0):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

"""
Creates a sphere of points on the surface of the unit sphere
"""
def fibonacci_sphere(num_samples):
    golden_angle = np.pi * (3 - np.sqrt(5))
    indices = np.arange(num_samples)
    z = 1.0 - 2.0 * indices / (num_samples - 1)
    r = np.sqrt(1.0 - z**2)
    theta = golden_angle * indices
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.stack([x, y, z], axis=-1)
    return points