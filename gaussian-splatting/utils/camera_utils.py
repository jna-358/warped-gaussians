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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    lens_mask = None
    if cam_info.lens_mask is not None:
        lens_mask = PILtoTorch(cam_info.lens_mask, resolution)
        lens_mask = lens_mask[0, ...]

    distortion_params = None
    if cam_info.distortion_params is not None:
        distortion_params = torch.from_numpy(cam_info.distortion_params.copy()).float().cuda()

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  lens_mask=lens_mask, distortion_params=distortion_params, fisheye_fov=cam_info.fisheye_fov, 
                  ortho_scale=cam_info.ortho_scale)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def interpolate_poses(poseA, poseB, alpha):
    poseA_np = poseA.cpu().numpy()
    poseB_np = poseB.cpu().numpy()

    poseA_np = np.linalg.inv(poseA_np)
    poseB_np = np.linalg.inv(poseB_np)

    RA = poseA_np[:3, :3]
    RB = poseB_np[:3, :3]
    TA = poseA_np[3, :3]
    TB = poseB_np[3, :3]

    R_sequence = Rotation.from_matrix([RA, RB])
    slerp = Slerp([0, 1], R_sequence)
    R_current = slerp(alpha).as_matrix()
    T_current = alpha * TB + (1 - alpha) * TA

    tform_current = np.eye(4)
    tform_current[:3, :3] = R_current
    tform_current[3, :3] = T_current

    tform_current = np.linalg.inv(tform_current)

    tform_current_tensor = torch.from_numpy(tform_current).float().cuda()

    return tform_current_tensor