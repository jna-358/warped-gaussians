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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.blender_fisheye_loader import read_extrinsics_blender_fisheye
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.distortion_utils import approx_distortion_poly
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import glob
import json

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    lens_mask: np.array = None
    distortion_params: np.array = None
    fisheye_fov: np.array = None
    ortho_scale: np.float32 = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    lens_mask: Image.Image = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def hashNumpyDict(d):
    hash_list = []
    for key in sorted(d.keys()):
        hash_list.append(hash(key))
        hash_list.append(hash(d[key].tobytes()))
    return hash(tuple(hash_list))

def npz2dict(npz, keys=[]):
    return {key: npz[key] for key in keys}

def approximate_fisheye_blender(intrinsics, fisheye_poly_degree=8):
    coeffs = np.array([intrinsics[f"fisheye_k{i}"] for i in range(5)]) * (-1) # Negative sign added for strage reasons
    print(f"Polynomial coefficients: {coeffs}")

    # Get the sensor size (mm)
    sensor_width_mm = intrinsics["sensor_width_mm"]
    sensor_height_mm = intrinsics["sensor_height_mm"]
    print(f"Sensor size: {sensor_width_mm}mm x {sensor_height_mm}mm")

    # Set the field of view
    fov = np.pi / 2
    focal_length = sensor_width_mm / (2 * np.tan(fov / 2))

    # Get the resolution
    res_x = intrinsics["resolution_x"]
    res_y = intrinsics["resolution_y"]

    # Compute the maximum angle
    poly = np.polynomial.Polynomial(coeffs)
    max_dist = np.sqrt(sensor_width_mm ** 2 + sensor_height_mm ** 2) / 2

    # Fit polynomial to the inverted data theta(r2) -> r2(theta)
    r2 = np.linspace(0, max_dist, 1000)
    theta_in = poly(r2)
    theta_out = np.arctan2(r2, focal_length)

    poly_full = np.polyfit(theta_in, theta_out, fisheye_poly_degree)[::-1]
    theta_out_pred = np.sum(c * theta_in ** i for i, c in enumerate(poly_full)) # custom_polyval(poly_full, theta_in)
    error = np.sqrt(np.mean((theta_out_pred - theta_out) ** 2))
    print(f"Approximation error: {error:.2e} rad")
    
    return {
        **intrinsics,
        "focal_length": focal_length,
        "fov": fov,
        "max_dist": max_dist,
        "rmse": error,
        "distortion_params": poly_full
    }

def readBlenderOrthoCameras(path, white_background):
    image_paths = sorted(glob.glob(os.path.join(path, "image", "*.png")))
    names = sorted([os.path.splitext(os.path.basename(p))[0] for p in image_paths])
    image_data_paths = [(name, os.path.join(path, "image", name + ".png"), os.path.join(path, "metadata", name + ".npz")) for name in names]
    image_data_paths = {
        name: {
            "image_path": os.path.join(path, "image", name + ".png"), 
            "metadata_path":  os.path.join(path, "metadata", name + ".npz")
        } for name in names}
    

    # Extract intrinsics from metadata
    intrinsics_keys = [
        "K",
        "ortho_scale",
        "width",
        "height",
    ]

    intrinsics = {
        name: npz2dict(np.load(image_data_paths[name]["metadata_path"]), keys=intrinsics_keys) for name in names
    }

    # Extract extrinsics from metadata
    extrinsics_keys = [
        "camera_matrix"
    ]

    extrinsics = {
        name: npz2dict(np.load(image_data_paths[name]["metadata_path"]), keys=extrinsics_keys) for name in names
    }

    # Rotation correction matrix
    R_corr = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Load cam_infos
    cam_infos = []

    for idx, name in enumerate(names):
        # Extrinsics
        extrinsic_matrix = extrinsics[name]["camera_matrix"]
        extrinsic_matrix[:3, :3] = extrinsic_matrix[:3, :3] @ R_corr
        extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        R = extrinsic_matrix[:3, :3].T
        T = extrinsic_matrix[:3, 3]

        # Intrinsics
        K = intrinsics[name]["K"]
        ortho_scale = intrinsics[name]["ortho_scale"] / 10.0
        width = intrinsics[name]["width"]
        height = intrinsics[name]["height"]
        fx = width / ortho_scale
        fy = height / ortho_scale
        fov_x = 2 * np.arctan(width / (2 * fx))
        fov_y = 2 * np.arctan(height / (2 * fy))
        uid = idx

        # Load image
        image_path = image_data_paths[name]["image_path"]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=fov_y.item(), FovX=fov_x.item(), image=image,
                              image_path=image_path, image_name=name, width=width.item(),
                              height=height.item(), lens_mask=None, distortion_params=None,
                              fisheye_fov=None, ortho_scale=ortho_scale.item())
        
        cam_infos.append(cam_info)
    
    return cam_infos        


def readBlenderFisheyeCameras(path, fisheye_poly_degree=8):
    image_paths = sorted(glob.glob(os.path.join(path, "image", "*.png")))
    names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    image_data_paths = [(name, os.path.join(path, "image", name + ".png"), os.path.join(path, "metadata", name + ".npz")) for name in names]
    image_data_paths = {
        name: {
            "image_path": os.path.join(path, "image", name + ".png"), 
            "metadata_path":  os.path.join(path, "metadata", name + ".npz")
        } for name in names}

    # Extract intrinsics from metadata
    intrinsics_keys = [
        "K",
        "sensor_width_mm",
        "sensor_height_mm",
        "resolution_x",
        "resolution_y",
        "fisheye_fov",
        "fisheye_lens",
        "fisheye_k0",
        "fisheye_k1",
        "fisheye_k2",
        "fisheye_k3",
        "fisheye_k4"]
    intrinsics = {
        name: npz2dict(np.load(image_data_paths[name]["metadata_path"]), keys=intrinsics_keys) for name in names
    }

    # Extract extrinsics from metadata
    extrinsics_keys = [
        "camera_matrix"]
    extrinsics = {
        name: npz2dict(np.load(image_data_paths[name]["metadata_path"]), keys=extrinsics_keys) for name in names
    }

    # Find distinct intrinsics
    intrinsics_unique = {}
    for name in names:
        h = hashNumpyDict(intrinsics[name])
        if h not in intrinsics_unique:
            intrinsics_unique[h] = intrinsics[name]
    intrinsics_unique = {h: approximate_fisheye_blender(intrinsics, fisheye_poly_degree=fisheye_poly_degree) for h, intrinsics in intrinsics_unique.items()}

    
    R_corr = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Load cam_infos
    cam_infos = []
    for idx, name in enumerate(names):
        # Extrinsics
        extrinsic_matrix = extrinsics[name]["camera_matrix"]
        extrinsic_matrix[:3, :3] = extrinsic_matrix[:3, :3] @ R_corr
        extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        R = extrinsic_matrix[:3, :3].T
        T = extrinsic_matrix[:3, 3]

        # Intrinsics
        intrinsics_hash = hashNumpyDict(intrinsics[name])
        FovY = intrinsics_unique[intrinsics_hash]["fov"]
        FovX = intrinsics_unique[intrinsics_hash]["fov"]
        width = intrinsics_unique[intrinsics_hash]["resolution_x"].item()
        height = intrinsics_unique[intrinsics_hash]["resolution_y"].item()
        uid = idx
        distortion_params = intrinsics_unique[intrinsics_hash]["distortion_params"]
        fisheye_fov = intrinsics_unique[intrinsics_hash]["fisheye_fov"]

        # Load image
        image_path = image_data_paths[name]["image_path"]
        image = Image.open(image_path)

        # Load lens mask (if available)
        lens_mask_path = os.path.join(path, "lens.png")
        lens_mask = Image.open(lens_mask_path) if os.path.exists(lens_mask_path) else None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=name, width=width, 
                              height=height, lens_mask=lens_mask,
                              distortion_params=distortion_params,
                              fisheye_fov=fisheye_fov)
        cam_infos.append(cam_info)
    
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']

    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    if 'red' in [p.name for p in vertices.properties]:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.ones_like(positions)

    if 'nx' in [p.name for p in vertices.properties]:    
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.random.randn(positions.shape[0], 3)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readBlenderOrthoInfo(path, white_background, eval, llffhold=8):
    # Create cam_infos
    cam_infos = readBlenderOrthoCameras(path, white_background)

    # Split into train and test
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Load sparse point cloud
    ply_path = os.path.join(path, "sparse.ply")
    pcd = fetchPly(ply_path)

    # Create scene_info
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info

# sceneLoadTypeCallbacks["BlenderFisheye"](args.source_path, args.white_background, args.eval, fisheye_poly_degree=args.fisheye_poly_degree)
def readBlenderFisheyeInfo(path, background, eval, llffhold=8, fisheye_poly_degree=8):
    # Create cam_infos
    cam_infos = readBlenderFisheyeCameras(path, fisheye_poly_degree=fisheye_poly_degree)

    # Split into train and test
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Load sparse point cloud
    ply_path = os.path.join(path, "sparse.ply")
    pcd = fetchPly(ply_path)

    # Create scene_info
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info

def quat_to_rotm(quats):
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    N = len(w)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
    R[:, 0, 1] = 2 * x * y - 2 * z * w
    R[:, 0, 2] = 2 * x * z + 2 * y * w
    R[:, 1, 0] = 2 * x * y + 2 * z * w
    R[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
    R[:, 1, 2] = 2 * y * z - 2 * x * w
    R[:, 2, 0] = 2 * x * z - 2 * y * w
    R[:, 2, 1] = 2 * y * z + 2 * x * w
    R[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2
    return R

# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
def scannet_images(colmap_dir):
    with open(os.path.join(colmap_dir, "dslr", "colmap", "images.txt")) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if len(line) > 0 and line[0] != "#"]
    lines = lines[::2]

    # Sort by NAME
    lines = sorted(lines, key=lambda x: x.split()[-1])
    filenames = [line.split()[-1 ]for line in lines]

    # Extract quaternions and positions
    quats_list = [[float(p) for p in line.split()[1:5]] for line in lines]
    pos_list = [[float(p) for p in line.split()[5:8]] for line in lines]
    quats = np.array(quats_list)
    pos = np.array(pos_list)

    # Convert to transformation matrices
    Rs = quat_to_rotm(quats)
    Ts = np.zeros((len(lines), 4, 4), dtype=np.float32)
    Ts[:, :3, :3] = Rs
    Ts[:, :3, 3] = pos
    Ts[:, 3, 3] = 1.0

    return Ts, filenames

# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
def scannet_images_dict(colmap_dir):
    with open(os.path.join(colmap_dir, "dslr", "colmap", "images.txt")) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if len(line) > 0 and line[0] != "#"]
    lines = lines[::2]

    # Sort by NAME
    lines = sorted(lines, key=lambda x: x.split()[-1])

    dataDict = {}
    for line in lines:
        image_id = int(line.split()[0])
        quat = [float(p) for p in line.split()[1:5]]
        pos = [float(p) for p in line.split()[5:8]]
        name = line.split()[-1]
        camera_id = int(line.split()[-2])

        # Check if image exists
        if not os.path.exists(os.path.join(colmap_dir, "dslr", "resized_images", name)):
            print(f"Image {name} does not exist!")
            continue

        # Convert to transformation matrices
        R = quat_to_rotm(np.array([quat]))[0]
        T = np.zeros((4, 4), dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = pos

        dataDict[image_id] = {"R": R, "T": T, "name": name, "camera_id": camera_id}

    return dataDict

# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
def scannet_cameras(colmap_dir):
    with open(os.path.join(colmap_dir, "dslr", "colmap", "cameras.txt")) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if len(line) > 0 and line[0] != "#"]
    line_parts = lines[0].split()
    intrinsics = {}
    intrinsics["camera_id"] = int(line_parts[0])
    intrinsics["model"] = line_parts[1]
    intrinsics["width"] = int(line_parts[2])
    intrinsics["height"] = int(line_parts[3])
    intrinsics["fx"] = float(line_parts[4])
    intrinsics["fy"] = float(line_parts[5])
    intrinsics["cx"] = float(line_parts[6])
    intrinsics["cy"] = float(line_parts[7])
    intrinsics["distortion_params"] = np.array(list(map(float, line_parts[8:])))

    return intrinsics


# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
def scannet_cameras_dict(colmap_dir):
    with open(os.path.join(colmap_dir, "dslr", "colmap", "cameras.txt")) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if len(line) > 0 and line[0] != "#"]

    dataDict = {}
    for line in lines:
        line_parts = line.split()
        camera_id = int(line_parts[0])
        camera_model = line_parts[1]
        width = int(line_parts[2])
        height = int(line_parts[3])
        fx = float(line_parts[4])
        fy = float(line_parts[5])
        cx = float(line_parts[6])
        cy = float(line_parts[7])
        distortion_params = np.array(list(map(float, line_parts[8:])))
        dataDict[camera_id] = {
            "model": camera_model,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "distortion_params": distortion_params
        }

    # Approximate using polynomial
    return dataDict                            


def readScannetCameras(path, fisheye_poly_degree=8):
    # Read poses and intrinsics
    images_dict = scannet_images_dict(path)
    cameras_dict = scannet_cameras_dict(path)
    cameras_dict = approx_distortion_poly(cameras_dict, fisheye_poly_degree=fisheye_poly_degree)

    print("Approximation errors for polynomial distortion:")
    for camera_id, camera in cameras_dict.items():
        print(f" - Camera {camera_id}")
        for prop in ["mse", "max_fov_monotonic_deg"]:
            print(f"   - {prop}: {camera[prop]}")

    cam_infos = []
    for image_id, image_data in images_dict.items():
        uid = image_id

        # Extrinsics
        R = image_data["R"].T
        T = image_data["T"][:3, 3]

        # Intrinsics
        cam = cameras_dict[image_data["camera_id"]]
        image_width = cam["width"]
        image_height = cam["height"]
        FovX = 2 * np.arctan(image_width / (2 * cam["fx"]))
        FovY = 2 * np.arctan(image_height / (2 * cam["fy"]))
        image_name = image_data["name"]
        image_path = os.path.join(path, "dslr", "resized_images", image_name)
        image = Image.open(image_path)
        distortion_params = cam["distortion_params"]
        max_fov = cam["max_fov_monotonic"]

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image_width, height=image_height,
                              lens_mask=None, distortion_params=distortion_params, fisheye_fov=max_fov)
        
        cam_infos.append(cam_info)
        
    return cam_infos


def readScannetppInfo(path, images, eval, llffhold=8, fisheye_poly_degree=8):
    # Read cam_infos
    cam_infos_unsorted = readScannetCameras(path, fisheye_poly_degree=fisheye_poly_degree) 
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x: x.image_name)

    # # Read train/test split (NOT USED BY FISHEYE-GS)
    # with open(os.path.join(path, "dslr", "train_test_lists.json")) as f:
    #     train_test_lists = json.load(f)

    # if eval:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_test_lists["train"]]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_test_lists["test"]]
    # else:
    #     train_cam_infos = cam_infos
    #     test_cam_infos = []

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Load sparse point cloud
    ply_path = os.path.join(path, "dslr", "colmap", "points3D.ply")
    pcd = fetchPly(ply_path)

    # Create scene_info
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info



def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "BlenderFisheye": readBlenderFisheyeInfo,
    "ScanNetPP": readScannetppInfo,
    "BlenderOrtho": readBlenderOrthoInfo
}