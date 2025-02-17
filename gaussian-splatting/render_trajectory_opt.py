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
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import configparser
from utils.general_utils import build_rotation, rotation_matrix_to_quaternion_batched
from utils.camera_utils import interpolate_poses
import time
import json
import shutil
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

torch.autograd.set_detect_anomaly(True)


def upload_image(image, topic):
    image = Image.fromarray(image)
    image_io = BytesIO()
    image.save(image_io, format='JPEG')
    image_io.seek(0)

    url = "http://127.0.0.1:5000/post_image"
    files = {'file': ('image.jpeg', image_io, 'image/jpeg')}
    data = {'topic': topic}
    try:
        response = requests.post(url, files=files, data=data)

        if response.status_code != 200:
            print(f"Failed to send image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send image: {e}")

def scaling_loss(scaling):
    zero = torch.tensor(0.0, device="cuda")
    sorting_error = torch.max(scaling[:, 1] - scaling[:, 0], zero) \
        + torch.max(scaling[:, 2] - scaling[:, 1], zero)
    return torch.mean(sorting_error)

def concat_images(images):
    # Convert to numpy
    images = [image.cpu().numpy() for image in images]
    images = [np.transpose(image, (1, 2, 0)) for image in images]
    images = [np.clip(image, 0.0, 1.0) for image in images]
    images = [(image * 255).astype(np.uint8) for image in images]

    # Convert grayscale to RGB
    images = [image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) for image in images]

    # Resize to smallest height
    min_height = min([image.shape[0] for image in images])
    images = [cv2.resize(image, (int(image.shape[1] * min_height / image.shape[0]), min_height)) for image in images]

    # Padding
    padding = 10
    stroke = 10
    indices = np.arange(min_height)
    is_white =(((indices // stroke) % 5 != 0) * 255).astype(np.uint8)
    image_padding = np.repeat(is_white[:, None], padding, axis=1)
    image_padding = np.repeat(image_padding[:, :, None], 3, axis=2)
    images = [np.concatenate([image, image_padding], axis=1) if i < len(images) - 1 else image for i, image in enumerate(images)]

    # Concatenate
    image = np.concatenate(images, axis=1)

    return image

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Unshuffle training cameras (for reference rendering)
    train_names = [(i, viewpoint.image_name) for i, viewpoint in enumerate(scene.getTrainCameras().copy())]
    train_names.sort(key=lambda x: x[1])
    unshuffled_indices = [x[0] for x in train_names]

    num_rendered = 0
    num_rendered_max = 300

    for iteration in range(first_iter, opt.iterations + 1):  
        while True:
            config = configparser.ConfigParser()
            config.read("control.ini")
            if len(config.keys()) > 1:
                break
       
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, 
                            gaussians, pipe, bg, 
                            render_coords=False)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 \
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Backprop
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f} Num: {gaussians.get_xyz.shape[0]:.2e}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()


            # # Densification
            # if config["stop-training"]["skip-training"] != "true":
            #     if iteration < opt.densify_until_iter and config["mods"]["densification"] == "true":
            #         # Keep track of max radii in image-space for pruning
            #         gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #         gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #         if iteration == opt.densify_from_iter + 1: # First densification
            #             print("Starting densification")

            #         if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #             size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #             print("Densifying (and pruning)")
            #             gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
            #         if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #             print("Resetting opacity")
            #             gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                num_nan = gaussians.nan_to_zero_grad()
                if num_nan > 0:
                    print(f"Found {num_nan} NaNs in gradients")
                if gaussians.skybox_distance is not None:
                    is_skybox = torch.norm(gaussians.get_xyz, dim=1) > gaussians.skybox_distance * 0.9
                    # print(f"Skybox points: {torch.sum(is_skybox)}")

                gaussians.optimizer.step()

                # Enforce skybox distance 
                if gaussians.skybox_distance is not None:
                    scaling_factor = gaussians.skybox_distance  / torch.norm(gaussians._xyz[is_skybox], dim=1)
                    gaussians._xyz[is_skybox] *= scaling_factor[:, None]

                gaussians.optimizer.zero_grad(set_to_none = True) # gaussians._rotation has grad, but not gaussians.get_rotation

            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            # Render test view
            if iteration % 5 == 0: # True:
                torch.cuda.empty_cache()

                frame_curr = num_rendered
                num_rendered += 1
                
                # R_first = np.array([[ 8.98111615e-01, -4.39767584e-01, -8.27227617e-07],
                #     [-4.87896990e-02, -9.96384791e-02, -9.93826815e-01],
                #     [ 4.37052735e-01,  8.92567446e-01, -1.10942603e-01]])
                # R_last = np.array([[-6.46459269e-01, -7.62948500e-01, -8.14103758e-07],
                #     [-5.32627289e-03,  4.51410883e-03, -9.99975627e-01],
                #     [ 7.62929908e-01, -6.46443508e-01, -6.98185941e-03]])
                # T_first = np.array([17.81942  , 14.989132 ,  1.5306752])
                # T_last = np.array([15.843243 , 22.691687 ,  1.5306752])
                
                # R_sequence = Rotation.from_matrix([R_first, R_last])
                # slerp = Slerp([0, 1], R_sequence)
                # T_current = T_first + (T_last - T_first) * frame_curr / num_rendered_max
                # R_current = slerp(frame_curr / num_rendered_max).as_matrix()
                viewpoint_cam_test = scene.getTrainCameras().copy()[1]

                pose_first = scene.getTrainCameras().copy()[0].world_view_transform
                pose_last = scene.getTrainCameras().copy()[30].world_view_transform
                pose_current = interpolate_poses(pose_first, pose_last, frame_curr / num_rendered_max)

                relative_progress = frame_curr / num_rendered_max
                num_views = len(scene.getTrainCameras().copy())
                current_view_index_float = relative_progress * (num_views - 1)
                offset = current_view_index_float % 1
                current_view_index = int(current_view_index_float)
                next_view_index = current_view_index + 1

                if offset < 1e-6:
                    pose_current = scene.getTrainCameras().copy()[current_view_index].world_view_transform
                else:
                    pose_before = scene.getTrainCameras().copy()[current_view_index].world_view_transform
                    pose_after = scene.getTrainCameras().copy()[next_view_index].world_view_transform
                    pose_current = interpolate_poses(pose_before, pose_after, offset)

                viewpoint_cam_test.world_view_transform = pose_current

                # tform = np.eye(4)
                # tform[:3, :3] = R_current
                # tform[3, :3] = T_current
                # tform = np.linalg.inv(tform)

                # viewpoint_cam_test.world_view_transform = torch.from_numpy(tform).float().cuda()

                # viewpoint_cam_test.world_view_transform[:3, :3] = torch.from_numpy(R_current).float().cuda()
                # viewpoint_cam_test.world_view_transform[3, :3] = torch.from_numpy(T_current).float().cuda()



                render_pkg_fisheye = render(viewpoint_cam_test, gaussians, pipe, background, 
                                            render_coords=False)
                image_test_fisheye = render_pkg_fisheye["render"]
                depth_image = render_pkg_fisheye["depth"]

                depth_image = (depth_image - torch.min(depth_image)) / (torch.max(depth_image) - torch.min(depth_image))

                image_gt = viewpoint_cam_test.original_image.to("cuda")
                image_error = torch.abs(image_test_fisheye - image_gt)
                images = [image_test_fisheye]

                # if config["mods"]["show_gt"] == "true":
                #     images.append(image_gt)
                # if config["mods"]["show_error"] == "true":
                #     images.append(image_error)
                # if config["mods"]["show_depth"] == "true":
                #     images.append(depth_image)
                image_test_all = concat_images(images)

                upload_image(image_test_all, os.path.basename(dataset.source_path) + f"__{dataset.expname}")

                os.makedirs(out_dir:=os.path.join(dataset.model_path, "trajectory_opt"), exist_ok=True)
                torchvision.utils.save_image(image_test_fisheye, os.path.join(out_dir, f"{frame_curr:05d}.png"))

                torch.cuda.empty_cache()
                
                if num_rendered >= num_rendered_max:
                    return

            if iteration == opt.iterations:
                render_dir = os.path.join("render", os.path.basename(dataset.source_path))
                os.makedirs(render_dir, exist_ok=True)
                viewpoint_stack = scene.getTrainCameras().copy()
                for viewpoint_cam in tqdm(viewpoint_stack, desc="Rendering all viewpoints"):
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background, render_coords=False)
                    image = render_pkg["render"]
                    image = torch.clamp(image, 0.0, 1.0)
                    image = (image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    image = Image.fromarray(image)
                    image.save(os.path.join(render_dir, viewpoint_cam.image_name + ".png"))
                
                torch.cuda.empty_cache()


        if opt.dryrun:
            print("Dryrun completed, exiting.")
            return

def prepare_output_and_logger(model, opt, pipe):    
    if not model.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        if model.expname != "":
            model.model_path = os.path.join("./output/", model.expname)
        else:
            model.model_path = os.path.join(".", "output", f"{os.path.basename(model.source_path)}_{unique_str[0:10]}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
