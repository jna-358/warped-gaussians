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

torch.autograd.set_detect_anomaly(True)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
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

    for iteration in range(first_iter, opt.iterations + 1):  
        # print(f"ITERATION {iteration}")
        # torch.cuda.empty_cache() # Look into cuda implementation of gaussians

        # Read config
        
        while True:
            config = configparser.ConfigParser()
            config.read("control.ini")
            if len(config.keys()) > 1:
                break

        if config["stop-training"]["skip-training"] != "true":
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
                                fisheye=True, render_coords=False, 
                                is_test=False)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            lamda_scaling = 0
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            # loss_scaling = scaling_loss(gaussians.get_scaling)
            loss = (1.0 - opt.lambda_dssim) * Ll1 \
                + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) #\
                # + lamda_scaling * loss_scaling
            loss.backward() # Scale not populated with gradient, check covariance matrix

            iter_end.record()

        with torch.no_grad():
            # Progress bar
            if config["stop-training"]["skip-training"] != "true":
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f} Num: {gaussians.get_xyz.shape[0]:.2e}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

            # # Log and save
            if config["stop-training"]["skip-training"] != "true":
                renderKWArgs = {"fisheye": True}
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), renderKWArgs)
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            # Densification
            if config["stop-training"]["skip-training"] != "true":
                if iteration < opt.densify_until_iter and config["mods"]["densification"] == "true":
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration == opt.densify_from_iter + 1: # First densification
                        print("Starting densification")

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        print("Densifying (and pruning)")
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        print("Resetting opacity")
                        gaussians.reset_opacity()

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
            if iteration % int(config["rendering"]["render_interval"]) == 0:
                torch.cuda.empty_cache()
                test_id = int(config["extrinsics"]["pose-id"])
                test_id_unshuffled = unshuffled_indices[test_id]
                render_coords = config["mods"]["show_coords"] == "true"
                viewpoint_cam_test = scene.getTrainCameras().copy()[test_id_unshuffled]
                render_pkg_pinhole = render(viewpoint_cam_test, gaussians, pipe, background, render_coords=render_coords, fisheye=False)
                render_pkg_fisheye = render(viewpoint_cam_test, gaussians, pipe, background, 
                                            render_coords=render_coords, fisheye=True, is_test=True)
                image_test_pinhole = render_pkg_pinhole["render"]
                image_test_fisheye = render_pkg_fisheye["render"]
                depth_image = render_pkg_fisheye["depth"]

                depth_image = (depth_image - torch.min(depth_image)) / (torch.max(depth_image) - torch.min(depth_image))

                image_gt = viewpoint_cam_test.original_image.to("cuda")
                image_error = torch.abs(image_test_fisheye - image_gt)
                images = [image_test_fisheye] # , image_gt, image_both, image_error]

                if config["mods"]["show_original"] == "true":
                    images.append(image_test_pinhole)
                if config["mods"]["show_gt"] == "true":
                    images.append(image_gt)
                if config["mods"]["show_error"] == "true":
                    images.append(image_error)
                if config["mods"]["show_depth"] == "true":
                    images.append(depth_image)
                image_test_all = concat_images(images)

                upload_image(image_test_all, os.path.basename(dataset.source_path))
                torch.cuda.empty_cache()

            if iteration == opt.iterations:
                render_dir = os.path.join("render", os.path.basename(dataset.source_path))
                os.makedirs(render_dir, exist_ok=True)
                viewpoint_stack = scene.getTrainCameras().copy()
                for viewpoint_cam in tqdm(viewpoint_stack, desc="Rendering all viewpoints"):
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background, fisheye=True, render_coords=False)
                    image = render_pkg["render"]
                    image = torch.clamp(image, 0.0, 1.0)
                    image = (image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    image = Image.fromarray(image)
                    image.save(os.path.join(render_dir, viewpoint_cam.image_name + ".png"))
                
                torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", f"{os.path.basename(args.source_path)}_{unique_str[0:10]}")
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, renderKWArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, **renderKWArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
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
