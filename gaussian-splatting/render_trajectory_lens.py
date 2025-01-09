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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import numpy as np

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, num_frames=100, view_start=0, view_end=-1):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    # Get first and last view
    first_view = views[view_start]
    last_view = views[view_end]
    R_first = Rotation.from_matrix(first_view.world_view_transform[:3, :3].cpu().numpy())
    R_last = Rotation.from_matrix(last_view.world_view_transform[:3, :3].cpu().numpy())
    T_first = first_view.world_view_transform[3, :3].cpu().numpy()
    T_last = last_view.world_view_transform[3, :3].cpu().numpy()
    R_sequence = Rotation.from_matrix([first_view.R, last_view.R])

    print(f"R_first = {R_first.as_matrix()!r}")
    print (f"R_last = {R_last.as_matrix()!r}")
    print(f"T_first = {T_first!r}")
    print(f"T_last = {T_last!r}")

    # Create the interpolator object
    slerp = Slerp([0, 1], R_sequence)
    view = views[0]
    distortion_backup = view.distortion_params[3].item()
    distortion_range = (np.cos(np.linspace(0, 2*np.pi, num_frames)) / 2) + 0.5
    distortion_range = distortion_range * (1.0 - 0.6) + 0.6

    interp_range = -np.cos(np.linspace(0, 2*np.pi, num_frames)) / 2 + 0.5
    interp_range *= 0.2

    for idx in tqdm(range(num_frames), desc="Rendering progress"):
        view.distortion_params[3] = distortion_range[idx] * distortion_backup

        R_first = np.array([[-0.92705891,  0.18158915, -0.32800481],
            [ 0.37462881,  0.48289257, -0.79149733],
            [ 0.01466375, -0.8566447 , -0.51569839]])
        R_last = np.array([[-0.99874697, -0.03221657,  0.03829603],
            [-0.04837185,  0.42524006, -0.90378706],
            [ 0.01283191, -0.90450704, -0.4262656 ]])
        T_first = np.array([ 3.226703 , -1.1035808,  4.4786425]) 
        T_last = np.array([3.6878889, 0.4953386, 2.4650903])
        
        R_sequence = Rotation.from_matrix([R_first, R_last])
        slerp = Slerp([0, 1], R_sequence)
        T_current = T_first + (T_last - T_first) * interp_range[idx]
        R_current = slerp(interp_range[idx]).as_matrix()

        view.world_view_transform[:3, :3] = torch.from_numpy(R_current).float().cuda()
        view.world_view_transform[3, :3] = torch.from_numpy(T_current).float().cuda()
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, num_frames=300, view_start=0, view_end=12)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)