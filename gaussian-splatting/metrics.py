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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import glob

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(os.path.join(renders_dir, fname))
        gt = Image.open(os.path.join(gt_dir, fname))
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    print("")

    if len(model_paths) == 1:
        pbar = tqdm(model_paths, desc="Evaluating models", leave=True)
    else:
        pbar = model_paths

    for scene_dir in pbar:
        try:
            print("Scene:", scene_dir)
            full_dict = {}
            iter_dirs = [os.path.basename(iter_dir) for iter_dir in glob.glob(os.path.join(scene_dir, "render", "iter_*"))]

            for iter_dir in iter_dirs:
                subset_dirs = [os.path.basename(subset_dir) for subset_dir in glob.glob(os.path.join(scene_dir, "render", iter_dir, "*"))]

                full_dict[iter_dir] = {}

                for subset in subset_dirs:
                    full_dict[iter_dir][subset] = {}
                    test_dir = os.path.join(scene_dir, "render", iter_dir, subset)

                    gt_dir = os.path.join(test_dir, "gt")
                    renders_dir = os.path.join(test_dir, "rgb")
                    renders, gts, image_names = readImages(renders_dir, gt_dir)

                    ssims = []
                    psnrs = []
                    lpipss = []

                    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress", leave=len(model_paths) == 1):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    print("")

                    # Save as json
                    perImage = {
                            name: {
                                "SSIM": ssims[idx].item(),
                                "PSNR": psnrs[idx].item(),
                                "LPIPS": lpipss[idx].item(),
                            } for idx, name in enumerate(image_names)
                        }
                    
                    local_dict = {
                        "SSIM": torch.tensor(ssims).mean().item(),
                        "PSNR": torch.tensor(psnrs).mean().item(),
                        "LPIPS": torch.tensor(lpipss).mean().item(),
                    }

                    with open(os.path.join(test_dir, "results.json"), 'w') as fp:
                        json.dump({ **local_dict, "perImage": perImage}, fp, indent=4)

                    full_dict[iter_dir][subset] = local_dict

            # Save full dict to scene_dir
            with open(os.path.join(scene_dir, "results.json"), 'w') as fp:
                json.dump(full_dict, fp, indent=4)
                    
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print(e)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--parent', '-p', action='store_true', default=False)
    args = parser.parse_args()

    # Find child directories
    output_dirs = []
    if args.parent:
        for parent_path in args.model_paths:
            child_paths = glob.glob(os.path.join(parent_path, "*"))
            for child_path in child_paths:
                # Check if results.json exists
                if not os.path.exists(os.path.join(child_path, "results.json")):
                    output_dirs.append(child_path)
    else:
        output_dirs = args.model_paths
                
    print(f"Evaluating {len(output_dirs)} models")
    evaluate(output_dirs)
