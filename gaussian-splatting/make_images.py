import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob
import json
import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.offsetbox import AnchoredText
import argparse

plt.rcParams["font.family"] = "Times New Roman"

filetypes = ["pdf", "png", "svg"]

pretty_names = {
    "archiviz": "Archiviz",
    "barbershop": "Barbershop",
    "classroom": "Classroom",
    "monk": "Monk",
    "pabellon": "Pabellon",
    "sky": "Sky",
    "bedroom": "Bedroom",
    "kitchen": "Kitchen",
    "office_day": "Office Day",
    "office_night": "Office Night",
    "tool_room": "Tool Room",
    "utility_room": "Utility Room",
    "utility": "Utility Room",
    "ours": "Ours",
    "fisheyegs": "Fisheye-GS",
    "bathtub": "Bathtub",
    "conference_room": "Conference Room",
    "electrical_room": "Electrical Room",
    "hotel": "Hotel",
    "plant": "Plant",
    "printer": "Printer",
}

def render_scannet(results_dir):
    # Load images
    choices = {
        "bedroom": 8,
        "kitchen": 0,
        "office_day": 3,
        "office_night": 22,
        "tool_room": 26,
        "utility_room": 3,
    }

    rois = {
        # "bedroom": [(278, 561, 218, 197), (1437, 853, 163, 164)],
        # "office_day": [(66, 60, 379, 557)],
        # "tool_room": [(963, 11, 626, 561)],
        # "utility_room": [(938, 181, 371, 359)],
    }

    scenes = sorted(list(choices.keys()))
    
    data = {}
    for scene in scenes:
        # Find our result directory
        ours_result_dirs = glob.glob(os.path.join(results_dir, f"scannet_{scene}")) # f"./output/first_run/scannet_{scene}_*")
        assert len(ours_result_dirs) == 1
        ours_result_dir = ours_result_dirs[0]

        # Find fisheyegs result directory
        fisheyegs_result_dirs = glob.glob(os.path.join(results_dir, f"fisheyegs_{scene}")) # f"./output/first_run/fisheyegs_{scene}")
        assert len(fisheyegs_result_dirs) == 1
        fisheyegs_result_dir = fisheyegs_result_dirs[0]

        # Find image paths
        iter_dirs_ours = glob.glob(os.path.join(os.path.join(ours_result_dir, "render"), "iter_*"))
        iter_dir_ours = sorted(iter_dirs_ours, key=lambda x: int(os.path.basename(x).split("_")[-1]))[-1]
        image_path_ours = sorted(glob.glob(os.path.join(iter_dir_ours, "test", "rgb", "*.png")))[choices[scene]]
        gt_path = os.path.join(iter_dir_ours, "test", "gt", os.path.basename(image_path_ours))
        iter_dirs_fisheyegs = glob.glob(os.path.join(os.path.join(fisheyegs_result_dir, "test"), "ours_*"))
        iter_dir_fisheyegs = sorted(iter_dirs_fisheyegs, key=lambda x: int(os.path.basename(x).split("_")[-1]))[-1]
        image_path_fisheyegs = sorted(glob.glob(os.path.join(iter_dir_fisheyegs, "renders", "*.png")))[choices[scene]]

        # Load per-image psnr
        psnr_ours = json.load(open(os.path.join(iter_dir_ours, "test", "results.json")))["perImage"][os.path.basename(image_path_ours)]["PSNR"]
        psnr_fisheyegs = json.load(open(os.path.join(fisheyegs_result_dir, "per_view.json")))[os.path.basename(iter_dir_fisheyegs)]["PSNR"][f"{choices[scene]:05d}.png"]

        data[scene] = {
            "ours": {
                "path": image_path_ours,
                "psnr": psnr_ours,
            },
            "fisheyegs": {
                "path": image_path_fisheyegs,
                "psnr": psnr_fisheyegs,
            },
            "gt": {
                "path": gt_path,
            }
        }

    
    # Plot images as grid
    for subimage in range(2):
        scenes_local = scenes[:3] if subimage == 0 else scenes[3:]

        rows = 3
        cols = len(scenes_local)
        fig = plt.figure(figsize=(cols*3, rows*3))
        grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.1, label_mode="all")

        for idx, scene in enumerate(scenes_local):
            # Load images
            image_ours = Image.open(data[scene]["ours"]["path"])
            image_fisheyegs = Image.open(data[scene]["fisheyegs"]["path"])
            image_gt = Image.open(data[scene]["gt"]["path"])

            # Downsample images
            downscaling_factor = 0.5
            image_ours = image_ours.resize((int(image_ours.width*downscaling_factor), int(image_ours.height*downscaling_factor)), Image.LANCZOS)
            image_fisheyegs = image_fisheyegs.resize((int(image_fisheyegs.width*downscaling_factor), int(image_fisheyegs.height*downscaling_factor)), Image.LANCZOS)
            image_gt = image_gt.resize((int(image_gt.width*downscaling_factor), int(image_gt.height*downscaling_factor)), Image.LANCZOS)

            roi = None
            if scene in rois:
                roi = [[int(r[i] * downscaling_factor) for i in range(4)] for r in rois[scene]]

            # Plot images
            grid[2*cols+idx].imshow(image_ours, interpolation="none")
            # grid[0*cols+idx].set_title(pretty_names[scene])
            grid[1*cols+idx].imshow(image_fisheyegs, interpolation="none")
            grid[0*cols+idx].imshow(image_gt, interpolation="none")

            if roi:
                for r in roi:
                    for i in range(3):
                        grid[i*cols+idx].add_patch(plt.Rectangle((r[0], r[1]), r[2], r[3], edgecolor="red", facecolor="none", linewidth=1))


            # Add row labels
            if idx == 0:
                grid[2*cols+idx].set_ylabel("Ours")
                grid[1*cols+idx].set_ylabel("Fisheye-GS")
                grid[0*cols+idx].set_ylabel("Ground Truth")

            grid[0*cols+idx].set_xlabel(pretty_names[scene])
            grid[0*cols+idx].xaxis.set_label_position('top')

            # Hide axes (except for x and y labels)
            for i in range(3):
                grid[i*cols+idx].set_xticks([])
                grid[i*cols+idx].set_yticks([])
                grid[i*cols+idx].tick_params(axis='both', which='both', length=0)

                # Hide spines
                grid[i*cols+idx].spines['top'].set_visible(False)
                grid[i*cols+idx].spines['right'].set_visible(False)
                grid[i*cols+idx].spines['bottom'].set_visible(False)
                grid[i*cols+idx].spines['left'].set_visible(False)

                if i == 1:
                    psnr = data[scene]["fisheyegs"]["psnr"]
                elif i == 2:
                    psnr = data[scene]["ours"]["psnr"]
                else:
                    psnr = None

                if psnr:
                    grid[i*cols+idx].text(
                        0.97, 0.95, f"{psnr:.1f}dB", 
                        transform=grid[i*cols+idx].transAxes, 
                        fontsize=10, 
                        color="white", 
                        ha="right", 
                        va="top", 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none")
                    )

        output_dir = os.path.join("figures", os.path.basename(results_dir))
        os.makedirs(output_dir, exist_ok=True)
        for filetype in filetypes:
            plt.savefig(os.path.join(output_dir, f"scannet_{subimage+1}.{filetype}"), bbox_inches="tight")


"""
Plots two images of the blender scenes. The first row contains ground truth images, the second row contains our results.
"""
def render_blender(results_dir):
    choices = {
        "archiviz": 81,
        "barbershop": 41,
        "classroom": 1,
        "monk": 1,
        "pabellon": 33,
        "sky": 57,
    }

    rois = {
        "barbershop": [(418, 350, 292, 148)],
        "pabellon": [(656, 568, 354, 442)],
        "monk": [(579, 818, 328, 197)],
    }

    scenes = sorted(list(choices.keys()))
    
    data = {}
    for scene in scenes:
        # Find our result directory
        ours_result_dirs = glob.glob(os.path.join(results_dir, f"blender_{scene}_*")) # f"./output/first_run/blender_{scene}_*")
        assert len(ours_result_dirs) == 1
        ours_result_dir = ours_result_dirs[0]

        # Find image paths
        iter_dirs_ours = glob.glob(os.path.join(os.path.join(ours_result_dir, "render"), "iter_*"))
        iter_dir_ours = sorted(iter_dirs_ours, key=lambda x: int(os.path.basename(x).split("_")[-1]))[-1]
        image_path_ours = os.path.join(iter_dir_ours, "test", "rgb", f"frame{choices[scene]:04d}.png")
        gt_path = os.path.join(iter_dir_ours, "test", "gt", f"frame{choices[scene]:04d}.png")

        # Load per-image psnr
        psnr_ours = json.load(open(os.path.join(iter_dir_ours, "test", "results.json")))["perImage"][os.path.basename(image_path_ours)]["PSNR"]

        data[scene] = {
            "ours": {
                "path": image_path_ours,
                "psnr": psnr_ours,
            },

            "gt": {
                "path": gt_path,
            }
        }

    for subimage in range(2):
        scenes_local = scenes[:3] if subimage == 0 else scenes[3:]

        rows = 2
        cols = len(scenes_local)
        fig = plt.figure(figsize=(cols*3, rows*3))
        grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.1, label_mode="all")

        for idx, scene in enumerate(scenes_local):
            # Load images
            image_ours = Image.open(data[scene]["ours"]["path"])
            image_gt = Image.open(data[scene]["gt"]["path"])

            # Downsample images
            downscaling_factor = 0.5
            image_ours = image_ours.resize((int(image_ours.width*downscaling_factor), int(image_ours.height*downscaling_factor)), Image.LANCZOS)
            image_gt = image_gt.resize((int(image_gt.width*downscaling_factor), int(image_gt.height*downscaling_factor)), Image.LANCZOS)

            roi = None
            if scene in rois:
                roi = [[int(r[i] * downscaling_factor) for i in range(4)] for r in rois[scene]]

            # Plot images
            grid[1*cols+idx].imshow(image_ours, interpolation="none")
            grid[0*cols+idx].imshow(image_gt, interpolation="none")

            if roi:
                for r in roi:
                    for irow in range(rows):
                        grid[irow*cols+idx].add_patch(plt.Rectangle((r[0], r[1]), r[2], r[3], edgecolor="red", facecolor="none", linewidth=1))


            # Add row labels
            if idx == 0:
                grid[1*cols+idx].set_ylabel("Ours")
                grid[0*cols+idx].set_ylabel("Ground Truth")

            grid[0*cols+idx].set_xlabel(pretty_names[scene])
            grid[0*cols+idx].xaxis.set_label_position('top')

            # Hide axes (except for x and y labels)
            for i in range(2):
                grid[i*cols+idx].set_xticks([])
                grid[i*cols+idx].set_yticks([])
                grid[i*cols+idx].tick_params(axis='both', which='both', length=0)

                # Hide spines
                grid[i*cols+idx].spines['top'].set_visible(False)
                grid[i*cols+idx].spines['right'].set_visible(False)
                grid[i*cols+idx].spines['bottom'].set_visible(False)
                grid[i*cols+idx].spines['left'].set_visible(False)

                if i == 1:
                    psnr = data[scene]["ours"]["psnr"]
                else:
                    psnr = None
                
                if psnr:
                    grid[i*cols+idx].text(
                        0.97, 0.95, f"{psnr:.1f}dB", 
                        transform=grid[i*cols+idx].transAxes, 
                        fontsize=10, 
                        color="white", 
                        ha="right", 
                        va="top", 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none")
                    )
        output_dir = os.path.join("figures", os.path.basename(results_dir))
        os.makedirs(output_dir, exist_ok=True)
        for filetype in filetypes:
            plt.savefig(os.path.join(output_dir, f"blender_{subimage+1}.{filetype}"), bbox_inches="tight")


def render_skybox(results_dir):
    choice = 73
    rois = [(603, 58, 292, 383)]

    # Find result directories
    result_dirs = glob.glob(os.path.join(results_dir, f"skybox_*")) # glob.glob("./output/first_run/skybox_*")

    # Parse model namespace
    skybox_on = [eval("argparse."+open(os.path.join(d, "cfg_model")).read()).skybox for d in result_dirs]
    assert sum(skybox_on) == 1
    skybox_on = skybox_on.index(True)
    skybox_off = 1 - skybox_on

    result_dir_on = result_dirs[skybox_on]
    result_dir_off = result_dirs[skybox_off]

    iter_dir = sorted([os.path.basename(dir) for dir in glob.glob(os.path.join(result_dir_on, "render", "iter_*"))], key=lambda x: int(x.split("_")[-1]))[-1]

    image_path_on = os.path.join(result_dir_on, "render", iter_dir, "test", "rgb", f"frame{choice:04d}.png")
    image_path_off = os.path.join(result_dir_off, "render", iter_dir, "test", "rgb", f"frame{choice:04d}.png")
    image_path_gt = os.path.join(result_dir_on, "render", iter_dir, "test", "gt", f"frame{choice:04d}.png")
    depth_path_on = os.path.join(result_dir_on, "render", iter_dir, "test", "depth", f"frame{choice:04d}.png")
    depth_path_off = os.path.join(result_dir_off, "render", iter_dir, "test", "depth", f"frame{choice:04d}.png")

    image_on = Image.open(image_path_on)
    image_off = Image.open(image_path_off)
    image_gt = Image.open(image_path_gt)
    depth_on = Image.open(depth_path_on)
    depth_off = Image.open(depth_path_off)

    downscaling_factor = 0.5
    image_on = image_on.resize((int(image_on.width*downscaling_factor), int(image_on.height*downscaling_factor)), Image.LANCZOS)
    image_off = image_off.resize((int(image_off.width*downscaling_factor), int(image_off.height*downscaling_factor)), Image.LANCZOS)
    image_gt = image_gt.resize((int(image_gt.width*downscaling_factor), int(image_gt.height*downscaling_factor)), Image.LANCZOS)
    depth_on = depth_on.resize((int(depth_on.width*downscaling_factor), int(depth_on.height*downscaling_factor)), Image.LANCZOS)
    depth_off = depth_off.resize((int(depth_off.width*downscaling_factor), int(depth_off.height*downscaling_factor)), Image.LANCZOS)

    rois = [[int(r[i] * downscaling_factor) for i in range(4)] for r in rois]

    results_on = json.load(open(os.path.join(result_dir_on, "render", iter_dir, "test", "results.json")))
    results_off = json.load(open(os.path.join(result_dir_off, "render", iter_dir, "test", "results.json")))

    psnr_on = results_on["perImage"][f"frame{choice:04d}.png"]["PSNR"]
    psnr_off = results_off["perImage"][f"frame{choice:04d}.png"]["PSNR"]

    # Plot
    rows = 2
    cols = 3
    fig = plt.figure(figsize=(cols*3, rows*3))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.1, label_mode="all")

    grid[0*cols+0].imshow(image_gt, interpolation="none")
    grid[0*cols+1].imshow(image_on, interpolation="none")
    grid[0*cols+2].imshow(image_off, interpolation="none")

    grid[1*cols+1].imshow(depth_on, interpolation="none")
    grid[1*cols+2].imshow(depth_off, interpolation="none")

    # Plot ROIs
    for r in rois:
        for i in range(3):
            grid[i].add_patch(plt.Rectangle((r[0], r[1]), r[2], r[3], edgecolor="red", facecolor="none", linewidth=1))

    # Add column labels
    grid[0*cols+0].set_xlabel("Ground Truth")
    grid[0*cols+1].set_xlabel("Skybox Enabled")
    grid[0*cols+2].set_xlabel("Skybox Disabled")
    grid[0*cols+0].xaxis.set_label_position('top')
    grid[0*cols+1].xaxis.set_label_position('top')
    grid[0*cols+2].xaxis.set_label_position('top')

    # Add row labels
    grid[0*cols+0].set_ylabel("Color")
    grid[1*cols+1].set_ylabel("Depth")


    # Hide axes (except for x and y labels)
    for irow in range(rows):
        for icol in range(cols):
            grid[irow*cols+icol].set_xticks([])
            grid[irow*cols+icol].set_yticks([])
            grid[irow*cols+icol].tick_params(axis='both', which='both', length=0)

            # Hide spines
            grid[irow*cols+icol].spines['top'].set_visible(False)
            grid[irow*cols+icol].spines['right'].set_visible(False)
            grid[irow*cols+icol].spines['bottom'].set_visible(False)
            grid[irow*cols+icol].spines['left'].set_visible(False)

    # Add PSNR values
    args = [(1, psnr_on), (2, psnr_off)]
    for i, psnr in args:
        grid[0*cols+i].text(
            0.97, 0.95, f"{psnr:.1f}dB", 
            transform=grid[0*cols+i].transAxes, 
            fontsize=10, 
            color="white", 
            ha="right", 
            va="top", 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none")
        )

    # Save figure
    output_dir = os.path.join("figures", os.path.basename(results_dir))
    os.makedirs(output_dir, exist_ok=True)
    for filetype in filetypes:
        plt.savefig(os.path.join(output_dir, f"skybox.{filetype}"), bbox_inches="tight")

def render_ortho(results_dir):
    choices = [1, 5, 8]
    rois = []
    num_images = len(choices)

    # Load images
    iter_dirs = glob.glob(os.path.join(results_dir, "ortho_lego", "render", "iter_*"))
    iter_dir = sorted(iter_dirs, key=lambda x: int(os.path.basename(x).split("_")[-1]))[-1]
    image_paths = glob.glob(os.path.join(iter_dir, "test", "rgb", "*.png"))
    image_names = [os.path.basename(p).split(".")[0] for p in image_paths]
    image_names = sorted(image_names)
    image_names = [image_names[i] for i in choices]

    data = []
    for image_name in image_names:
        # Load prediction
        prediction = Image.open(os.path.join(iter_dir, "test", "rgb", f"{image_name}.png"))

        # Load ground truth
        gt = Image.open(os.path.join(iter_dir, "test", "gt", f"{image_name}.png"))

        # Load depth
        depth = Image.open(os.path.join(iter_dir, "test", "depth", f"{image_name}.png"))

        # Load psnr
        psnr = json.load(open(os.path.join(iter_dir, "test", "results.json")))["perImage"][f"{image_name}.png"]["PSNR"]

        data.append({
            "prediction": prediction,
            "gt": gt,
            "depth": depth,
            "psnr": psnr,
        })
    
    # Plot as 3 x num_images grid, top row is ground truth, middle is ours rgb and bottom is ours depth
    rows = 3
    cols = num_images
    fig = plt.figure(figsize=(cols*3, rows*3))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.1, label_mode="all")

    for idx, d in enumerate(data):
        grid[0*cols+idx].imshow(d["gt"], interpolation="none")
        grid[1*cols+idx].imshow(d["prediction"], interpolation="none")
        grid[2*cols+idx].imshow(d["depth"], interpolation="none")


        # Add row labels
        if idx == 0:
            grid[0*cols+idx].set_ylabel("Ground Truth")
            grid[1*cols+idx].set_ylabel("Ours (Color)")
            grid[2*cols+idx].set_ylabel("Ours (Depth)")

        # Hide axes (except for x and y labels)
        for irow in range(rows):
            grid[irow*cols+idx].set_xticks([])
            grid[irow*cols+idx].set_yticks([])
            grid[irow*cols+idx].tick_params(axis='both', which='both', length=0)

            # Hide spines
            grid[irow*cols+idx].spines['top'].set_visible(False)
            grid[irow*cols+idx].spines['right'].set_visible(False)
            grid[irow*cols+idx].spines['bottom'].set_visible(False)
            grid[irow*cols+idx].spines['left'].set_visible(False)

        # Add PSNR values
        grid[1*cols+idx].text(
            0.97, 0.95, f"{d['psnr']:.1f}dB", 
            transform=grid[1*cols+idx].transAxes, 
            fontsize=10, 
            color="white", 
            ha="right", 
            va="top", 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none")
        )

    # Save figure
    output_dir = os.path.join("figures", os.path.basename(results_dir))
    os.makedirs(output_dir, exist_ok=True)

    for filetype in filetypes:
        plt.savefig(os.path.join(output_dir, f"ortho.{filetype}"), bbox_inches="tight")

def render_blender_mirror(results_dir):
    roi = (468, 315, 267, 219)
    image_id = 25

    # Convert roi from cv2 to pil format
    roi = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])

    # Find result directories
    result_dirs = glob.glob(os.path.join(results_dir, "blender_barbershop_*"))
    assert len(result_dirs) == 1, "Expected exactly one result directory"
    result_dir = result_dirs[0]
    iter_dirs = glob.glob(os.path.join(result_dir, "render", "iter_*"))
    iter_dir = sorted(iter_dirs, key=lambda x: int(os.path.basename(x).split("_")[-1]))[-1]

    # Load images
    rgb = Image.open(os.path.join(iter_dir, "test", "rgb", f"frame{image_id:04d}.png"))
    gt = Image.open(os.path.join(iter_dir, "test", "gt", f"frame{image_id:04d}.png"))
    depth = Image.open(os.path.join(iter_dir, "test", "depth", f"frame{image_id:04d}.png"))
    error = Image.open(os.path.join(iter_dir, "test", "error", f"frame{image_id:04d}.png"))

    # Crop to ROI
    rgb = rgb.crop(roi)
    gt = gt.crop(roi)
    depth = depth.crop(roi)
    error = error.crop(roi)

    # Normalize depth and error
    depth_min = np.min(np.array(depth))
    depth_max = np.max(np.array(depth))
    depth = (np.array(depth) - depth_min) / (depth_max - depth_min) * 255
    depth = Image.fromarray(depth.astype(np.uint8))

    error_min = np.min(np.array(error))
    error_max = np.max(np.array(error))
    error = (np.array(error) - error_min) / (error_max - error_min) * 255
    error = Image.fromarray(error.astype(np.uint8))
    error = error.convert("L")


    # Plot in a 1x4 grid
    rows = 1
    cols = 4
    fig = plt.figure(figsize=(cols*3, rows*3))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.1, label_mode="all")

    grid[0].imshow(gt, interpolation="none")
    grid[1].imshow(rgb, interpolation="none")
    grid[2].imshow(depth, interpolation="none")
    grid[3].imshow(error, interpolation="none")

    # Add column labels
    grid[0].set_xlabel("Ground Truth")
    grid[1].set_xlabel("Ours (Color)")
    grid[2].set_xlabel("Ours (Depth)")
    grid[3].set_xlabel("Error")

    # Hide axes (except for x and y labels)
    for i in range(cols):
        grid[i].set_xticks([])
        grid[i].set_yticks([])
        grid[i].tick_params(axis='both', which='both', length=0)

        # Hide spines
        grid[i].spines['top'].set_visible(False)
        grid[i].spines['right'].set_visible(False)
        grid[i].spines['bottom'].set_visible(False)
        grid[i].spines['left'].set_visible(False)

    # Save figure
    output_dir = os.path.join("figures", os.path.basename(results_dir))
    os.makedirs(output_dir, exist_ok=True)

    for filetype in filetypes:
        plt.savefig(os.path.join(output_dir, f"blender_mirror.{filetype}"), bbox_inches="tight")
    
def render_scannet_extra(results_dir):
    scene_names = ["bathtub", 
                   "conference_room", 
                   "electrical_room", 
                   "plant", 
                   "printer"]
    choices = {
        "bathtub": 0,
        "conference_room": 0,
        "electrical_room": 0,
        "hotel": 0,
        "plant": 1,
        "printer": 3,
    }
    rois = []

    data = []
    for scene_name in scene_names:
        # Get iter directory
        iter_dirs = glob.glob(os.path.join(results_dir, f"scannet_extra_{scene_name}", "render", "iter_*"))
        iter_dir = sorted(iter_dirs, key=lambda x: int(os.path.basename(x).split("_")[-1]))[-1]

        # Load images
        rgb_path = sorted(glob.glob(os.path.join(iter_dir, "test", "rgb", "*.png")))[choices[scene_name]]
        gt_path = sorted(glob.glob(os.path.join(iter_dir, "test", "gt", "*.png")))[choices[scene_name]]
        depth_path = sorted(glob.glob(os.path.join(iter_dir, "test", "depth", "*.png")))[choices[scene_name]]
        error_path = sorted(glob.glob(os.path.join(iter_dir, "test", "error", "*.png")))[choices[scene_name]]

        rgb = Image.open(rgb_path)
        gt = Image.open(gt_path)
        depth = Image.open(depth_path)
        error = Image.open(error_path)

        # Load psnr
        psnr = json.load(open(os.path.join(iter_dir, "test", "results.json")))["perImage"][os.path.basename(rgb_path)]["PSNR"]

        data.append({
            "rgb": rgb,
            "gt": gt,
            "depth": depth,
            "error": error,
            "psnr": psnr,
        })

    # Plot as 5 x 3 grid, with each row being gt, ours (rgb), ours (depth)
    rows = len(scene_names)
    cols = 3

    fig = plt.figure(figsize=(cols*3, rows*3))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.1, label_mode="all")

    for idx, d in enumerate(data):
        grid[idx*cols+0].imshow(d["gt"], interpolation="none")
        grid[idx*cols+1].imshow(d["rgb"], interpolation="none")
        grid[idx*cols+2].imshow(d["depth"], interpolation="none")


        # Add labels
        grid[idx*cols+0].set_ylabel(pretty_names[scene_names[idx]])
        if idx == 0:
            grid[idx*cols+0].set_xlabel("Ground Truth")
            grid[idx*cols+1].set_xlabel("Ours (Color)")
            grid[idx*cols+2].set_xlabel("Ours (Depth)")
            grid[idx*cols+0].xaxis.set_label_position('top')
            grid[idx*cols+1].xaxis.set_label_position('top')
            grid[idx*cols+2].xaxis.set_label_position('top')
        
        # Hide axes (except for x and y labels)
        for icol in range(cols):
            grid[idx*cols+icol].set_xticks([])
            grid[idx*cols+icol].set_yticks([])
            grid[idx*cols+icol].tick_params(axis='both', which='both', length=0)

            # Hide spines
            grid[idx*cols+icol].spines['top'].set_visible(False)
            grid[idx*cols+icol].spines['right'].set_visible(False)
            grid[idx*cols+icol].spines['bottom'].set_visible(False)
            grid[idx*cols+icol].spines['left'].set_visible(False)

            if icol == 1:
                grid[idx*cols+icol].text(
                    0.97, 0.95, f"{d['psnr']:.1f}dB", 
                    transform=grid[idx*cols+icol].transAxes, 
                    fontsize=10, 
                    color="white", 
                    ha="right", 
                    va="top", 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none")
                )




    # Save figure
    output_dir = os.path.join("figures", os.path.basename(results_dir))
    os.makedirs(output_dir, exist_ok=True)

    for filetype in filetypes:
        plt.savefig(os.path.join(output_dir, f"scannet_extra.{filetype}"), bbox_inches="tight")






if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    args = parser.parse_args()

    # Remove trailing slash
    args.results_dir = args.results_dir.rstrip("/")

    # render_ortho(args.results_dir)
    # render_scannet(args.results_dir)
    # render_blender(args.results_dir)
    # render_skybox(args.results_dir)
    # render_blender_mirror(args.results_dir)
    render_scannet_extra(args.results_dir)