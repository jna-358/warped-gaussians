import argparse
import os
import glob
import pandas as pd
import json
from tensorboard.backend.event_processing import event_accumulator
import tqdm

def get_tensorboard_data(subdir, keys=[]):
    event_files = glob.glob(os.path.join(subdir, "events.out.tfevents.*"))
    assert len(event_files) == 1
    event_file = event_files[0]

    result_dict = {}
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    result_dict = {
        **{key: ea.Scalars(key)[-1].value for key in keys},
        "time": ea.Scalars(keys[0])[-1].wall_time
    }

    return result_dict

def read_namespace(namespace_file):
    with open(namespace_file, "r") as f:
        text = f.read()
    assert text.startswith("Namespace(") and text.endswith(")")
    namespace = eval(f"argparse.{text}")
    return namespace

if __name__ == "__main__":
    # Get input directory
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    args = parser.parse_args()

    # Remove trailing slash
    args.input_dir = args.input_dir.rstrip("/")

    # Get all subdirectories
    subdirs = [os.path.basename(sub) for sub in glob.glob(os.path.join(args.input_dir, "*"))]
    
    # Eval blender experiments
    subdirs_blender = [sub for sub in subdirs if sub.startswith("blender")]

    df_blender = pd.DataFrame(columns=["scene", "gaussians", "time", "psnr", "ssim", "lpips"])
    for idx, sub in enumerate(tqdm.tqdm(subdirs_blender, desc="Blender experiments")):
        # Parse results.json
        results_json = os.path.join(args.input_dir, sub, "results.json")
        results = json.load(open(results_json))
        iter_keys = list(results.keys())
        highest_iter = sorted(iter_keys, key=lambda x: int(x.split("_")[-1]))[-1]
        results = results[highest_iter]["test"]


        # Parse tensorboard data
        tensorboard_data = get_tensorboard_data(os.path.join(args.input_dir, sub), keys=["total_points"])


        scene = sub.split("_")[1]

        # Create a new row with index idx
        df_blender.loc[idx] = [scene, tensorboard_data["total_points"], tensorboard_data["time"], results["PSNR"], results["SSIM"], results["LPIPS"]]

    # Eval scannet experiments
    subdirs_scannet = sorted([sub for sub in subdirs if sub.startswith("scannet") and not sub.startswith("scannet_extra")])

    df_scannet = pd.DataFrame(columns=["scene", "method", "gaussians", "psnr", "ssim", "lpips"])
    for idx, sub in enumerate(tqdm.tqdm(subdirs_scannet, desc="Scannet experiments")):
        scene_parts = sub.split("_")
        scene = "_".join(scene_parts[1:]) # Remove "scannet" prefix

        # Parse our results.json
        results_json = os.path.join(args.input_dir, sub, "results.json")
        results = json.load(open(results_json))
        iter_keys = list(results.keys())
        highest_iter = sorted(iter_keys, key=lambda x: int(x.split("_")[-1]))[-1]
        results = results[highest_iter]["test"]

        # Parse their results.json
        results_fisheyegs = json.load(open(os.path.join(args.input_dir, f"fisheyegs_{scene}", "results.json")))
        key = sorted([k for k in results_fisheyegs.keys() if k.startswith("ours")], key=lambda x: int(x.split("_")[-1]))[-1]
        results_fisheyegs = results_fisheyegs[key]

        # Parse tensorboard data
        tensorboard_data = get_tensorboard_data(os.path.join(args.input_dir, sub), keys=["total_points"])

        # Parse tensorboard data (fisheyegs)
        tensorboard_data_fisheyegs = get_tensorboard_data(os.path.join(args.input_dir, f"fisheyegs_{scene}"), keys=["total_points"])

        # Create a new row with index idx
        df_scannet.loc[2*idx] = [scene, "ours", tensorboard_data["total_points"], results["PSNR"], results["SSIM"], results["LPIPS"]]
        df_scannet.loc[2*idx+1] = [scene, "fisheyegs", tensorboard_data_fisheyegs["total_points"], results_fisheyegs["PSNR"], results_fisheyegs["SSIM"], results_fisheyegs["LPIPS"]]

    # Eval jacobian experiments
    subdirs_jacobian = [sub for sub in subdirs if sub.startswith("jacobian")]
    df_jacobian = pd.DataFrame(columns=["scene", "jacobian", "gaussians", "time", "psnr", "ssim", "lpips"])
    for idx, sub in enumerate(tqdm.tqdm(subdirs_jacobian, desc="Jacobian experiments")):
        # Parse results.json
        results_json = os.path.join(args.input_dir, sub, "results.json")
        results = json.load(open(results_json))
        iter_keys = list(results.keys())
        highest_iter = sorted(iter_keys, key=lambda x: int(x.split("_")[-1]))[-1]
        results = results[highest_iter]["test"]

        # Parse pipe namespace
        pipe_data = read_namespace(os.path.join(args.input_dir, sub, "cfg_pipe"))

        # Parse tensorboard data
        tensorboard_data = get_tensorboard_data(os.path.join(args.input_dir, sub), keys=["total_points"])

        scene = "utility_room"

        # Create a new row with index idx
        df_jacobian.loc[idx] = [scene, not pipe_data.jacobians_off, tensorboard_data["total_points"], tensorboard_data["time"], results["PSNR"], results["SSIM"], results["LPIPS"]]

    # Eval skybox experiments (model.skybox)
    subdirs_skybox = [sub for sub in subdirs if sub.startswith("skybox")]
    df_skybox = pd.DataFrame(columns=["scene", "skybox", "gaussians", "time", "psnr", "ssim", "lpips"])
    for idx, sub in enumerate(tqdm.tqdm(subdirs_skybox, "Skybox experiments")):
        # Parse results.json
        results_json = os.path.join(args.input_dir, sub, "results.json")
        results = json.load(open(results_json))
        iter_keys = list(results.keys())
        highest_iter = sorted(iter_keys, key=lambda x: int(x.split("_")[-1]))[-1]
        results = results[highest_iter]["test"]

        # Parse pipe namespace
        pipe_data = read_namespace(os.path.join(args.input_dir, sub, "cfg_model"))

        # Parse tensorboard data
        tensorboard_data = get_tensorboard_data(os.path.join(args.input_dir, sub), keys=["total_points"])

        scene = "monk"

        # Create a new row with index idx
        df_skybox.loc[idx] = [scene, pipe_data.skybox, tensorboard_data["total_points"], tensorboard_data["time"], results["PSNR"], results["SSIM"], results["LPIPS"]]

    # Eval polydegree experiments
    subdirs_polydegree = [sub for sub in subdirs if sub.startswith("poydegree")]
    df_polydegree = pd.DataFrame(columns=["scene", "polydegree", "gaussians", "time", "psnr", "ssim", "lpips"])
    
    for idx, sub in enumerate(tqdm.tqdm(subdirs_polydegree, "Polydegree experiments")):
        scene = "utility_room"

        # Parse results.json
        results_json = os.path.join(args.input_dir, sub, "results.json")
        results = json.load(open(results_json))
        iter_keys = list(results.keys())
        highest_iter = sorted(iter_keys, key=lambda x: int(x.split("_")[-1]))[-1]
        results = results[highest_iter]["test"]

        # Parse model namespace (fisheye_poly_degree)
        model_data = read_namespace(os.path.join(args.input_dir, sub, "cfg_model"))

        # Parse tensorboard data
        tensorboard_data = get_tensorboard_data(os.path.join(args.input_dir, sub), keys=["total_points"])

        # Create a new row with index idx
        df_polydegree.loc[idx] = [scene, model_data.fisheye_poly_degree, tensorboard_data["total_points"], tensorboard_data["time"], results["PSNR"], results["SSIM"], results["LPIPS"]]

    # Order by polydegree
    df_polydegree = df_polydegree.sort_values(by="polydegree")

    # Latency experiments
    subdirs_scannet = sorted([sub for sub in subdirs if sub.startswith("scannet") and not sub.startswith("scannet_extra")])
    df_latency = pd.DataFrame(columns=["scene", "mean", "std"])

    for idx, sub in enumerate(tqdm.tqdm(subdirs_scannet, desc="Latency experiments")):
        scene_parts = sub.split("_")
        scene = "_".join(scene_parts[1:])
        latency_json = os.path.join(args.input_dir, sub, "latency.json")
        latency = json.load(open(latency_json))
        mean = latency["mean"]
        std = latency["std"]
        df_latency.loc[idx] = [scene, mean, std]

    # Ortho experiments
    subdirs_ortho = [sub for sub in subdirs if sub.startswith("ortho")]
    assert len(subdirs_ortho) == 1, "Only one ortho experiment is allowed"
    subdir = subdirs_ortho[0]

    # Parse results.json
    results_json = os.path.join(args.input_dir, subdir, "results.json")
    results = json.load(open(results_json))
    iter_keys = list(results.keys())
    highest_iter = sorted(iter_keys, key=lambda x: int(x.split("_")[-1]))[-1]
    results = results[highest_iter]["test"]

    # Parse tensorboard data
    tensorboard_data = get_tensorboard_data(os.path.join(args.input_dir, subdir), keys=["total_points"])

    # Create ortho dataframe
    df_ortho = pd.DataFrame(columns=["scene", "gaussians", "time", "psnr", "ssim", "lpips"])
    df_ortho.loc[0] = ["lego", tensorboard_data["total_points"], tensorboard_data["time"], results["PSNR"], results["SSIM"], results["LPIPS"]]


    # Collect scannet extra results
    scene_names = ["bathtub", "conference_room", "electrical_room", "plant", "printer"]
    subdirs = [os.path.join(args.input_dir, f"scannet_extra_{scene}") for scene in scene_names]
    df_scannet_extra = pd.DataFrame(columns=["scene", "gaussians", "psnr", "ssim", "lpips"])

    for idx, sub in enumerate(tqdm.tqdm(subdirs, desc="Scannet extra experiments")):
        scene = scene_names[idx]

        # Parse results.json
        results_json = os.path.join(sub, "results.json")
        results = json.load(open(results_json))
        iter_keys = list(results.keys())

        highest_iter = sorted(iter_keys, key=lambda x: int(x.split("_")[-1]))[-1]
        results = results[highest_iter]["test"]

        # Parse tensorboard data
        tensorboard_data = get_tensorboard_data(sub, keys=["total_points"])

        # Create a new row with index idx
        df_scannet_extra.loc[idx] = [scene, tensorboard_data["total_points"], results["PSNR"], results["SSIM"], results["LPIPS"]]


    # Save all to csv
    data = {
        "blender": df_blender,
        "scannet": df_scannet,
        "jacobian": df_jacobian,
        "skybox": df_skybox,
        "polydegree": df_polydegree,
        "latency": df_latency,
        "ortho": df_ortho,
        "scannet_extra": df_scannet_extra
    }
    
    output_dir = os.path.join("results", os.path.basename(args.input_dir))
    os.makedirs(output_dir, exist_ok=True)
    for key, df in data.items():
        df.to_csv(output_path:=os.path.join(output_dir, f"{key}.csv"), index=False)
        print(f"Saved {output_path}")