import os
import argparse
import multiprocessing as mp
import time
import random

gpus = ["0", "1", "2"]

tasks_blender = [
    f"python train.py --expname blender -r 1 --eval --skybox --fisheye_poly_degree 8 -s /data/blender-cycles/scenes/{scene}" for scene in [
        "archiviz",
        "barbershop",
        "classroom",
        "monk",
        "pabellon",
        "sky",
    ]
]

tasks_scannet = [
    f"python train.py --expname scannet -r 1 --eval --fisheye_poly_degree 8 -s /data/scannet/{scene}" for scene in [
        "bedroom",
        "kitchen",
        "office_day",
        "office_night",
        "tool_room",
        "utility_room",
    ]
]

tasks_jacobian = [
    "python train.py --expname jacobian -r 1 --eval --fisheye_poly_degree 8 --jacobians_off -s /data/scannet/utility_room",
    "python train.py -r 1 --eval --fisheye_poly_degree 8 -s /data/scannet/utility_room",
]

tasks_skybox = [
    "python train.py --expname skybox -r 1 --eval --skybox --fisheye_poly_degree 8 -s /data/blender-cycles/scenes/monk",
    "python train.py -r 1 --eval --fisheye_poly_degree 8 -s /data/blender-cycles/scenes/monk"
]

tasks_degree = [
    f"python train.py --expname poydegree -r 1 --eval --fisheye_poly_degree {i} -s /data/scannet/utility_room" for i in range(1, 11)
]

tasks_test = [
    "python train.py --expname test -r 1 --eval --fisheye_poly_degree 8 -s /data/scannet/utility_room"
]

tasks_all = tasks_blender + tasks_scannet + tasks_jacobian + tasks_skybox + tasks_degree


# # Add dryrun argument
# tasks_all = [" ".join(task.split()[:-2]) + " --dryrun " + " ".join(task.split()[-2:]) for task in tasks_all]


# Define a function to process a single string
def process(input_cmd):
    id = int(mp.current_process().name.split("-")[-1])-1
    gpu = gpus[id]
    print(f"[GPU {gpu}] {input_cmd}")

    cmd = ("docker run "
            "-v ./gaussian-splatting:/content/gaussian-splatting "
            "-v /datassd/jnazarenus/datasets:/data " 
            "-v ./cache:/root/.cache " 
            "--rm "
            f'--gpus "device={gpu}" ' 
            "--net=host "
            "--shm-size=32gb "
            "nazarenus/gaussians-fisheye:0.2 "
            f"{input_cmd}")

    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    # Check if you are within a docker container
    if os.path.exists("/.dockerenv"):
        raise RuntimeError("This script should not be run within a Docker container.")

    # Create a pool with one process per GPU
    with mp.Pool(len(gpus)) as pool:
        pool.starmap(process, [(input_cmd,) for input_cmd in tasks_all])
