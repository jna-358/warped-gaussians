import os
import argparse
import multiprocessing as mp
import time
import random

gpus = ["0", "1"]

tasks_blender = [
    f"python train.py --expname blender_{scene} --eval -r 1 --latency --skybox --fisheye_poly_degree 8 -s /data/blender-cycles/scenes/{scene} && python metrics.py -m ./output/blender_{scene}" for scene in [
        "archiviz",
        "barbershop",
        "classroom",
        "monk",
        "pabellon",
        "sky",
    ]
]

tasks_scannet = [
    f"python train.py --expname scannet_{scene} --eval -r 1 --latency --fisheye_poly_degree 8 -s /data/scannet/{scene} && python metrics.py -m ./output/scannet_{scene}" for scene in [
        "bedroom",
        "kitchen",
        "office_day",
        "office_night",
        "tool_room",
        "utility_room",
    ]
]

tasks_jacobian = [
    "python train.py --expname jacobian_off --eval -r 1 --latency --fisheye_poly_degree 8 --jacobians_off -s /data/scannet/utility_room && python metrics.py -m ./output/jacobian_off",
    "python train.py --expname jacobian_on --eval -r 1 --latency --fisheye_poly_degree 8 -s /data/scannet/utility_room && python metrics.py -m ./output/jacobian_on",
]

tasks_skybox = [
    "python train.py --expname skybox_on --eval -r 1 --latency --skybox --fisheye_poly_degree 8 -s /data/blender-cycles/scenes/monk && python metrics.py -m ./output/skybox_on",
    "python train.py --expname skybox_off --eval -r 1 --latency --fisheye_poly_degree 8 -s /data/blender-cycles/scenes/monk && python metrics.py -m ./output/skybox_off",
]

tasks_degree = [
    f"python train.py --expname poydegree_{i} --eval -r 1 --latency --fisheye_poly_degree {i} -s /data/scannet/utility_room && python metrics.py -m ./output/polydegree_{i}" for i in [2, 4, 6, 8, 10]
]

tasks_ortho = [
    "python train.py --expname ortho_lego --white_background --eval --latency -r 1 -s /data/blender-cycles/scenes/lego && python metrics.py -m ./output/ortho_lego",
]

tasks_scannet_extra = [
    f"python train.py --expname scannet_extra_{scene} --eval -r 1 --latency --fisheye_poly_degree 8 -s /data/scannet_extra/{scene} && python metrics.py -m ./output/scannet_extra_{scene}" for scene in [
        # "bathtub",
        # "conference_room",
        # "electrical_room",
        "hotel",
        # "plant",
        # "printer",
    ]
]


# tasks_all = tasks_scannet + tasks_jacobian + tasks_degree

tasks_all = tasks_blender


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
            f'bash -c "{input_cmd}"')

    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    # Check if you are within a docker container
    if os.path.exists("/.dockerenv"):
        raise RuntimeError("This script should not be run within a Docker container.")

    # Create a pool with one process per GPU
    with mp.Pool(len(gpus)) as pool:
        pool.starmap(process, [(input_cmd,) for input_cmd in tasks_all])