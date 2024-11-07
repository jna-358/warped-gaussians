import os
import argparse
import multiprocessing as mp
import time
import random

gpus = ["0", "1", "2"]

datasets = [
    "archiviz",
    "barbershop",
    "classroom",
    "monk",
    "pabellon",
    "sky"
]



# Define a function to process a single string
def process_dataset(dataset, input_cmd):
    cmd_mod = input_cmd.replace("<DATASET>", dataset)
    id = int(mp.current_process().name.split("-")[-1])-1
    gpu = gpus[id]
    print(f"GPU {gpu}: {cmd_mod}")


# sudo docker run \
#             -v ./gaussian-splatting:/content/gaussian-splatting \
#             -v /datassd/jnazarenus/datasets:/data \
#             -v ./bash_history.txt:/root/.bash_history \
#             --rm \
#             --gpus $GPUS \
#             --shm-size=32gb \
#             --net=host \
#             -it \
#             nazarenus/gaussians-fisheye:0.1

    cmd = ("docker run "
            "-v ./gaussian-splatting:/content/gaussian-splatting "
            "-v /datassd/jnazarenus/datasets:/data " 
            "-v ./cache:/root/.cache " 
            "--rm "
            f'--gpus "device={gpu}" ' 
            "--net=host "
            "--shm-size=32gb "
            "nazarenus/gaussians-fisheye:0.2 "
            f"{cmd_mod}")

    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    # Check if you are within a docker container
    if os.path.exists("/.dockerenv"):
        raise RuntimeError("This script should not be run within a Docker container.")

    parser = argparse.ArgumentParser()
    parser.add_argument("input_cmd", type=str)
    args = parser.parse_args()
    input_cmd = args.input_cmd

# Create a pool with 3 workers
with mp.Pool(len(gpus)) as pool:
    # Use the pool's map method to apply the function to each string
    pool.starmap(process_dataset, [(dataset, input_cmd) for dataset in datasets])

