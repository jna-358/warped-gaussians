# Warped Gaussians
This repository contains an extension to the 3D Gaussian Splatting method to arbitrary optics. This is achieved by using an intermediate fully differentiable warping step before rendering the warped scene with the original pinhole Gaussian rasterizer.

## Setup 
Either run the code locally with the same prequisites as the original 3DGS implementation or use the provided Dockerfile to build a suitable Docker image. You might need to adjust the CUDA-related parameters within the Dockerfile to fit to your experimental setup. Additionally, there are utility scripts for building and running the image. For running, adjust the mount paths in `run_docker.sh` to fit your directory structure.

## Data
The blender fisheye dataset is provided [at this link](https://nextcloud.mip.informatik.uni-kiel.de/index.php/s/oPJEnd7FQq9s86m). Aside from some scenes used for debugging, the main scenes are:
- archiviz
- barbershop
- classroom
- monk 
- pabellon
- sky
All those scenes are official blender demos and rendered using the cycles raytracing renderer.

## Configuration
For live modifications, some essential parameters are stored within the `gaussian-splatting/config.ini` file. Here, you can tune training and visualization parameters while the optimization is running. 