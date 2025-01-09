# Arbitrary Optics for Gaussian Splatting Using Space Warping
This repository contains the official implementation for the paper *Arbitrary Optics for Gaussian Splatting Using Space Warping*, submitted to J.Imaging (MDPI).
It contains an extension to the 3D Gaussian Splatting method to arbitrary optics. This is achieved by using an intermediate fully differentiable warping step before rendering the warped scene with the original pinhole Gaussian rasterizer. 

<p align="center">
  <img width="460" height="300" src="figures/teaser.gif">
</p>

# Authors

- **Jakob Nazarenus (Corresponding Author)**  
  [ORCID: 0000-0002-6800-2462](https://orcid.org/0000-0002-6800-2462)  
  Email: [jna@informatik.uni-kiel.de](mailto:jna@informatik.uni-kiel.de)  
  _Department of Computer Science, Kiel University, Germany_

- **Simin Kou**  
  [ORCID: 0000-0002-7222-2214](https://orcid.org/0000-0002-7222-2214)  
  _School of Engineering and Computer Science, Victoria University of Wellington, New Zealand_

- **Fang-Lue Zhang**  
  [ORCID: 0000-0002-8728-8726](https://orcid.org/0000-0002-8728-8726)  
  _School of Engineering and Computer Science, Victoria University of Wellington, New Zealand_

- **Reinhard Koch**  
  [ORCID: 0000-0003-4398-1569](https://orcid.org/0000-0003-4398-1569)  
  _Department of Computer Science, Kiel University, Germany_


## Setup 
Either run the code locally with the same prequisites as the original 3DGS implementation or use the provided Dockerfile to build a suitable Docker image using `build_docker.sh`. You might need to adjust the CUDA-related parameters within the Dockerfile to fit to your experimental setup. Additionally, there are utility scripts for building and running the image. For running, adjust the mount paths in `run_docker.sh` to fit your directory structure. A typical training command would look like this:
```bash
python train.py -s /data/scannet/utility_room
```

Aside from the orginal 3DGS parameters, there are the following additional parameters.
| Command  | Explanation |
| ------------- | ------------- |
| --expname [NAME] | Manually set a name for the experiment  |
| --latency  | Perform a latency analysis for the trained model  |
| --skybox | Enable the learned Gaussian skybox |
| --fisheye_poly_degree [DEGREE] | Set the distortion polynomial to a degree of [DEGREE] |

If you want to run all experiments of the dataset on multiple GPUs, specify their IDs in `scripts/run_batch.py`:
```python
gpus = ["0", "1", "2"]
```
and run the evaluation (outside docker) using `sudo python scripts/run_batch.py`.



## Data
### Synthetic Blender
The blender fisheye dataset is provided [at this link](https://nextcloud.mip.informatik.uni-kiel.de/index.php/s/oPJEnd7FQq9s86m).
Aside from some scenes used for debugging, the main scenes are:
- [archiviz](https://download.blender.org/demo/cycles/flat-archiviz.blend)
- [barbershop](https://svn.blender.org/svnroot/bf-blender/trunk/lib/benchmarks/cycles/barbershop_interior/)
- [classroom](https://download.blender.org/demo/test/classroom.zip)
- [monk](https://download.blender.org/demo/cycles/lone-monk_cycles_and_exposure-node_demo.blend)
- [pabellon](https://download.blender.org/demo/test/pabellon_barcelona_v1.scene_.zip)
- [sky](https://cloud.blender.org/p/gallery/5f4d1791cc1d7c5e0e8832d4)

All those scenes are [official blender demos](https://www.blender.org/download/demo-files/) and rendered using the cycles raytracing renderer.

### ScanNet++
To obtain the dataset, the user must apply at the [offical dataset website](https://kaldir.vc.in.tum.de/scannetpp/).
The scenes used for evalualtion are:
- Bedroom (e8ea9b4da8)
- Kitchen (bb87c292ad)
- Office Day (4ba22fa7e4)
- Office Night (8d563fc2cc)
- Tool Room (d415cc449b)
- Utility Room (0a5c013435)
In order the automatically recognize these scenes as scannet scenes, place an empty `scannetpp` file in the scene's `dslr` directory.


## Configuration
For live modifications, some essential parameters are stored within the `gaussian-splatting/config.ini` file. Here, you can tune training and visualization parameters while the optimization is running. 