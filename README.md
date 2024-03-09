# PIE-NeRF🍕: Physics-based Interactive Elastodynamics with NeRF

### [[Project Page](https://fytalon.github.io/pienerf/)] [[arXiv](https://arxiv.org/abs/2311.13099)] [[Video](https://www.youtube.com/watch?v=V96GfcMUH2Q)]

Yutao Feng<sup>1,2</sup>\*, Yintong Shang<sup>1</sup>\*, Xuan Li<sup>3</sup>, Tianjia Shao<sup>2</sup>, Chenfanfu Jiang<sup>3</sup>, Yin Yang<sup>1</sup> <br>
<sup>1</sup>University of Utah, <sup>2</sup>Zhejiang University, <sup>3</sup>University of California, Los Angeles <br>
*Equal contributions

![teaser-1.jpg](_resources/teaser-1.jpg)

Abstract: *We show that physics-based simulations can be seamlessly integrated with NeRF to generate high-quality elastodynamics of real-world objects. Unlike existing methods, we discretize nonlinear hyperelasticity in a meshless way, obviating the necessity for intermediate auxiliary shape proxies like a tetrahedral mesh or voxel grid. A quadratic generalized moving least square (Q-GMLS) is employed to capture nonlinear dynamics and large deformation on the implicit model. Such meshless integration enables versatile simulations of complex and codimensional shapes. We adaptively place the least-square kernels according to the NeRF density field to significantly reduce the complexity of the nonlinear simulation. As a result, physically realistic animations can be conveniently synthesized using our method for a wide range of hyperelastic materials at an interactive rate.*

## News
- [2024-03-10] Code Release.
- [2024-02-27] Our paper has been accpetd by CVPR 2024!

## Cloning the Repository
This repository uses original gaussian-splatting as a submodule. Use the following command to clone:

```shell
git clone --recurse-submodules git@github.com:XPandora/PhysGaussian.git
```

## Setup

### Python Environment
To prepare the Python environment needed to run PhysGaussian, execute the following commands:
```shell
conda create -n PhysGaussian python=3.9
conda activate PhysGaussian

pip install -r requirements.txt
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
```
By default, We use pytorch=2.0.0+cu118.
### Quick Start
We provide several pretrained [Gaussian Splatting models](https://drive.google.com/drive/folders/1EMUOJbyJ2QdeUz8GpPrLEyN4LBvCO3Nx?usp=drive_link) and their corresponding `.json` config files in the `config` directory.

To simulate a reconstructed 3D Gaussian Splatting scene, run the following command:
```shell
python gs_simulation.py --model_path <path to gs model> --output_path <path to output folder> --config <path to json config file> --render_img --compile_video
```
The images and video results will be saved to the specified output_path.

If you want a quick try, run:
```shell
pip install gdown
bash download_sample_model.sh
python gs_simulation.py --model_path ./model/ficus_whitebg-trained/ --output_path output --config ./config/ficus_config.json --render_img --compile_video --white_bg
```

## Custom Dynamics
To generate custom dynamics, follow these guidelines:

### Gaussian Splatting Reconstruction
Begin by reconstructing a 3D GS scene as per [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

### Data Preprocessing
Before simulating Gaussian kernels as continuum particles, perform the following preprocessing steps:
1. Remove Gaussian kernels with low opacity.
2. Rotate the 3D scene to make it align with the coordinate plane (e.g., bottom surface parallel to the xy plane).
3. Define a cuboid simulation area.
4. Center and scale the simulation area within a unit cube.
5. Optionally, fill internal voids with particles.

Related parameters, such as rotation axis and degree, should be provided in the config file. For [Nerf Synthetic Dataset](https://drive.google.com/file/d/18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG/view?usp=drive_link), the reconstructed results typically already align with the axis.  For custom datasets, we use 3D software, e.g. [Houdini](https://www.sidefx.com/), to view the distribution of the Gaussian kernels and determine how to rotate and select the scene for simulation readiness.

### Config Json File
A single `.json` file should detail all data preprocessing and simulation parameters for each scene. Key parameters include:

- Data Preprocessing Parameters:
    - `opacity_threshold`: Filters out Gaussian kernels with opacity below this threshold.
    - `rotation_degree (list)` and `rotation_axis (list)`: Rotate the scene to align with the grid.
    - `sim_area (list)`: Choose the particles within a bounding box for simulation. The expected format is `[xmin, xmax, ymin, ymax, zmin, zmax]`.
    - `particle_filling (dict)`: Specify a cubic area to fill internal particles. Tuning ```density_threshold``` and ```search_threshold``` is usually needed for optimal filling results. See more details below.
- Simulation Parameters:
    - `material`: Available material types include `jelly`, `metal`, `sand`, `foam`, `snow` and `plasticine`.
    - `E`: Young's modulus 
    - `nu`: Poisson's Ratio
    - `density`: Material density  
    - `g`: Gravity
    - `substep_dt`: Simulation time step size 
    - `n_grid`: MPM grid size
    - `boundary_conditions (list)`: Boundary conditions can be enforced on either particles or grids, allowing manipulation of Gaussian kernels via external forces.
- Export Parameters:
    - `frame_dt`: Duration of each frame
    - `frame_num`: Total number of frames to export
    - `default_camera_index`: Camera view index from the training set

Please see sample config files under the `config` folder for reference. 

#### Particle Filling
Optionally, we employ a ray-collision-based method to detect inner grids for particle filling. To use this, specify the following parameters:

- `n_grid`: Particle filling grid size.
- `density_threshold`: Grid cells with density above this threshold will be treated as part of the surface shell.
- `search_exclude_direction`: A parameter (list of ints) for internal filling condition 1 in PhysGaussian. We won't cast rays in these excluded directions. The mapping between ints and directions is: 0, 1, 2, 3, 4, 5 (+x, -x, +y, -y, +z, -z).
- `ray_cast_direction`: tA parameter for internal filling condition 2 in PhysGaussian. Along this direction, we will detect the number of collision times. The mapping between ints and directions is the same as `search_exclude_direction`.
- `max_particles_per cell`: The number of particles to fill for each grid cell.
- `boundary`: Specify a well-reconstructed cubic area to perform particle filling.

Note: This particle filling algorithm is sensitive to Gaussian kernel distribution and may produce unsatisfying filling results if Gaussians are too noisy.

#### Boundary Condition
To fix or move the reconstructed object, specify the boundary condition either on grids or particles. Some commonly used boundary condition types include:

- `bounding_box`: Prevents particles from moving outside the MPM simulation area.
- `cuboid`: Enforces a boundary condition on the grid. Also specify other necessary parameters:
    - `point`: Center of the cubic area, e.g. `[1, 1, 1]`
    - `size`: Size of the cubic area (half of the width, height and depth), e.g. `[0.2, 0.2, 0.2]`
    - `vecloticy`: Velocity assigned to the grids, e.g. `[0, 0, 0]`
    - `start_time` and `end_time`: Time duration of this boundary condition
- `enforce_particle_translation`: Enforces a boundary condition on particles with parameters similar to those for grids.

## TODO

- Add more pretrained models.
- Colab script.

## Citation

```
@misc{feng2023pienerf,
      title={PIE-NeRF: Physics-based Interactive Elastodynamics with NeRF}, 
      author={Yutao Feng and Yintong Shang and Xuan Li and Tianjia Shao and Chenfanfu Jiang and Yin Yang},
      year={2023},
      eprint={2311.13099},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}      
```