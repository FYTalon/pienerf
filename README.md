# PIE-NeRF🍕: Physics-based Interactive Elastodynamics with NeRF

### [[Project Page](https://fytalon.github.io/pienerf/)] [[arXiv](https://arxiv.org/abs/2311.13099)] [[Video](https://www.youtube.com/watch?v=V96GfcMUH2Q)]

Yutao Feng<sup>1,2</sup>\*, Yintong Shang<sup>1</sup>\*, Xuan Li<sup>3</sup>, Tianjia Shao<sup>2</sup>, Chenfanfu Jiang<sup>3</sup>, Yin Yang<sup>1</sup> <br>
<sup>1</sup>University of Utah, <sup>2</sup>Zhejiang University, <sup>3</sup>University of California, Los Angeles <br>
*Equal contributions

Abstract: *We show that physics-based simulations can be seamlessly integrated with NeRF to generate high-quality elastodynamics of real-world objects. Unlike existing methods, we discretize nonlinear hyperelasticity in a meshless way, obviating the necessity for intermediate auxiliary shape proxies like a tetrahedral mesh or voxel grid. A quadratic generalized moving least square (Q-GMLS) is employed to capture nonlinear dynamics and large deformation on the implicit model. Such meshless integration enables versatile simulations of complex and codimensional shapes. We adaptively place the least-square kernels according to the NeRF density field to significantly reduce the complexity of the nonlinear simulation. As a result, physically realistic animations can be conveniently synthesized using our method for a wide range of hyperelastic materials at an interactive rate.*

## News
- [2024-03-10] Code Release.
- [2024-02-27] Our paper has been accpetd by CVPR 2024!

## Setup

The following setup is tested on Windows 11, with RTX 3060 and RTX 4080.

Our code is developed based on [torch-ngp](https://github.com/ashawkey/torch-ngp), a pytorch implementation of Instant-NGP.

### Cloning the Repository

```
git clone https://github.com/FYTalon/pienerf.git
```

### Python Environment

To prepare the Python environment needed to run PIE-NeRF, execute the following commands:
```shell
conda create -n pienerf python=3.10
conda activate pienerf
conda install ninja trimesh opencv tensorboardX numpy pandas tqdm matplotlib rich packaging scipy -c conda-forge
pip install imageio lpips torch-ema PyMCubes pysdf dearpygui torchmetrics
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install warp-lang==0.13.0 plyfile kornia
SET DISTUTILS_USE_SDK=1
# run the following line in each folder: raymarching, shencoder, gridencoder 
python setup.py build_ext --inplace && pip install .
```
### Quick Start
We provide several pretrained NeRF models and sampled point cloud files (with configs encoded) [here](https://drive.google.com/drive/folders/1gF56IjQpdXauV9gP8vbouRTnuwxR7mxa).

To simulate a reconstructed NeRF scene, 

first sample point cloud (the path `model/chair` should contain a file like `checkpoints/ngp*.pth`)

```
python point_sampling.py --dataset_type synthetic --workspace model/chair --exp_name chair_s  --sub_coeff 0.25 --sub_res 40
```

or you can skip this step by using provided point cloud.

Putting `chair.ply` into folder `model/chair`, and run

```shell
python main.gui.py --dataset_type synthetic --workspace model/chair --exp_name chair -O --max_iter_num 1 --num_seek_IP 3 --sim_dx 0.05
```
Press space to start simulation (or press again to pause). Left click on the object while ctrl is pressed to add force. Press key Q or right click to stop force.

![](\assets\gui.png)



### Parameters
Key command line parameters include:

- Point Sampling Parameters:
    - `sub_coeff`: The bigger, the more boundary points sampled.
    - `sub_res`: The bigger, the more grid points sampled.
- Simulation Parameters:
    - `sim_dt`: 
    - `sim_dx`:
    - `sim_iters`:
    - `sim_stiff`
- Rendering Parameters:
    - `max_iter_num`: For quadratic ray bending. The maximum number of of Newton iterations when solving for the rest shape position. More iterations give (possibly) better rendering quality and slower rendering speed.
    - `num_seek_IP`: For quadratic ray bending. The number of IPs to seek for each query point. At most 3 (i.e., valid values are 1, 2, 3). The rest position will be weighted sum of rest positions calculated by these IPs.



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