# <img src="media/logo.png" width="3%" alt="logo"> XMoP: Whole-Body Control Policy for Zero-shot Cross-Embodiment Neural Motion Planning
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?style=flat&logo=ubuntu&logoColor=white)
![Python](https://img.shields.io/badge/python-3.10.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg) <br>
<p align="center">
  <a href="https://prabinrath.github.io/xmop">[Project page]</a> •
  <a href="https://prabinrath.github.io/xmop/resources/paper.pdf">[Paper]</a> •
  <a href="https://huggingface.co/prabinrath/xmop">[Model]</a> •
  <a href="https://huggingface.co/datasets/prabinrath/xmop">[Data]</a>
</p>
<p align="center">
  <a href="https://prabinrath.github.io/">Prabin Kumar Rath</a><sup>1</sup>,
  <a href="https://nakulgopalan.github.io/">Nakul Gopalan</a><sup>1</sup> <br>
  <sup>1</sup>Arizona State University
</p>
XMoP is a novel configuration-space neural policy that solves motion planning problems zero-shot for unseen robotic manipulators, which has not been achieved by any prior robot learning algorithm. We formulate C-space control as a link-wise SE(3) pose transformation method, and showcase its scalability for data-driven policy learning. XMoP uses fully synthetic data to train models for motion planning and collision detection while demonstrating strong sim-to-real generalization with a 70% success rate. Our work demonstrates for the first time that C-space behavior cloning policies can be learned without embodiment bias and that these learned behaviors can be transferred to novel unseen embodiments in a zero-shot manner. This repository contains the implementation, data generation, and evaluation scripts for XMoP. <br><br>

<div align="center">
  <img src="media/approach.png" alt="approach">
</div> <br>

## Table of Contents
- 🦾 [Real-world rollouts](#-real-world-rollouts)
- 🛠️ [Installation](#%EF%B8%8F-installation)
  - [Docker setup](#1-docker-setup-recommended)
  - [Local setup](#2-local-setup)
- 🛝 [Try it out](#-try-it-out)
  - [Run examples](#1-run-examples)
  - [Data generation](#2-data-generation)
  - [Training](#3-training)
- 🩹 [Add a new manipulator](#-add-a-new-manipulator)
- 🐢 [ROS package](#-ros-package)
- 🏷️ [License](#%EF%B8%8F-license)
- 🙏 [Acknowledgement](#-acknowledgement)
- 📝 [Citation](#-citation)

## 🦾 Real-world rollouts
All rollouts shown below use XMoP with a fixed set of frozen policy parameters which were completely trained on synthetic (`robots and environments`) planning demonstration data.
<div align="center" style="max-width: 100%;margin: 0; padding: 0;">
  <img src="media/xmop_webpage_banner.gif" alt="realworld" style="width: 100%; height: auto;margin: 0; padding: 0;">
</div>

## 🛠️ Installation
The system has been tested on Ubuntu 22.04 with Intel 19 12th Gen CPU, 64GB RAM, and NVIDIA RTX 3090 GPU.
Clone the source from github using the below command.
```
git clone https://github.com/prabinrath/xmop.git -b main
```
### 1. Docker setup (recommended)
> Install docker

Install docker from this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-22-04).
Install `nvidia-container-toolkit` from this [tutorial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). It is recommended to `reboot` your system after installation of these software.

> Download image

The docker image comes pre-configured with all required dependencies for data generation, training, inference, and benchmarking.
```
cd <xmop-root-directory>
bash xmop_dev_docker.sh
```
If you are getting a `Permission denied` error, then use `sudo` before `bash`.
> Exit container
```
ctrl+d
```
### 2. Local setup
> System dependencies
```
sudo apt install -y libgl1-mesa-glx libgl1-mesa-dri \
liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev libnlopt-cxx-dev \
git wget
```
> Setup conda env
```
conda create -n xmop_dev python=3.10.13
conda activate xmop_dev
cd <xmop-root-directory>
bash setup_xmop_conda_dev_env.sh
```
The conda env needs to be deactivated and activated again to set the `PYTHONPATH` for xmop.
```
conda deactivate
conda activate xmop_dev
```
> OMPL (optional)

Ompl is required for data generation and baseline experiments. Install ompl with python bindings from [here](https://github.com/ompl/ompl). Setting up OMPL can be challenging and tricky to configure manually. To simplify the process, we recommend using our docker container, which comes pre-configured with required dependencies.

## 🛝 Try it out!
> Download datasets and benchmark assets
```
cd <xmop-root-directory>
bash download_resources.sh all
```
### 1. Run examples
For each of the following demos, you can change the value of `URDF_PATH` variable to run on different robots.
> Run XMoP planning demo
```
python examples/multistep_collisionfree_rollout.py
```
> Run XMoP-S reaching demo
```
python examples/singlestep_reaching_rollout.py
```
> Run XCoD collision detection demo
```
python examples/real_robot_collision_detection.py
```
> Run whole-body pose reconstruction with XCoD out-of-distribution demo
```
python examples/xmop_reconstruction_xcod_ood.py
```
> Run XCoD Ompl hybrid planning demo
```
python examples/ompl_xcod_hybrid_planning.py
```
### 2. Data generation
Data generation can be run on separate systems for consecutive fragments. The (`<start-idx>`, `<end-idx>`) refer to the MpiNets dataset indices. For our data generation we ran 32 consecutive fragments independently each with 100,000 problems. For example, (0, 100k), (100k, 200k), ..., (3m, 3.27m).
> Generate XMoP planning demonstration dataset
```
python data_gen/data_gen_planning.py --start_idx <start-idx> --end_idx <end-idx> --num_proc <number-of-cpus-to-use>
```
> Merge dataset fragments to generate a cohesive dataset, uncomment line (26-28) in the script below and run it
```
python data_gen/merge_and_visualize_traj_dataset.py
```
> Some problems might still remain unsolved, hence retry solving them
```
python data_gen/data_gen_retry_unsolved_planning.py --start_idx <start-idx> --end_idx <end-idx> --num_proc <number-of-cpus-to-use>
```
> Once planning demonstrations are generated, next generate collision detection dataset
```
python data_gen/data_gen_collision.py --start_idx <start-idx> --end_idx <end-idx> --num_proc <number-of-cpus-to-use>
```
### 3. Training
Use `-h` flag to see the parameters.
> Train XMoP diffusion policy
```
python training/train_xmop.py
```
> Train XMoP-S reaching policy
```
python training/train_xmop_s.py
```
> Train XCoD collision model
```
python training/train_xcod.py
```
## 🩹 Add a new manipulator
XMoP is a zero-shot policy that generalizes to unseen manipulators within a distribution. While setup for 7 commercial robots are provided, you can add new manipulators by following these steps:
- Modify the new robot's URDF
  - Add `base_link` and `gripper_base_target` dummy frames to the URDF, see existing robot descriptions in the `urdf/` folder for reference.
  - Save the modified file with a name ending in `_sample.urdf`.
- Update the `RealRobotPointSampler` Class
  - In `common/robot_point_sampler.py`, add a name keyword for the new manipulator to the constructor of the `RealRobotPointSampler` class.
  - **Examples:** `sawyer` for the Sawyer robot, `kinova6` for the 6-FoF Kinova robot, and so on.
- Configure link groups by modifying the `config/robot_point_sampler.yaml` file. Each robot requires four fields:
  - **semantic_map:** Specify a numerical ID for each URDF link. XCoD, considers links with the same ID are a single link.
  - **ee_links:** List IDs corresponding to end-effector links.
  - **home_config:** Define the home joint configuration of the robot.
  - **pose_skip_links:** List of link names that should not be considered for pose-tokens. The pre-trained XMoP policy currently allows max 8 pose-tokens supporting 6-DoF and 7-DoF manipulators. Hence, this list needs to be specified accordingly to ingore kinematic chain branches and redundant links.

By following these steps, you can successfully add new manipulators to be controlled with the pre-trained XMoP policy, expanding its capabilities to work with a wider range of robots. If you successfully add a new robot, please consider raising a [PR](https://github.com/prabinrath/xmop/pulls)!

<i>Note: The pre-trained XMoP policy does not generalize to out-of-distribution robots that have very different scale or morphology compared to the synthetic embodiment distribution it was trained on. For such novel class of robots, please follow our data generation scripts to design in-distribution synthetic robots to train XMoP from scratch.</i>

## 🐢 ROS package
![ROS Noetic](https://img.shields.io/badge/ROS-Noetic-green.svg) package for real-world deployment of XMoP along with usage tutorial can be found [here](https://github.com/prabinrath/xmop_ros).
## 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.
## 🙏 Acknowledgement
[MpiNets](https://mpinets.github.io/) • [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) • [Diffusion Transformer](https://www.wpeebles.com/DiT)
## 📝 Citation
If you find this codebase useful in your research, please cite [the XMoP paper](https://arxiv.org/pdf/2409.15585):
```bibtex
@article{rath2024xmop,
      title={XMoP: Whole-Body Control Policy for Zero-shot Cross-Embodiment Neural Motion Planning}, 
      author={Prabin Kumar Rath and Nakul Gopalan},
      year={2024},
      eprint={2409.15585},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.15585}, 
}
```