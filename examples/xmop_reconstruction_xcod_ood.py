import time
import torch
import numpy as np
import yaml
from xcod import XCoD
from training import MpiNetDataset
from common import RealRobotPointSampler
from common.o3d_viz import Open3DVisualizer
from common.rotation_torch import rot6d_to_rotmat
from diffusers import DDPMScheduler

device = 'cuda'

color_map = np.array([[0.0,0.0,1.0], [0.0,1.0,0.0], [1.0,0.0,0.0]])

mpinet_dataset_path = 'resources/datasets/mpinet_dataset/train.hdf5'
panda_urdf_path = 'urdf/franka_panda/panda.urdf'

mpinet_dataset = MpiNetDataset('global_solutions', 
                               mpinet_dataset_path, 
                               panda_urdf_path, 
                               num_obstacle_points=4096*4,
                               sample_color=True)
random_indices = np.random.choice(len(mpinet_dataset), 10, replace=False)

# URDF_PATH = 'urdf/sawyer/sawyer_sample.urdf'
# URDF_PATH = 'urdf/franka_panda/panda_sample.urdf'
# URDF_PATH = 'urdf/ur_robots/ur10_sample.urdf'
URDF_PATH = 'urdf/kuka_iiwa14/iiwa14_sample.urdf'
# URDF_PATH = 'urdf/kinova/kinova_6dof_sample.urdf'

with open("config/robot_point_sampler.yaml") as file:
    robot_point_sampler = RealRobotPointSampler(
        urdf_path=URDF_PATH, 
        config=yaml.safe_load(file)['xmop_planning'],
        num_robot_points=4096,
        device='cuda')

noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2')
noise_scheduler.set_timesteps(10)

model = XCoD(
    pretrained=True,
    stride=[2, 2],
    enc_depths=[2, 4, 2],
    enc_channels=[32, 64, 128],
    enc_num_head=[2, 4, 8],
    enc_patch_size=[256, 256, 256],
    dec_depths=[2, 2],
    dec_channels=[32, 64],
    dec_num_head=[4, 8],
    dec_patch_size=[256, 256],
    ).to('cuda')
model.eval()

for idx in random_indices:
    (obstacle_surface_pts,_), obstacle_config, _ = mpinet_dataset[idx]

    j_orig = np.random.uniform(robot_point_sampler.jl_limits, robot_point_sampler.ju_limits, 
                           (1,robot_point_sampler.DOF))
    
    orig_link_poses_dict = robot_point_sampler.get_link_poses(
        torch.as_tensor(j_orig, device='cuda'), nine_d=True)
    orig_link_poses = []
    for key, link_pose in orig_link_poses_dict.items():
        orig_link_poses.append(link_pose.unsqueeze(0))
    orig_link_poses = torch.cat(orig_link_poses, dim=1)
    noisy_link_poses = noise_scheduler.add_noise(orig_link_poses, 
                                           torch.randn((robot_point_sampler.DOF+1,9), device='cuda'), 
                                           noise_scheduler.timesteps)
    noisy_link_poses = torch.vstack((noisy_link_poses, orig_link_poses))
    noisy_link_poses = noisy_link_poses.view(-1,9)
    noisy_link_poses_homo = torch.zeros((noisy_link_poses.shape[0],4,4), device=device)
    noisy_link_poses_homo[:,:3,:3] = rot6d_to_rotmat(noisy_link_poses[:,3:])
    noisy_link_poses_homo[:,:3,3] = noisy_link_poses[:,:3]
    noisy_link_poses_homo[:,3,3] = 1.0
    noisy_link_poses_homo = noisy_link_poses_homo.view(noise_scheduler.timesteps.shape[0]+1,
                                                       robot_point_sampler.DOF+1,4,4)
    
    obstacle_surface_pts = torch.from_numpy(obstacle_surface_pts).to(device) \
    .repeat((noisy_link_poses_homo.shape[0],1,1))
    manip_surface_pts = robot_point_sampler.sample_robot_points_batch(
            joint_config_batch=None,
            target_poses_batch=noisy_link_poses_homo)
    surface_pts = torch.cat((manip_surface_pts, obstacle_surface_pts.float()), dim=1)

    for i in range(noisy_link_poses_homo.shape[0]):
        B, N = 1, surface_pts[i].shape[0]
        batch_coord = surface_pts[i][:,:3]
        batch_feat = surface_pts[i]
        input_dict = dict(
            coord=batch_coord.to(device),
            feat=batch_feat.to(device),
            batch=torch.arange(B).repeat_interleave(N).to(device),
            grid_size=0.01
        )

        query_indices = torch.zeros((surface_pts[i].shape[0],)).bool()
        query_indices[surface_pts[i][:,3]!=0] = True
        with torch.no_grad():
            start = time.perf_counter()
            output_dict = model(input_dict)
            print(f'Inference time: {time.perf_counter()-start} for index: {idx}')
            seg_logits = output_dict['feat']
            labels_pred = torch.argmax(seg_logits, dim=1) + 1
        
        color_pts = np.ones((surface_pts[i].shape[0],3), dtype=np.float32) * \
            np.asarray([[0., 0., 1.]], dtype=np.float32)
        color_pts[query_indices.numpy()] = color_map[labels_pred[query_indices].cpu().numpy()]
        viz_handle = Open3DVisualizer(window_name='Collision Prediction')
        viz_handle.update_buffers(surface_pts[i][:,:3].cpu().numpy(), color_pts)
        print("INFO: Close the Collision Prediction window for next step")
        viz_handle.vis.run()
        del viz_handle