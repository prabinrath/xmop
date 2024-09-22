import time
import torch
import numpy as np
import yaml
from xcod import XCoD
from training import MpiNetDataset
from common import RealRobotPointSampler, BulletRobotEnv
from common.o3d_viz import Open3DVisualizer
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

URDF_PATH = 'urdf/sawyer/sawyer_sample.urdf'
# URDF_PATH = 'urdf/franka_panda/panda_sample.urdf'
# URDF_PATH = 'urdf/ur_robots/ur5_sample.urdf'
# URDF_PATH = 'urdf/kuka_iiwa14/iiwa14_sample.urdf'
# URDF_PATH = 'urdf/kinova/kinova_7dof_sample.urdf'

with open("config/robot_point_sampler.yaml") as file:
    robot_point_sampler = RealRobotPointSampler(
        urdf_path=URDF_PATH, 
        config=yaml.safe_load(file)['xcod_ik'],
        num_robot_points=4096,
        device='cuda')

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

sim_handle = BulletRobotEnv(gui=True)
sim_handle.load_robot(URDF_PATH)

for idx in random_indices:
    (obstacle_surface_pts,_), obstacle_config, _ = mpinet_dataset[idx]
    sim_handle.load_obstacles(obstacle_config)

    js = np.random.uniform(robot_point_sampler.jl_limits, robot_point_sampler.ju_limits, 
                           (1,robot_point_sampler.DOF))
    js_th = torch.tensor(js, requires_grad=False, device=device)
    sim_handle.marionette_robot(js[0,:])
    manip_surface_pts = robot_point_sampler.sample_robot_points_batch(js_th).squeeze()
    obstacle_surface_pts = torch.from_numpy(obstacle_surface_pts).to(device)
    surface_pts = torch.cat((manip_surface_pts, obstacle_surface_pts.float()), dim=0)

    B, N = 1, surface_pts.shape[0]
    batch_coord = surface_pts[:,:3]
    batch_feat = surface_pts
    input_dict = dict(
        coord=batch_coord.to(device),
        feat=batch_feat.to(device),
        batch=torch.arange(B).repeat_interleave(N).to(device),
        grid_size=0.01
    )

    query_indices = torch.zeros((surface_pts.shape[0],)).bool()
    query_indices[surface_pts[:,3]!=0] = True
    with torch.no_grad():
        start = time.perf_counter()
        output_dict = model(input_dict)
        print(f'Inference time: {time.perf_counter()-start} for index: {idx}')
        seg_logits = output_dict['feat']
        labels_pred = torch.argmax(seg_logits, dim=1) + 1
    
    color_pts = np.ones((surface_pts.shape[0],3), dtype=np.float32) * \
        np.asarray([[0., 0., 1.]], dtype=np.float32)
    color_pts[query_indices.numpy()] = color_map[labels_pred[query_indices].cpu().numpy()]
    viz_handle = Open3DVisualizer(window_name='Collision Prediction')
    viz_handle.update_buffers(surface_pts[:,:3].cpu().numpy(), color_pts)
    print("INFO: Close the Collision Prediction window for next scenario")
    viz_handle.vis.run()
    del viz_handle

    sim_handle.remove_obstacles()