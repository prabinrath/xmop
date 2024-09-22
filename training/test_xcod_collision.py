import torch
from torch.utils.data import DataLoader
import numpy as np
from xcod_dataset import XCoDDataset
from xcod import XCoD
from common.o3d_viz import Open3DVisualizer

color_map = np.array([[0.0,0.0,1.0], [0.0,1.0,0.0], [1.0,0.0,0.0]])

dataset = XCoDDataset('global_solutions', 
                mpinet_dataset_path='resources/datasets/mpinet_dataset/train.hdf5', 
                traj_dataset_root='resources/datasets/traj_dataset/',
                coll_dataset_root='resources/datasets/coll_dataset/',
                panda_urdf_path='urdf/franka_panda/panda.urdf', 
                n_dof_template_path='urdf/n_dof_template.xacro',
                traj_index=(0, 3270000),
                num_obstacle_points=4096*4,
                num_robot_points=4096*2,
                max_len=10,
                )

dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)

model = XCoD(
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
checkpoint = torch.load('checkpoints/XCoD_Collision_Model/XCoD_Collision_Model_terminal.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

for surface_pts, labels in dataloader:
    B, N = surface_pts.shape[:2]
    batch_coord = surface_pts[:,:,:3].view(-1,3)
    batch_feat = surface_pts.view(-1,4)
    input_dict = dict(
        coord=batch_coord.to('cuda').float(),
        feat=batch_feat.to('cuda').float(),
        batch=torch.arange(B).repeat_interleave(N).to('cuda'),
        grid_size=0.01
    )
    labels = labels.to('cuda').view(-1).long()
    
    labels = labels.squeeze()
    surface_pts = surface_pts.squeeze()
    labels_gt = labels + 1
    query_indices = labels != -1

    color_pts = color_map[labels_gt.cpu().numpy()]
    viz_handle = Open3DVisualizer(window_name='Collision Ground Truth')
    viz_handle.update_buffers(surface_pts[:,:3].numpy(), color_pts)
    viz_handle.vis.run()
    print("INFO: Close the Collision Ground Truth window to see the collision prediction")
    del viz_handle

    with torch.no_grad():
        output_dict = model(input_dict)
        seg_logits = output_dict['feat']
        labels_pred = torch.argmax(seg_logits, dim=1) + 1

    color_pts[query_indices.cpu().numpy()] = color_map[labels_pred[query_indices].cpu().numpy()]
    viz_handle = Open3DVisualizer(window_name='Collision Prediction')
    viz_handle.update_buffers(surface_pts[:,:3].numpy(), color_pts)
    print("INFO: Close the Collision Prediction window for next scenario")
    viz_handle.vis.run()
    del viz_handle
