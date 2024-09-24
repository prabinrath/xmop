import torch
from training import MpiNetDataset
from benchmark.ompl_xcod_planner_bench import OmplXCoDPlanner
from xcod import XCoD, XCoDTracIK
from common import BulletRobotEnv, RealRobotPointSampler
import numpy as np
import time
import yaml

device = 'cuda'

mpinet_dataset = MpiNetDataset('global_solutions', 
                               'resources/datasets/mpinet_dataset/train.hdf5', 
                               'urdf/franka_panda/panda.urdf',
                               num_obstacle_points=4096*4,
                               sample_color=True)
random_indices = np.random.choice(len(mpinet_dataset), 10, replace=False)

URDF_PATH = 'urdf/sawyer/sawyer_sample.urdf'
# URDF_PATH = 'urdf/franka_panda/panda_sample.urdf'
# URDF_PATH = 'urdf/ur_robots/ur10_nmp_sample.urdf'
# URDF_PATH = 'urdf/kuka_iiwa14/iiwa14_sample.urdf'
# URDF_PATH = 'urdf/kinova/kinova_7dof_sample.urdf'

with open("config/robot_point_sampler.yaml") as file:
    robot_point_sampler = RealRobotPointSampler(
        urdf_path=URDF_PATH, 
        config=yaml.safe_load(file)['ompl_xcod_planning'],
        num_robot_points=4096,
        device='cuda')

coll_model = XCoD(
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
coll_model.eval()

nc_classical_mp = OmplXCoDPlanner(
                    coll_model=coll_model,
                    robot_point_sampler=robot_point_sampler,
                    path_cost_threshold=20,
                    min_search_time=20, 
                    timeout=100,
                    verbose=False)
xcod_ik = XCoDTracIK(coll_model, robot_point_sampler, URDF_PATH)

sim_handle = BulletRobotEnv(gui=True)
sim_handle.load_robot(URDF_PATH)


success = 0
for idx in random_indices:
    (obstacle_surface_pts,_), obstacle_config, eef_plan = mpinet_dataset[idx]
    obstacle_surface_pts = torch.as_tensor(obstacle_surface_pts, dtype=torch.float32, device='cuda')

    start_ee_pose = eef_plan[0]
    start_ee_pose = torch.as_tensor(np.concatenate(start_ee_pose, axis=0), 
                    dtype=torch.float32, device='cuda')
    start_js = xcod_ik.collision_free_ik(start_ee_pose, obstacle_surface_pts, 
                                       check_ee=True, gitr=10, num_samples=64)
    
    goal_ee_pose = eef_plan[-1]
    goal_ee_pose = torch.as_tensor(np.concatenate(goal_ee_pose, axis=0), 
                    dtype=torch.float32, device='cuda')
    goal_js = xcod_ik.collision_free_ik(goal_ee_pose, obstacle_surface_pts, 
                                      check_ee=True, gitr=10, num_samples=64)
    
    if start_js is not None and goal_js is not None:
        traj, cost = nc_classical_mp.plan(start_js.cpu().numpy(), 
                                     goal_js.cpu().numpy(),
                                     obstacle_surface_pts,
                                     planner="ait_star",
                                     # traj_resolution=0.05,
                                     # validate_plan=True
                                     )
        if traj is not None:
            print(f'IK + Planning successful')
            success += 1

            sim_handle.remove_obstacles()
            sim_handle.load_obstacles(obstacle_config)
            sim_handle.set_dummy_state(eef_plan[0][0], eef_plan[0][1])
            time.sleep(0.5)
            # visualize n_dof plan
            for js in traj:
                sim_handle.marionette_robot(js)
                time.sleep(0.05)
            sim_handle.set_dummy_state(eef_plan[-1][0], eef_plan[-1][1])
            time.sleep(0.5)
        else:
            print(f'Hybrid planning failed')
    else:
        print(f'IK failed')

print(f'reaching succcess: {success/len(random_indices)}')