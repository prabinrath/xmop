import torch
from training import MpiNetDataset
from xcod import XCoDTracIK, XCoD
from xmop import XMoP
from common import BulletRobotEnv, RealRobotPointSampler
import yaml
import time
import numpy as np

device = 'cuda'

mpinet_dataset = MpiNetDataset('global_solutions', 
                               'resources/datasets/mpinet_dataset/train.hdf5', 
                               'urdf/franka_panda/panda.urdf', 
                               num_obstacle_points=4096*2,
                               sample_color=True)
random_indices = np.random.choice(len(mpinet_dataset), 10, replace=False)

# URDF_PATH = 'urdf/sawyer/sawyer_sample.urdf'
URDF_PATH = 'urdf/franka_panda/panda_sample.urdf'
# URDF_PATH = 'urdf/ur_robots/ur5_nmp_sample.urdf'
# URDF_PATH = 'urdf/kuka_iiwa14/iiwa14_sample.urdf'
# URDF_PATH = 'urdf/kinova/kinova_7dof_sample.urdf'

MAX_ROLLOUT_STEPS = 200

with open("config/robot_point_sampler.yaml") as file:
    robot_point_sampler = RealRobotPointSampler(
        urdf_path=URDF_PATH, 
        config=yaml.safe_load(file)['xmop_planning'],
        device=device)

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

xcod_ik = XCoDTracIK(coll_model, robot_point_sampler, URDF_PATH)

planner_config = dict(
        mode='multistep',
        urdf_path=URDF_PATH,
        config_folder='config/',
        model_dir=None,
        smoothing_factor=0.25
    )
neural_planner = XMoP(planner_config, 
                    validator=coll_model,
                    robot_point_sampler=robot_point_sampler,
                    device=device)

MAX_ROLLOUT_STEPS = 200
sim_handle = BulletRobotEnv(gui=True)
sim_handle.load_robot(URDF_PATH)

success=0
for idx in random_indices:
    (obstacle_surface_pts,_), obstacle_config, eef_plan = mpinet_dataset[idx]
    obstacle_surface_pts = torch.as_tensor(obstacle_surface_pts, dtype=torch.float32, device=device)
    sim_handle.load_obstacles(obstacle_config)
    goal_eef = eef_plan[-1]
    sim_handle.set_dummy_state(goal_eef[0], goal_eef[1])

    # MpiNet start
    start_ee_pose = eef_plan[0]
    start_ee_pose = torch.as_tensor(np.concatenate(start_ee_pose, axis=0), 
                    dtype=torch.float32, device=device)
    start_js = xcod_ik.collision_free_ik(start_ee_pose, 
                obstacle_surface_pts,
                check_ee=False,
                num_samples=16, gitr=10).cpu().numpy()
    if start_js is None:
        print('IK Failed')
        sim_handle.remove_obstacles()
        continue
    sim_handle.marionette_robot(start_js)

    data_dict_batch = neural_planner.prepare_data_dict_batch(goal_eef)
    observation = dict(
        curr_js=start_js,
        obstacle_surface_pts=obstacle_surface_pts
    )
    for _ in range(MAX_ROLLOUT_STEPS):
        next_js, reached_goal = neural_planner.plan_online(
            observation, data_dict_batch, goal_thresh=0.05
        )
        for js in next_js:
            sim_handle.marionette_robot(js)  
            time.sleep(0.05)
        observation = dict(
            curr_js=next_js[-1],
            obstacle_surface_pts=obstacle_surface_pts
        )
        if reached_goal:
            success+=1
            break

    sim_handle.remove_obstacles()  

print(f'planning success: {success/len(random_indices)}')