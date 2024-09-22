import torch
from training import MpiNetDataset
from common.nearby_ik import NearbyIK
from xmop import XMoP
from common import BulletRobotEnv
from common import JointConfigTool
import numpy as np

device = 'cuda'

mpinet_dataset = MpiNetDataset('global_solutions', 
                               'resources/datasets/mpinet_dataset/train.hdf5', 
                               'urdf/franka_panda/panda.urdf',
                               sample_color=True)
random_indices = np.random.choice(len(mpinet_dataset), 10, replace=False)

# URDF_PATH = 'urdf/sawyer/sawyer_sample.urdf'
# URDF_PATH = 'urdf/franka_panda/panda_sample.urdf'
URDF_PATH = 'urdf/ur_robots/ur10_nmp_sample.urdf'
# URDF_PATH = 'urdf/kuka_iiwa14/iiwa14_sample.urdf'
# URDF_PATH = 'urdf/kinova/kinova_7dof_sample.urdf'

planner_config = dict(
        mode='singlestep',
        urdf_path=URDF_PATH,
        config_folder='config/',
        model_dir=None,
        smoothing_factor=0,
    )
neural_planner = XMoP(planner_config, device=device)

MAX_ROLLOUT_STEPS = 1000
sim_handle = BulletRobotEnv(gui=True)
sim_handle.load_robot(URDF_PATH)

joint_config_tool = JointConfigTool(bounds=np.vstack((neural_planner.robot_point_sampler.jl_limits, 
                                                      neural_planner.robot_point_sampler.ju_limits)).T)
ik_solver = NearbyIK(URDF_PATH, neural_planner.robot_point_sampler.home_config)

success=0
for idx in random_indices:
    _, _, eef_plan = mpinet_dataset[idx]
    goal_eef = eef_plan[-1]
    sim_handle.set_dummy_state(goal_eef[0], goal_eef[1])
    goal_ee_pose = torch.as_tensor(np.concatenate(goal_eef, axis=0), 
                    dtype=torch.float32, device=device) 

    # mpinet start
    start_ee_pose = eef_plan[0]
    start_ee_pose = torch.as_tensor(np.concatenate(start_ee_pose, axis=0), 
                    dtype=torch.float32)
    start_js, _ = ik_solver.solve_nearby_pair(start_ee_pose, goal_ee_pose.cpu())
    if start_js is None:
        print('IK Failed')
        continue
    sim_handle.marionette_robot(start_js)  

    # normal start
    # start_ee_pose = goal_eef
    # start_ee_pose = torch.as_tensor(np.concatenate(start_ee_pose, axis=0), 
    #                 dtype=torch.float32)
    # start_js = ik_solver.solve_nearby(start_ee_pose)
    # if start_js is None:
    #     print('IK Failed')
    #     sim_handle.remove_obstacles()
    #     continue
    # start_js = joint_config_tool.clamp(np.random.normal(start_js, 
    #                                     0.5*np.ones((start_js.shape[0],))))
    # sim_handle.marionette_robot(start_js)

    data_dict_batch = neural_planner.prepare_data_dict_batch(goal_eef)
    observation = dict(
        curr_js=start_js,
    )
    for _ in range(MAX_ROLLOUT_STEPS):
        next_js, reached_goal = neural_planner.plan_online(
            observation, data_dict_batch, goal_thresh=0.05
        )
        sim_handle.marionette_robot(next_js[0])  
        observation = dict(
            curr_js=next_js[-1],
        )
        if reached_goal:
            success+=1
            break

print(f'reaching succcess: {success/len(random_indices)}')