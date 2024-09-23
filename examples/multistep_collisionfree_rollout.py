import torch
from benchmark import PlanningBenchmarker
from xcod import XCoDTracIK, XCoD
from xmop import XMoP
from common import BulletRobotEnv, RealRobotPointSampler
import argparse
import yaml
import time
import numpy as np

device = 'cuda'

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='XMoP+XCoD')
    parser.add_argument('--robot_name', default='franka', type=str, help='Robot for benchmarking')
    parser.add_argument('--visualize', default=False, type=bool, help='Whether to visualize')
    parser.add_argument('--max_rollout_steps', default=200, type=int, help='Planning time')
    args = parser.parse_args()
    print(args)
    problem_handle = PlanningBenchmarker(robot_name=args.robot_name, num_obstacle_points=4096*2)
    random_indices = np.random.choice(len(problem_handle), 10, replace=False)


    with open("config/robot_point_sampler.yaml") as file:
        robot_point_sampler = RealRobotPointSampler(
            urdf_path=problem_handle.urdf_path, 
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

    xcod_ik = XCoDTracIK(coll_model, robot_point_sampler, problem_handle.urdf_path)

    planner_config = dict(
            mode='multistep',
            urdf_path=problem_handle.urdf_path,
            config_folder='config/',
            model_dir=None,
            smoothing_factor=0.25
        )
    neural_planner = XMoP(planner_config, 
                        validator=coll_model,
                        robot_point_sampler=robot_point_sampler,
                        device=device)


    sim_handle = BulletRobotEnv(gui=True)
    sim_handle.load_robot(problem_handle.urdf_path)

    success=0
    for idx in random_indices:
        obstacle_surface_pts, obstacle_config, _, _, start_pose, goal_pose = \
                    problem_handle.get_problem(idx, ee_pose=True)
        obstacle_surface_pts = torch.as_tensor(obstacle_surface_pts, dtype=torch.float32, device=device)
        sim_handle.load_obstacles(obstacle_config)
        sim_handle.set_dummy_state(goal_pose[0], goal_pose[1])

        # get start_js using xcod_ik
        start_ee_pose = torch.as_tensor(np.concatenate(start_pose, axis=0), 
                        dtype=torch.float32, device=device)
        start_js = xcod_ik.collision_free_ik(start_ee_pose, 
                    obstacle_surface_pts,
                    check_ee=False,
                    num_samples=16, gitr=10)
        if start_js is None:
            print('IK Failed')
            sim_handle.remove_obstacles()
            continue
        else:
            start_js = start_js.cpu().numpy()
        sim_handle.marionette_robot(start_js)

        data_dict_batch = neural_planner.prepare_data_dict_batch(goal_pose)
        observation = dict(
            curr_js=start_js,
            obstacle_surface_pts=obstacle_surface_pts
        )
        for _ in range(args.max_rollout_steps):
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