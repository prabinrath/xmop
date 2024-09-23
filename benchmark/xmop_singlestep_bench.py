from xmop import XMoP
from benchmark import PlanningBenchmarker, MetricsEvaluator, BulletCollisionChecker
from common import TrajDataManager, RedirectStream
from multiprocessing import Pool, RawArray
from pathlib import Path
import shutil
import numpy as np
import datetime
import random
import argparse
import torch
import time
import os


def bench_process(robot_name, max_rollout_steps, start_idx, end_idx, gpu, idx):
    np.random.seed()
    random.seed()
    torch.random.seed()
    print(f'workhorse started with identifier: {idx}')
    experiment_name = f'singlestep-xmop-{args.robot_name}-mpinet-noobs'
    traj_dir = f'benchmark/results/temp/{experiment_name}'
    log_dir = f'log/{experiment_name}'
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    problem_handle = PlanningBenchmarker(robot_name=robot_name)
    planner_config = dict(
        mode='singlestep',
        urdf_path=problem_handle.urdf_path,
        config_folder='config/',
        model_dir=None,
        smoothing_factor=0,
    )
    neural_planner = XMoP(planner_config, device=f'cuda:{gpu}')

    Path(traj_dir).mkdir(parents=True, exist_ok=True)
    traj_mgr = TrajDataManager(traj_dir, start_idx, end_idx, mode='w')

    with RedirectStream(os.path.join(log_dir, f'{experiment_name}-{idx}.log')):
        for idx in range(start_idx, end_idx):
            _, _, start_js, _, _, goal_pose = problem_handle.get_problem(idx, ee_pose=True)

            observation = dict(
                curr_js=start_js,
                goal_eef=goal_pose,
            )
            start_time = time.perf_counter()
            traj = neural_planner.plan_offline(observation, max_rollout_steps, goal_thresh=0.01, exact=False)
            duration = time.perf_counter() - start_time

            if traj is not None:
                print(f'Evaluation Success for {idx}')
                traj_mgr.save_trajectory(idx, np.zeros((traj.shape[1], 10)), 
                                         np.zeros((traj.shape[1], 7)), traj)
                duration_arr[idx] = duration
            else:
                print(f'Evaluation Failed for {idx}')
                duration_arr[idx] = float('inf')

def clean_dirs(path):
    directory = Path(path)
    for item in directory.glob('*'):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

def init_pool(shared_arr):
    global duration_arr
    duration_arr = shared_arr

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='XMoP+XCoD+Singlestep')
    parser.add_argument('--robot_name', default='franka', type=str, help='Robot for benchmarking')
    parser.add_argument('--visualize', default=False, type=bool, help='Whether to visualize')
    parser.add_argument('--max_rollout_steps', default=200, type=int, help='Planning time')
    parser.add_argument('--num_proc', default=5, type=int, help='Number of workhorse processes')
    parser.add_argument('--num_gpu', default=1, type=int, help='Number of workhorse processes')
    args = parser.parse_args()
    print(args)

    experiment_name = f'singlestep-xmop-{args.robot_name}-mpinet-noobs'
    print(experiment_name)
    problem_handle = PlanningBenchmarker(robot_name=args.robot_name)
    sim_handle = BulletCollisionChecker(problem_handle.urdf_path, gui=args.visualize)
    metrics = MetricsEvaluator(problem_handle.urdf_path,
                               experiment_name=experiment_name,
                               sim_handle=sim_handle,
                               urdf_handle=problem_handle.urdf_handle)
    
    loaded_metrics = metrics.load(0, len(problem_handle))
    if not loaded_metrics:
        buckets = np.array_split(range(len(problem_handle)), args.num_proc) 
        task_assignments = [[args.robot_name, args.max_rollout_steps,
                              buckets[idx][0], buckets[idx][-1]+1, idx%args.num_gpu, idx] for idx in range(args.num_proc)]
        
        shared_arr = RawArray('f', len(problem_handle))
        try:
            start_time = time.perf_counter()
            with Pool(initializer=init_pool, initargs=(shared_arr,)) as pool:
                pool.starmap(bench_process, task_assignments)
            print(f'all workhorses completed in {datetime.timedelta(seconds=time.perf_counter()-start_time)}')
        except:
            print('premature termination')

        metrics.setup_trajectory_manager(0, len(problem_handle))
        metrics.traj_mgr.merge_traj_datasets(f'benchmark/results/temp/{experiment_name}')
        duration_arr = np.frombuffer(shared_arr, dtype=np.float32, count=len(shared_arr))
        
        for idx in range(len(problem_handle)):
            _, _, _, goal_js, _, goal_pose = problem_handle.get_problem(idx, ee_pose=args.visualize)
            if args.visualize:
                sim_handle.set_dummy_state(goal_pose[0], goal_pose[1])

            if np.isinf(duration_arr[idx]):
                print(f'Evaluation Failed for {idx}')
                metrics.evaluate_trajectory(
                    idx, None, None, None, None, skip_metrics=True)
            else:
                _, _, _, traj = metrics.traj_mgr.retrieve_trajectory(idx)
                if args.visualize:
                    # visualize n_dof plan
                    time.sleep(0.5)
                    for js in traj:
                        sim_handle.marionette_robot(js)
                        time.sleep(0.05)
                    time.sleep(0.5)
                metrics.evaluate_trajectory(
                    idx,
                    traj,
                    0.1,
                    problem_handle.urdf_handle.link_fk(cfg=goal_js, 
                            link="gripper_base_target", use_names=True),
                    duration_arr[idx],
                    save_traj=False
                )

    metrics_evaluation = metrics.metrics()
    metrics.print_metrics(metrics_evaluation)
    if not loaded_metrics:
        metrics.save(0, len(problem_handle))
    
    clean_dirs(f'benchmark/results/temp')
    