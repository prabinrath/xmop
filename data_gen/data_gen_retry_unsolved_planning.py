from pathlib import Path
import shutil
from urdf import UrdfSampler
from data_gen.ompl_planner import OmplPlanner
from training.mpinet_dataset import MpiNetDataset
from common import BulletRobotEnv, TrajDataManager, RedirectStream
from urdfpy import URDF
import numpy as np
import time
import tempfile
import random
from multiprocessing import Pool, Value, RawArray
import datetime
import argparse


def gen_process(dataset_path, panda_urdf_path, start_idx, end_idx, id):
    np.random.seed()
    random.seed()
    chkpt = 100
    print(f'workhorse started with identifier: {id}')
    Path(f'resources/datasets/traj_dataset/temp/{id}').mkdir(parents=True, exist_ok=True)
    idx_map = np.frombuffer(arr_mp, dtype=np.int32, count=len(arr_mp))
    with RedirectStream(f'log/{id}.log'):
        dataset = MpiNetDataset("global_solutions", dataset_path, panda_urdf_path, num_obstacle_points=None)
        planner = OmplPlanner(collision_inflation=0.01, path_cost_threshold=20.0, verbose=False)
        sim_handle = BulletRobotEnv(gui=False)
        traj_mgr = TrajDataManager(f'resources/datasets/traj_dataset/temp/{id}', start_idx, end_idx, mode='w')
        total = 0
        success = 0
        durations = []
        costs = []
        while True:
            with idx_mp.get_lock():
                idx_value = idx_mp.value
                idx_mp.value += 1
            if idx_value >= len(unsolved_mp):
                break
            
            total += 1
            start_time = time.perf_counter()
            idx = unsolved_mp[idx_value]
            _, obstacle_config, eef_plan = dataset[idx]
            sim_handle.load_obstacles(obstacle_config)

            # sample robot and get motion plan
            n_dof_plan = None
            urdf_sampler = UrdfSampler(n_dof_template_path='urdf/n_dof_template.xacro')
            for retry in range(10):
                print(f'Resampling Trail: {retry+1}')
                with tempfile.NamedTemporaryFile(suffix='.urdf') as file:
                    # hint: random robots are more successful in solving planning problems
                    kinematics, dynamics, urdf_text = urdf_sampler.sample_robot(
                        constraint=np.random.choice(a=['random', 'sawyer', 'ur5'], p=[0.34, 0.33, 0.33], replace=True))
                    file.write(urdf_text)
                    urdf_handle = URDF.load(file.name)
                    sim_handle.load_robot(file.name)
                assert sim_handle.DOF == len(urdf_handle.actuated_joints)
                n_dof_plan, cost = planner.plan(sim_handle, urdf_handle, eef_plan, traj_resolution=0.05)
                if n_dof_plan is None:
                    print('Planning failed')
                    sim_handle.remove_robot()
                else:
                    costs.append(cost)
                    traj_mgr.save_trajectory(idx, kinematics, dynamics, n_dof_plan)   
                    break

            durations.append(time.perf_counter()-start_time)
            sim_handle.clear_scene()
            if n_dof_plan is None:
                print('Could not solve the planning problem. Skipping environment.')
                continue
            
            success += 1
            idx_map[idx-start_idx] = id
            if total%chkpt==0:
                print(f'checkpoint: success:{success} out of total:{total}')

    if total:
        with open(f'log/{id}.log', 'a') as f:
            f.write(f'success:{success} out of total:{total}\n')
            f.write(f'min_time:{min(durations)}, max_time:{max(durations)}, avg_time:{sum(durations)/len(durations)}\n')
            if success:
                f.write(f'max_cost:{max(costs)}, avg_cost:{sum(costs)/len(costs)}\n')

def clean_dirs(path):
    directory = Path(path)
    for item in directory.glob('*'):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

def init_pool(shared_idx, shared_arr, shared_unsolved_arr):
    global idx_mp
    global arr_mp
    global unsolved_mp
    idx_mp = shared_idx
    arr_mp = shared_arr
    unsolved_mp = shared_unsolved_arr

def main():    
    parser = argparse.ArgumentParser(description='XMoP Data Generation Retry')
    parser.add_argument('--mpinet_dataset', default='resources/datasets/mpinet_dataset/train.hdf5', type=str, help='MpiNet train.hdf5 path')
    parser.add_argument('--panda_urdf', default='urdf/franka_panda/panda.urdf', type=str, help='Franka Panda urdf path')
    parser.add_argument('--start_idx', default=0, type=int, help='Starting MpiNet index')
    parser.add_argument('--end_idx', default=3270000, type=int, help='Ending MpiNet index')
    parser.add_argument('--num_proc', default=5, type=int, help='Number of workhorse processes')
    args = parser.parse_args()
    print(args)

    dataset_path = args.mpinet_dataset
    panda_urdf_path = args.panda_urdf
    start_idx, end_idx = args.start_idx, args.end_idx
    num_tasks = args.num_proc
    traj_mgr = TrajDataManager(f'resources/datasets/traj_dataset/', start_idx, end_idx)
    unsolved_indices = traj_mgr.get_unsolved_indices()
    del traj_mgr
    task_assignments = [[dataset_path, panda_urdf_path, start_idx, end_idx, id] for id in range(num_tasks)]

    idx = Value('i', 0)
    
    arr = RawArray('i', end_idx-start_idx)
    idx_map = np.frombuffer(arr, dtype=np.int32, count=len(arr))
    idx_map[:] = -1

    unsolved_arr = RawArray('i', unsolved_indices.shape[0])
    unsolved_map = np.frombuffer(unsolved_arr, dtype=np.int32, count=len(unsolved_arr))
    unsolved_map[:] = unsolved_indices

    print(f'found {unsolved_indices.shape[0]} unsolved environments')
    try:
        start_time = time.perf_counter()
        with Pool(initializer=init_pool, initargs=(idx, arr, unsolved_arr)) as pool:
            pool.starmap(gen_process, task_assignments)
        print(f'all workhorses completed in {datetime.timedelta(seconds=time.perf_counter()-start_time)}')
    except:
        print('premature termination')
    np.save('resources/datasets/traj_dataset/temp/imap.npy', idx_map)
    print(f'solved {np.where(idx_map!=-1)[0].shape[0]} out of {unsolved_indices.shape[0]} unsolved environments')

    # merge trajectories and clean up
    traj_mgr = TrajDataManager('resources/datasets/traj_dataset/', start_idx, end_idx, mode='a')
    traj_mgr.merge_traj_datasets2('resources/datasets/traj_dataset/temp/')
    clean_dirs('resources/datasets/traj_dataset/temp/')

main()