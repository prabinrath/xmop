from pathlib import Path
import shutil
from urdf import NDofGenerator
from training import MpiNetDataset
from common import TrajDataManager, BulletRobotEnv, JointConfigTool, RedirectStream
from common import CollisionDataManager
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
    Path(f'resources/datasets/coll_dataset/temp/{id}').mkdir(parents=True, exist_ok=True)
    idx_map = np.frombuffer(arr_mp, dtype=np.int32, count=len(arr_mp))
    with RedirectStream(f'log/{id}.log'):
        dataset = MpiNetDataset("global_solutions", dataset_path, panda_urdf_path, num_obstacle_points=None)
        sim_handle = BulletRobotEnv(gui=False)
        traj_mgr = TrajDataManager(f'resources/datasets/traj_dataset/', 0, 3270000)
        ndof_generator = NDofGenerator(template_path='urdf/n_dof_template.xacro',
                                            joint_gap=0.005, base_axis=2, base_offset=0.03)
        coll_mgr = CollisionDataManager(f'resources/datasets/coll_dataset/temp/{id}', start_idx, end_idx, mode='w')
        durations = []
        total = 0
        while True:
            with idx_mp.get_lock():
                idx = idx_mp.value
                idx_mp.value += 1
            if idx >= end_idx:
                break
            
            total += 1
            start_time = time.perf_counter()
            _, obstacle_config, _ = dataset[idx]
            sim_handle.load_obstacles(obstacle_config)
            dof, kinematics, dynamics, traj = traj_mgr.retrieve_trajectory(idx)
            joint_bounds = np.vstack((kinematics[:,3], kinematics[:,4]))
            joint_config_tool = JointConfigTool(bounds=joint_bounds.T)
            urdf_text = ndof_generator.get_urdf(kinematics, dynamics)
            with tempfile.NamedTemporaryFile(suffix='.urdf') as file:
                file.write(urdf_text)
                sim_handle.load_robot(file.name)
            episode_len = traj.shape[0]
            assert sim_handle.DOF == dof
            while True:                
                # sample an intermediate index from the trajectory
                start_ts = np.random.randint(episode_len)
                joint_config = np.copy(traj[start_ts])                
                noise = np.random.rand() # emulate diffusion time step
                # add joint state noise
                joint_noise = noise * np.random.randn(*joint_config.shape)
                joint_config += joint_noise
                joint_config = joint_config_tool.clamp(joint_config)
                sim_handle.marionette_robot(joint_config)
                sim_handle.perform_collision_check()
                collision_links = sim_handle.complete_collision_check()
                if -1 in collision_links:
                    collision_links.remove(-1) # remove baselink collision
                if len(collision_links) > 0:
                    joints = np.zeros((7,))
                    joints[0:dof] = joint_config
                    collision = np.zeros((7,), dtype=np.int8)
                    collision[np.asarray(list(collision_links), dtype=np.int32)] = 1
                    coll_mgr.save_collision(idx, joints, collision)
                    break
            durations.append(time.perf_counter()-start_time)
            sim_handle.clear_scene()
            idx_map[idx-start_idx] = id
            if total%chkpt==0:
                print(f'checkpoint total:{total}')

    with open(f'log/{id}.log', 'a') as f:
        f.write(f'checkpoint total:{total}\n')
        f.write(f'min_time:{min(durations)}, max_time:{max(durations)}, avg_time:{sum(durations)/len(durations)}\n')

def clean_dirs(path):
    directory = Path(path)
    for item in directory.glob('*'):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

def init_pool(shared_idx, shared_arr):
    global idx_mp
    global arr_mp
    idx_mp = shared_idx
    arr_mp = shared_arr

def main():    
    parser = argparse.ArgumentParser(description='XMoP Data Generation')
    parser.add_argument('--mpinet_dataset', default='resources/datasets/mpinet_dataset/train.hdf5', type=str, help='MpiNet train.hdf5 path')
    parser.add_argument('--panda_urdf', default='urdf/franka_panda/panda.urdf', type=str, help='Franka Panda urdf path')
    parser.add_argument('--traj_dataset', default='resources/datasets/traj_dataset/', type=str, help='traj dataset 0_3270000.h5 path')
    parser.add_argument('--start_idx', default=0, type=int, help='Starting MpiNet index')
    parser.add_argument('--end_idx', default=3270000, type=int, help='Ending MpiNet index')
    parser.add_argument('--num_proc', default=5, type=int, help='Number of workhorse processes')
    args = parser.parse_args()
    print(args)

    dataset_path = args.mpinet_dataset
    panda_urdf_path = args.panda_urdf
    start_idx, end_idx = args.start_idx, args.end_idx
    num_tasks = args.num_proc
    task_assignments = [[dataset_path, panda_urdf_path, start_idx, end_idx, id] for id in range(num_tasks)]

    idx = Value('i', start_idx)
    arr = RawArray('i', end_idx-start_idx)
    idx_map = np.frombuffer(arr, dtype=np.int32, count=len(arr))
    idx_map[:] = -1

    try:
        start_time = time.perf_counter()
        with Pool(initializer=init_pool, initargs=(idx, arr)) as pool:
            pool.starmap(gen_process, task_assignments)
        print(f'all workhorses completed in {datetime.timedelta(seconds=time.perf_counter()-start_time)}')
    except:
        print('premature termination')
    np.save('resources/datasets/coll_dataset/temp/imap.npy', idx_map)

    # merge collisions and clean up
    coll_mgr = CollisionDataManager('resources/datasets/coll_dataset/', start_idx, end_idx, mode='w')
    coll_mgr.merge_coll_datasets2('resources/datasets/coll_dataset/temp/')
    clean_dirs('resources/datasets/coll_dataset/temp/')

main()