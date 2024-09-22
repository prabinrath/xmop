import h5py
import numpy as np
import os
import glob

class TrajDataManager():
    def __init__(self, traj_dataset_root, start_idx, end_idx, chkpt_interval=100, mode='r'):
        self.traj_dataset_path = os.path.join(traj_dataset_root, f'traj_{start_idx}_{end_idx}.h5')
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.chkpt_interval = chkpt_interval
        self.mode = mode
        self.create_dataset(self.traj_dataset_path)
        self.bkp_idx_track = 0

    def create_dataset(self, traj_dataset_path):
        if self.mode=='w':
            self.f = h5py.File(traj_dataset_path, self.mode)
            self.meta_dset = self.f.create_dataset("metadata", (2,), data=np.asarray([self.start_idx, self.end_idx]))
            # stores successful plan indices
            self.success_dset = self.f.create_dataset("success", (1,), maxshape=(None,), dtype='i4')
            # [total, dof, 10]
            self.kin_dset = self.f.create_dataset("kinematics", (self.end_idx-self.start_idx, 7, 10), dtype='f4')
            # [total, dof, 7]
            self.dyn_dset = self.f.create_dataset("dynamics", (self.end_idx-self.start_idx, 7, 7), dtype='f4')
            # [unlimited, dof]
            self.traj_dset = self.f.create_dataset("trajectories", (1,7), maxshape=(None,7), dtype='f4')
            # [total, [len_key, idx_begin, idx_end]]
            self.lenmap_dset = self.f.create_dataset("lenmap", (self.end_idx-self.start_idx, 4), dtype='i4')
        elif self.mode=='r' or self.mode=='a':
            if not os.path.exists(traj_dataset_path):
                # dont allow file creation even in append mode
                raise Exception('Dataset Not Found')
            self.f = h5py.File(traj_dataset_path, self.mode)
            self.meta_dset = self.f["metadata"]
            # existing h5 file has fixed start and end indices
            assert self.start_idx == self.meta_dset[0] and self.end_idx == self.meta_dset[1]
            self.success_dset = self.f["success"]
            self.kin_dset = self.f["kinematics"]
            self.dyn_dset = self.f["dynamics"]
            self.traj_dset = self.f["trajectories"]
            self.lenmap_dset = self.f["lenmap"]
        else:
            raise Exception('Invalid Mode')
    
    def close_dataset(self):
        self.f.close()
    
    def __del__(self):
        self.close_dataset()

    @property
    def success_indices(self):
        return self.success_dset[1:]
    
    def append_block(self, start_sub_idx, end_sub_idx, lenmap, kinematics, dynamics, trajectories, success):
        idx_begin = self.success_dset.shape[0]
        idx_end = idx_begin + success.shape[0]
        self.success_dset.resize(idx_begin+success.shape[0], axis=0)  
        self.success_dset[idx_begin:idx_end] = success
        self.kin_dset[start_sub_idx:end_sub_idx,:] = kinematics
        self.dyn_dset[start_sub_idx:end_sub_idx,:] = dynamics
        traj_len = trajectories.shape[0]
        idx_begin = self.traj_dset.shape[0]
        idx_end = idx_begin + traj_len
        self.traj_dset.resize(idx_begin+traj_len, axis=0)
        self.traj_dset[idx_begin:idx_end,:] = trajectories
        self.lenmap_dset[start_sub_idx:end_sub_idx,:] = lenmap   
        
    def save_trajectory(self, idx, kinematics, dynamics, traj):
        assert self.mode == 'a' or self.mode=='w'
        assert idx >= self.start_idx and idx <self.end_idx
        assert self.lenmap_dset[idx-self.start_idx,0] == 0 # don't allow overwriting
        self.success_dset.resize(self.success_dset.shape[0]+1, axis=0)
        self.success_dset[-1] = idx
        idx -= self.start_idx # relative index
        self.kin_dset[idx,:kinematics.shape[0],:] = kinematics.astype(np.float32)
        self.dyn_dset[idx,:dynamics.shape[0],:] = dynamics.astype(np.float32)
        traj_len = traj.shape[0]
        idx_begin = self.traj_dset.shape[0]
        idx_end = idx_begin + traj_len
        self.traj_dset.resize(idx_begin+traj_len, axis=0)
        self.traj_dset[idx_begin:idx_end,:traj.shape[1]] = traj.astype(np.float32)
        self.lenmap_dset[idx,:] = np.asarray([1.0, idx_begin, idx_end, traj.shape[1]], dtype=np.uint32)
        self.bkp_idx_track += 1

        # backup data in case of unprecedented issues
        if self.bkp_idx_track%self.chkpt_interval==0:
            self.close_dataset()
            self.mode = 'a'
            self.create_dataset(self.traj_dataset_path)

    def retrieve_trajectory(self, idx):
        assert idx >= self.start_idx and idx <self.end_idx
        idx -= self.start_idx # relative index
        len_key = self.lenmap_dset[idx]
        if len_key[0] == 0:
            raise Exception('Unsolved Plan Query')
        idx_begin, idx_end, dof = len_key[1], len_key[2], len_key[3]
        kinematics = self.kin_dset[idx][:dof,:]
        dynamics = self.dyn_dset[idx][:dof,:]
        traj = self.traj_dset[idx_begin:idx_end,:dof]
        return dof, kinematics, dynamics, traj

    def merge_traj_datasets(self, traj_dataset_root):
        assert self.mode == 'a' or self.mode=='w'
        manager_list = []
        for file in glob.glob(os.path.join(traj_dataset_root, '*.h5')):
            file_split = file.split('/')
            name_split = file_split[-1].split('.')[0].split('_')
            start_sub_idx, end_sub_idx = int(name_split[-2]), int(name_split[-1])
            mgr = TrajDataManager(traj_dataset_root, start_sub_idx, end_sub_idx)
            manager_list.append((start_sub_idx, end_sub_idx, mgr))

        manager_list.sort(key=lambda x: x[0])
        # ensure no overlap and continuous fragments
        for i in range(len(manager_list)-1):
            assert manager_list[i+1][0] - manager_list[i][1] == 0
        assert manager_list[0][0] == self.start_idx and manager_list[-1][1] == self.end_idx
        
        for start_sub_idx, end_sub_idx, mgr in manager_list:
            success = mgr.success_dset[1:]
            kinematics = mgr.kin_dset[:]
            dynamics = mgr.dyn_dset[:]
            trajectories = mgr.traj_dset[1:]
            lenmap = mgr.lenmap_dset[:]
            lenmap[:,1:3] += self.traj_dset.shape[0]-1
            # append data with relative index
            self.append_block(start_sub_idx-self.start_idx, end_sub_idx-self.start_idx, 
                              lenmap, kinematics, dynamics, trajectories, success)

    def merge_traj_datasets2(self, traj_dataset_root):
        assert self.mode == 'a' or self.mode=='w'
        manager_dict = {}
        for file in glob.glob(os.path.join(traj_dataset_root, '*/*.h5')):
            file_split = file.split('/')
            map_idx = int(file_split[-2])
            name_split = file_split[-1].split('.')[0].split('_')
            start_sub_idx, end_sub_idx = int(name_split[-2]), int(name_split[-1])
            assert self.start_idx == start_sub_idx and self.end_idx == end_sub_idx
            mgr = TrajDataManager(os.path.join(traj_dataset_root, file_split[-2]), start_sub_idx, end_sub_idx)
            manager_dict[map_idx] = mgr
        
        idx_map = np.load(os.path.join(traj_dataset_root, 'imap.npy'))
        # np.where(idx_map==-1)
        for idx in range(idx_map.shape[0]):
            if idx_map[idx] < 0:
                # unsolved plan
                continue
            # retrieve and save relative index
            _, kinematics, dynamics, traj = manager_dict[idx_map[idx]].retrieve_trajectory(idx+self.start_idx)
            self.save_trajectory(idx+self.start_idx, kinematics, dynamics, traj)
    
    def get_unsolved_indices(self):
        # relative index
        return np.where(self.lenmap_dset[:,0] == 0)[0] + self.start_idx