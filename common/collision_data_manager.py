import h5py
import numpy as np
import os
import glob

class CollisionDataManager():
    def __init__(self, coll_dataset_root, start_idx, end_idx, chkpt_interval=100, mode='r'):
        self.coll_dataset_path = os.path.join(coll_dataset_root, f'coll_{start_idx}_{end_idx}.h5')
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.chkpt_interval = chkpt_interval
        self.mode = mode
        self.create_dataset(self.coll_dataset_path)
        self.bkp_idx_track = 0

    def create_dataset(self, coll_dataset_path):
        if self.mode=='w':
            self.f = h5py.File(coll_dataset_path, self.mode)
            self.meta_dset = self.f.create_dataset("metadata", (2,), data=np.asarray([self.start_idx, self.end_idx]))
            # [total, (j1, j2, ...)]
            self.joints_dset = self.f.create_dataset("joints", (self.end_idx-self.start_idx, 7), dtype='f4')
            self.coll_dset = self.f.create_dataset("collision", (self.end_idx-self.start_idx, 7), dtype='i')
        elif self.mode=='r' or self.mode=='a':
            if not os.path.exists(coll_dataset_path):
                # dont allow file creation even in append mode
                raise Exception('Dataset Not Found')
            self.f = h5py.File(coll_dataset_path, self.mode)
            self.meta_dset = self.f["metadata"]
            # existing h5 file has fixed start and end indices
            assert self.start_idx == self.meta_dset[0] and self.end_idx == self.meta_dset[1]
            self.joints_dset = self.f["joints"]
            self.coll_dset = self.f["collision"]
        else:
            raise Exception('Invalid Mode')
    
    def close_dataset(self):
        self.f.close()
    
    def __del__(self):
        self.close_dataset()
        
    def save_collision(self, idx, joints, collision):
        assert self.mode == 'a' or self.mode=='w'
        assert idx >= self.start_idx and idx <self.end_idx
        idx -= self.start_idx # relative index
        self.joints_dset[idx] = joints
        self.coll_dset[idx] = collision
        self.bkp_idx_track += 1
        # backup data in case of unprecedented issues
        if self.bkp_idx_track%self.chkpt_interval==0:
            self.close_dataset()
            self.mode = 'a'
            self.create_dataset(self.coll_dataset_path)

    def retrieve_collision(self, idx):
        assert idx >= self.start_idx and idx <self.end_idx
        idx -= self.start_idx # relative index
        joints = self.joints_dset[idx]
        collision = self.coll_dset[idx]
        return joints, collision
    
    def merge_traj_datasets(self, coll_dataset_root):
        assert self.mode == 'a' or self.mode=='w'
        manager_list = []
        for file in glob.glob(os.path.join(coll_dataset_root, '*.h5')):
            file_split = file.split('/')
            name_split = file_split[-1].split('.')[0].split('_')
            start_sub_idx, end_sub_idx = int(name_split[-2]), int(name_split[-1])
            mgr = CollisionDataManager(coll_dataset_root, start_sub_idx, end_sub_idx)
            manager_list.append((start_sub_idx, end_sub_idx, mgr))

        manager_list.sort(key=lambda x: x[0])
        # ensure no overlap and continuous fragments
        for i in range(len(manager_list)-1):
            assert manager_list[i+1][0] - manager_list[i][1] == 0
        assert manager_list[0][0] == self.start_idx and manager_list[-1][1] == self.end_idx
        
        for start_sub_idx, end_sub_idx, mgr in manager_list:
            self.joints_dset[start_sub_idx:end_sub_idx] = mgr.joints_dset[:]
            self.coll_dset[start_sub_idx:end_sub_idx] = mgr.coll_dset[:]

    def merge_coll_datasets2(self, coll_dataset_root):
        assert self.mode == 'a' or self.mode=='w'
        manager_dict = {}
        for file in glob.glob(os.path.join(coll_dataset_root, '*/*.h5')):
            file_split = file.split('/')
            map_idx = int(file_split[-2])
            name_split = file_split[-1].split('.')[0].split('_')
            start_sub_idx, end_sub_idx = int(name_split[-2]), int(name_split[-1])
            assert self.start_idx == start_sub_idx and self.end_idx == end_sub_idx
            mgr = CollisionDataManager(os.path.join(coll_dataset_root, file_split[-2]), start_sub_idx, end_sub_idx)
            manager_dict[map_idx] = mgr
        
        idx_map = np.load(os.path.join(coll_dataset_root, 'imap.npy'))
        for idx in range(idx_map.shape[0]):
            if idx_map[idx] < 0:
                # unsolved scene
                continue
            # retrieve and save relative index
            joints, collision = manager_dict[idx_map[idx]].retrieve_collision(idx+self.start_idx)
            self.save_collision(idx+self.start_idx, joints, collision)
