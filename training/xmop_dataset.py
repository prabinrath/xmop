import io
import numpy as np
from urdf import NDofGenerator
from common import TrajDataManager, JointConfigTool
from urdfpy import URDF
from collections import defaultdict


class XMoPDataset():
    def __init__(self, 
                 traj_dataset_root, 
                 n_dof_template_path,
                 traj_index,
                 horizon=16,
                 obs_config_noise = 0.015,
                 max_dof=7,
                 max_len=None):
        self.traj_mgr = TrajDataManager(traj_dataset_root, traj_index[0], traj_index[1])
        if max_len is not None:
            self.success_indices = np.random.choice(self.traj_mgr.success_indices, max_len, replace=False)
        else:
            self.success_indices = self.traj_mgr.success_indices
        self.ndof_generator = NDofGenerator(template_path=n_dof_template_path,
                                            joint_gap=0.005, base_axis=2, base_offset=0.03)
        assert horizon >= 1
        self.horizon = horizon
        self.obs_config_noise = obs_config_noise
        self.max_dof = max_dof
    
    def __len__(self):
        return self.success_indices.shape[0]

    def __getitem__(self, index):
        mpinet_idx = self.success_indices[index]
        dof, kinematics, dynamics, traj = self.traj_mgr.retrieve_trajectory(mpinet_idx)
        assert dof<=self.max_dof
        urdf_text = self.ndof_generator.get_urdf(kinematics, dynamics)
        urdf_io_handle = io.BytesIO(initial_bytes=urdf_text)
        urdf_io_handle.name = 'n_dof_robot'
        urdf_handle = URDF.load(urdf_io_handle)
        joint_bounds = np.zeros((2,self.max_dof))
        joint_bounds[:,:dof] = np.vstack((kinematics[:,3], kinematics[:,4]))
        joint_config_tool = JointConfigTool(bounds=joint_bounds[:,:dof].T)

        episode_len = traj.shape[0]
        target_config = np.copy(traj[-1])
        # augment trajectory for horizon
        traj = np.vstack((traj, np.ones((self.horizon, dof))*target_config))

        # sample an intermediate index from the trajectory
        start_ts = np.random.randint(episode_len)
        joint_config_c =  np.copy(traj[start_ts])
        joint_config_n_h =  np.copy(traj[start_ts+1:start_ts+1+self.horizon])    

        # add joint state observation noise
        joint_config_c += self.obs_config_noise * np.random.randn(*joint_config_c.shape)
        joint_config_c = joint_config_tool.clamp(joint_config_c)

        # get link templates
        manip_semantic_link_map = defaultdict(list)
        for link in urdf_handle.links:
            manip_semantic_link_map[int(link.name[6])].append(link)
        
        # assign random sub links from link templates
        fk_link_indices = [np.random.choice(len(manip_semantic_link_map[key]))
                           for key in range(dof)]
        fk_link_indices.append(1) # gripper

        fk_dict = urdf_handle.link_fk(cfg=joint_config_c, use_names=True)
        # get rigid link frames from urdf
        PA = []
        for key, idx in zip(list(range(dof+1)), fk_link_indices):
            link = manip_semantic_link_map[key][idx]
            PA.append((fk_dict[link.name]@link.visuals[0].origin)[None,:,:])
        PA = np.concatenate(PA)
        PA_9d = np.zeros((self.max_dof+1,9))
        T_9d = np.concatenate((PA[:,:3,3], PA[:,:3,0], PA[:,:3,1]), axis=1)
        PA_9d[:dof] = T_9d[:dof]
        PA_9d[-1] = T_9d[dof]

        fk_dict = urdf_handle.link_fk(cfg=target_config, use_names=True)
        target_ee_link = manip_semantic_link_map[dof][fk_link_indices[-1]]
        goal_pose = fk_dict[target_ee_link.name]@link.visuals[0].origin
        goal_pose_9d = np.concatenate((goal_pose[:3,3], 
                                         goal_pose[:3,0], 
                                         goal_pose[:3,1]))
        
        TAB_9d_h = []
        for joint_config_n in joint_config_n_h:
            fk_dict = urdf_handle.link_fk(cfg=joint_config_n, use_names=True)
            # get rigid link frames from urdf
            PB = []
            for key, idx in zip(list(range(dof+1)), fk_link_indices):
                link = manip_semantic_link_map[key][idx]
                PB.append((fk_dict[link.name]@link.visuals[0].origin)[None,:,:])
            PB = np.concatenate(PB)

            # compute relative frame transformation for regression
            TAB = PB@np.linalg.inv(PA)
            TAB_9d = np.zeros((self.max_dof+1,9))
            T_9d = np.concatenate((TAB[:,:3,3], TAB[:,:3,0], TAB[:,:3,1]), axis=1)
            TAB_9d[:dof] = T_9d[:dof]
            TAB_9d[-1] = T_9d[dof]
            TAB_9d_h.append(TAB_9d[None,:,:])

        TAB_9d_h = np.vstack(TAB_9d_h)

        # build DOF mask
        dof_mask = np.full((self.max_dof+1), True)
        dof_mask[:dof] = False
        dof_mask[-1] = False

        # prepare data dict
        data_dict = dict(
            joint_bounds=joint_bounds.astype(np.float32),
            PA_9d=PA_9d.astype(np.float32),
            TAB_9d_h=TAB_9d_h.astype(np.float32),
            goal_pose_9d=goal_pose_9d.astype(np.float32),
            dof_mask=dof_mask,
        )
        return data_dict
