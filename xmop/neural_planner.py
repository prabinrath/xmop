from .xmop_s_planning_policy import SinglestepPosePlanningPolicy
from .xmop_planning_policy import MpcMultistepPosePlanningPolicy
from .js_retrieval import JointStateRetriever
from common import RealRobotPointSampler
from xcod import XCoD
from urdfpy import URDF
import numpy as np
from common.rotation_torch import rot6d_to_rotmat, quat_to_rotmat
import yaml
import torch
import os


class XMoP():
    def __init__(self, planner_config, validator=None, robot_point_sampler=None, device='cuda'):
        self.device = device
        self.smoothing_factor = planner_config['smoothing_factor']

        if robot_point_sampler is None:
            with open(os.path.join(planner_config['config_folder'], "robot_point_sampler.yaml")) as file:
                self.robot_point_sampler = RealRobotPointSampler(
                    urdf_path=planner_config['urdf_path'], 
                    config=yaml.safe_load(file)['xmop_planning'],
                    device=device)
        else:
            self.robot_point_sampler = robot_point_sampler
        
        self.mode = planner_config['mode']
        if self.mode == 'multistep':    
            if validator is None:            
                self.validator = XCoD(
                    pretrained=True,
                    model_dir=planner_config['model_dir'],
                    stride=[2, 2],
                    enc_depths=[2, 4, 2],
                    enc_channels=[32, 64, 128],
                    enc_num_head=[2, 4, 8],
                    enc_patch_size=[256, 256, 256],
                    dec_depths=[2, 2],
                    dec_channels=[32, 64],
                    dec_num_head=[4, 8],
                    dec_patch_size=[256, 256],
                    ).to(device)
                self.validator.eval()
            else:
                self.validator = validator

            with open(os.path.join(planner_config['config_folder'], "xmop_planning_policy.yaml")) as file:
                policy_config = yaml.safe_load(file)
                self.predict_horizon = policy_config['noise_model']['horizon']
                self.num_samples = policy_config['num_samples']
                self.planner = MpcMultistepPosePlanningPolicy(policy_config, self.robot_point_sampler).to(device)
                planner_state_dict = torch.hub.load_state_dict_from_url("https://huggingface.co/prabinrath/xmop/resolve/main/xmop.pth",
                                                                         planner_config['model_dir'])
                self.planner.load_state_dict(planner_state_dict['ema_model'])
                self.planner.eval()

        elif self.mode == 'singlestep':
            with open(os.path.join(planner_config['config_folder'], "xmop_s_reaching_policy.yaml")) as file:
                policy_config = yaml.safe_load(file)
                self.predict_horizon = 1
                self.num_samples = 1
                self.planner = SinglestepPosePlanningPolicy(policy_config).to(device)
                planner_state_dict = torch.hub.load_state_dict_from_url("https://huggingface.co/prabinrath/xmop/resolve/main/xmop_s.pth",
                                                                         planner_config['model_dir'])
                self.planner.load_state_dict(planner_state_dict['ema_model'])
                self.planner.eval()

        else:
            raise Exception('Invalid Mode')
        
        # assign dof mask
        self.dof_mask = torch.full((self.planner.max_dof+1,), True, device=self.device)
        self.dof_mask[:self.robot_point_sampler.DOF] = False
        self.dof_mask[-1] = False

        self.urdf_handle = URDF.load(planner_config['urdf_path'])
        self.js_ret = JointStateRetriever(self.urdf_handle, cost_th=0.5)

    def step(self, data_dict_batch, curr_link_poses, obstacle_surface_pts=None):
        curr_js = data_dict_batch['curr_js']
        if self.mode == 'multistep':
            assert obstacle_surface_pts is not None
            PA = torch.cat(list(curr_link_poses.values()))
            PA_9d = torch.zeros((self.planner.max_dof+1,9), device=self.device)
            T_9d = torch.cat((PA[:,:3,3], PA[:,:3,0], PA[:,:3,1]), dim=1)
            PA_9d[:self.robot_point_sampler.DOF] = T_9d[:self.robot_point_sampler.DOF]
            PA_9d[-1] = T_9d[self.robot_point_sampler.DOF]
            data_dict_batch['PA_9d'] = PA_9d.unsqueeze(0).repeat(self.num_samples,1,1)

            rel_poses_p, act_horizon = self.planner.get_action(data_dict_batch,
                                            PA.unsqueeze(0).repeat(self.num_samples,1,1,1),
                                            obstacle_surface_pts.unsqueeze(0).repeat(self.num_samples,1,1),
                                            self.validator)
            assert act_horizon <= self.predict_horizon
            rel_poses_p = rel_poses_p[~data_dict_batch['dof_mask'][0].repeat(self.predict_horizon)]

            TAB_p = torch.zeros(((self.robot_point_sampler.DOF+1)*self.predict_horizon,4,4), device=self.device)
            TAB_p[:,:3,:3] = rot6d_to_rotmat(rel_poses_p[:,3:])
            TAB_p[:,:3,3] = rel_poses_p[:,:3]
            TAB_p[:,3,3] = 1
            TAB_p = TAB_p.view(self.predict_horizon,self.robot_point_sampler.DOF+1,4,4)
            PB_p = TAB_p@PA

            short_horizon_traj = []
            for h in range(act_horizon):
                target_poses_9d = {}
                for key, pose in zip(curr_link_poses.keys(), PB_p[h].cpu().numpy()):
                    target_poses_9d[self.urdf_handle.link_map[key].visuals[0].geometry] = \
                        np.concatenate((pose[:3,3], pose[:3,0], pose[:3,1]))
                joint_config_n_p = self.js_ret.retrieve_js(target_poses_9d, curr_js)
                curr_js = self.smoothing_factor * curr_js + (1-self.smoothing_factor) * joint_config_n_p
                short_horizon_traj.append(curr_js)
            
            return short_horizon_traj
        
        elif self.mode == 'singlestep':
            PA = torch.cat(list(curr_link_poses.values()))
            PA_9d = torch.zeros((self.planner.max_dof+1,9), device=self.device)
            T_9d = torch.cat((PA[:,:3,3], PA[:,:3,0], PA[:,:3,1]), dim=1)
            PA_9d[:self.robot_point_sampler.DOF] = T_9d[:self.robot_point_sampler.DOF]
            PA_9d[-1] = T_9d[self.robot_point_sampler.DOF]
            data_dict_batch['PA_9d'] = PA_9d.unsqueeze(0)

            with torch.no_grad():
                rel_poses_p = self.planner.get_action(data_dict_batch)

            rel_poses_p = rel_poses_p[~data_dict_batch['dof_mask']]

            TAB_p = torch.zeros((self.robot_point_sampler.DOF+1,4,4), device=self.device)
            TAB_p[:,:3,:3] = rot6d_to_rotmat(rel_poses_p[:,3:])
            TAB_p[:,:3,3] = rel_poses_p[:,:3]
            TAB_p[:,3,3] = 1
            PB_p = TAB_p@PA

            target_poses_9d = {}
            for key, pose in zip(curr_link_poses.keys(), PB_p.cpu().numpy()):
                target_poses_9d[self.urdf_handle.link_map[key].visuals[0].geometry] = \
                    np.concatenate((pose[:3,3], pose[:3,0], pose[:3,1]))
            joint_config_n_p = self.js_ret.retrieve_js(target_poses_9d, curr_js)
            curr_js = self.smoothing_factor * curr_js + (1-self.smoothing_factor) * joint_config_n_p
            return [curr_js]
        
    def get_curr_link_poses(self, curr_js, goal_pose_9d, goal_thresh):
        curr_link_poses = self.robot_point_sampler.get_link_poses(
            torch.from_numpy(curr_js).to(self.device).unsqueeze(0))
        curr_pose = curr_link_poses['gripper_base_target'].squeeze()
        curr_pose_9d = torch.cat((curr_pose[:3,3], 
                                curr_pose[:3,0],
                                curr_pose[:3,1]))

        goal_ik_dist = torch.linalg.norm(goal_pose_9d-curr_pose_9d)   
        if goal_ik_dist < goal_thresh:
            return curr_link_poses, True
        return curr_link_poses, False
    
    def prepare_data_dict_batch(self, goal_eef):
        # specify goal ee pose
        goal_ee_pose = torch.as_tensor(np.concatenate(goal_eef, axis=0), 
                dtype=torch.float32, device=self.device)
        goal_rot_mat = quat_to_rotmat(goal_ee_pose[3:].unsqueeze(0)).squeeze()
        goal_pose_9d = torch.cat((goal_ee_pose[:3], 
                                goal_rot_mat[:,0],
                                goal_rot_mat[:,1]))
        
        # prepare data dict
        data_dict_batch = dict(
            goal_pose_9d=goal_pose_9d.unsqueeze(0).repeat(self.num_samples,1),
            dof_mask=self.dof_mask.unsqueeze(0).repeat(self.num_samples,1)
        )

        return data_dict_batch
        
    def plan_online(self, observation, data_dict_batch, goal_thresh=0.05):
        # specify environment pointcloud
        if self.mode == 'multistep':
            obstacle_surface_pts = torch.as_tensor(observation['obstacle_surface_pts'], 
                                                dtype=torch.float32, device=self.device)
        else:
            obstacle_surface_pts = None

        curr_link_poses, reached_goal = self.get_curr_link_poses(
                observation['curr_js'],
                data_dict_batch['goal_pose_9d'][0],
                goal_thresh
            )  
        data_dict_batch['curr_js'] = observation['curr_js']
        next_js = self.step(
                data_dict_batch,
                curr_link_poses,
                obstacle_surface_pts
            )
        
        return np.asarray(next_js), reached_goal
    
    def plan_offline(self, observation, max_rollout_steps, goal_thresh=0.05, exact=True):
        # specify environment pointcloud
        if self.mode == 'multistep':
            obstacle_surface_pts = torch.as_tensor(observation['obstacle_surface_pts'], 
                                                dtype=torch.float32, device=self.device)
        else:
            obstacle_surface_pts = None

        data_dict_batch = self.prepare_data_dict_batch(observation['goal_eef'])
        data_dict_batch['curr_js'] = observation['curr_js']
        curr_link_poses, reached_goal = self.get_curr_link_poses(
                observation['curr_js'],
                data_dict_batch['goal_pose_9d'][0],
                goal_thresh
            )  

        # rollout policy
        planned_traj = []
        step_count = 0
        while not reached_goal and step_count < max_rollout_steps:
            planned_traj += self.step(
                data_dict_batch,
                curr_link_poses,
                obstacle_surface_pts
            )

            data_dict_batch['curr_js'] = planned_traj[-1]
            curr_link_poses, reached_goal = self.get_curr_link_poses(
                planned_traj[-1],
                data_dict_batch['goal_pose_9d'][0],
                goal_thresh
            )
            
            step_count += 1
        
        if exact and not reached_goal:
            return None
        if len(planned_traj) > 0:
            return np.vstack(planned_traj)
        return np.asarray([])
