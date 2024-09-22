import trimesh
import torch
from .torch_urdf import TorchURDF
import numpy as np
from pathlib import Path
from common.rotation_torch import quat_to_rotmat
from common.homo_transform_torch import transform_point_cloud_torch
from collections import OrderedDict


class RealRobotPointSampler():
    # Partly borrowed from https://github.com/fishbotics/robofin/blob/main/robofin/samplers.py

    def __init__(self, urdf_path, config, num_robot_points=4096, device='cpu', num_link_points=None):
        assert 'sample.urdf' in urdf_path # ensure modified urdf
        self.device = device
        # configs for specific robots
        if 'ur_robot' in urdf_path:
            self.robot = 'ur_robot'
        elif 'sawyer' in urdf_path:
            self.robot = 'sawyer'
        elif 'franka' in urdf_path:
            self.robot = 'franka'
        elif 'kuka' in urdf_path:
            self.robot = 'kuka'
        elif 'kinova_6' in urdf_path:
            self.robot = 'kinova_6'
        elif 'kinova_7' in urdf_path:
            self.robot = 'kinova_7'
        else:
            raise Exception('Unsupported Robot')
        self.load_config(config[self.robot])
        
        # load urdf and determine sampling ratios
        self.urdf_handle = TorchURDF.load(urdf_path, device=device)
        self.DOF = len(self.urdf_handle.actuated_joints)
        self.jl_limits = np.asarray([joint.limit.lower for joint in self.urdf_handle.actuated_joints], 
                                    dtype=np.float32)
        self.ju_limits = np.asarray([joint.limit.upper for joint in self.urdf_handle.actuated_joints], 
                                    dtype=np.float32)
        self.vel_limits = [joint.limit.velocity for joint in self.urdf_handle.actuated_joints]
        meshes = []
        for link in self.urdf_handle.links:
            if len(link.visuals) > 0:
                visual = link.visuals[0]
                if visual.geometry.mesh is not None:
                    meshes.append((link.name, trimesh.load( Path(urdf_path).parent /
                        visual.geometry.mesh.filename, force="mesh")))
        self.areas = [mesh.bounding_box_oriented.area for _, mesh in meshes]
        self.num_points = np.round(num_robot_points * np.array(self.areas) / 
                                   np.sum(self.areas)).astype(int)
        
        rounding_error = num_robot_points - np.sum(self.num_points)
        if rounding_error > 0:
            while rounding_error > 0:
                j = np.random.choice(np.arange(len(self.num_points)))
                self.num_points[j] += 1
                rounding_error = num_robot_points - np.sum(self.num_points)
        elif rounding_error < 0:
            while rounding_error < 0:
                j = np.random.choice(np.arange(len(self.num_points)))
                self.num_points[j] -= 1
                rounding_error = num_robot_points - np.sum(self.num_points)
        
        # cache the link pointclouds for faster processing
        self.points = {}
        self.semantic_points = []
        for i, (name, mesh) in enumerate(meshes):
            link_pc, _ = trimesh.sample.sample_surface(mesh, self.num_points[i])
            self.points[name] = torch.as_tensor(link_pc, dtype=torch.float32).to(device)
            self.semantic_points.append(torch.ones((link_pc.shape[0], 1), dtype=torch.float32) 
                                        * self.semantic_map[name])
        self.semantic_points = torch.vstack(self.semantic_points).to(device)
        if num_link_points is not None:
            semantic_points_np = self.semantic_points.squeeze().cpu().numpy()
            self.num_link_points = num_link_points
            self.link_sampling_indices = []
            for d in range(self.DOF+1):
                indices = np.where(semantic_points_np == d)[0]
                if d == self.DOF:
                    self.ee_start_idx = np.min(indices)
                ind_indices = np.random.choice(indices.shape[0], num_link_points, replace=True)
                self.link_sampling_indices.append(indices[ind_indices][None,:])
            self.link_sampling_indices = torch.from_numpy(np.concatenate(self.link_sampling_indices)
                                                          ).to(device)

        # cache ee pointcloud for target transformation
        default_joint_config_batch = torch.zeros((1, self.DOF), dtype=torch.float32, device=device)
        fk_dict = self.urdf_handle.link_fk_batch(cfgs=default_joint_config_batch, 
                                                            use_names=True)
        ee_base_trans = fk_dict['gripper_base_target'].detach().clone()
        ee_base_trans[0,:,:] = ee_base_trans[0,:,:].inverse()
        manip_surface_pts = self.sample_robot_points_batch(default_joint_config_batch)
        mask = manip_surface_pts[:,:,3]==self.ee_links[0]
        for i in range(1,len(self.ee_links)):
            mask = torch.logical_or(mask, manip_surface_pts[:,:,3]==self.ee_links[i])
        self.ee_points = manip_surface_pts[mask].view(1,-1,4)
        self.ee_points = transform_point_cloud_torch(self.ee_points, ee_base_trans, in_place=True)
        # np.save('surface_pts.npy', self.ee_points.squeeze().cpu().numpy())
    
    def load_config(self, config):
        self.semantic_map = config['semantic_map']
        self.ee_links = config['ee_links']
        if 'home_config' in config:
            self.home_config = config['home_config']
        if 'pose_skip_links' in config:
            self.pose_skip_links = config['pose_skip_links']
        else:
            self.pose_skip_links = None

    def sample_robot_points_batch(self, joint_config_batch, target_poses_batch=None, separated=False):
        if joint_config_batch is not None:
            assert target_poses_batch is None
            assert joint_config_batch.shape[1] == self.DOF
            assert isinstance(joint_config_batch, torch.Tensor)
            B = joint_config_batch.shape[0]
            fk_dict = self.urdf_handle.visual_geometry_fk_batch(cfgs=joint_config_batch, use_names=True)
            manip_surface_pts = []
            for name, pcd in self.points.items():
                link_pcd = pcd.detach().clone()
                link_pose = fk_dict[name]
                transformed_pcd = transform_point_cloud_torch(
                    link_pcd.repeat((B, 1, 1)),
                    link_pose,
                    in_place=True,
                )
                manip_surface_pts.append(transformed_pcd)
        else:
            assert joint_config_batch is None
            assert target_poses_batch.shape[1] == self.DOF+1
            assert isinstance(target_poses_batch, torch.Tensor)
            B = target_poses_batch.shape[0]
            manip_surface_pts = []
            for name, pcd in self.points.items():
                link_pcd = pcd.detach().clone()
                id_ = self.semantic_map[name]
                if id_ not in self.ee_links:
                    link_pose = target_poses_batch[:,id_]
                    transformed_pcd = transform_point_cloud_torch(
                        link_pcd.repeat((B, 1, 1)),
                        link_pose,
                        in_place=True,
                    )
                    manip_surface_pts.append(transformed_pcd)
            ee_pcd = self.ee_points[:,:,:3].detach().clone()
            manip_surface_pts.append(transform_point_cloud_torch(
                        ee_pcd.repeat((B, 1, 1)),
                        target_poses_batch[:,self.DOF],
                        in_place=True,
                    ))
             
        manip_surface_pts = torch.cat(manip_surface_pts, dim=1)
        if separated:
            assert self.num_link_points is not None
            index_tensor_expanded = self.link_sampling_indices.unsqueeze(0).expand(B, -1, -1)
            manip_surface_pts = torch.gather(manip_surface_pts.unsqueeze(1).expand(-1, self.DOF+1, -1, -1), 2, 
                                            index_tensor_expanded.unsqueeze(-1).expand(-1, -1, -1, 3))
        else:
            manip_surface_pts = torch.cat((manip_surface_pts, self.semantic_points.repeat((B, 1, 1))), dim=-1)
            
        return manip_surface_pts
    
    def sample_ee_points_batch(self, target_pose_batch, joint_config_batch=None, label=None, separated=False):
        if joint_config_batch is not None:
            assert target_pose_batch is None
            assert isinstance(joint_config_batch, torch.Tensor)
            B = joint_config_batch.shape[0]
            fk_dict = self.urdf_handle.link_fk_batch(cfgs=joint_config_batch, 
                                                            use_names=True)
            ee_base_trans = fk_dict['gripper_base_target']
            pos_batch = ee_base_trans[:,:3,3]
            rotmat_batch = ee_base_trans[:,:3,:3]
        else:
            assert joint_config_batch is None
            assert isinstance(target_pose_batch, torch.Tensor)
            B = target_pose_batch.shape[0]
            pos_batch = target_pose_batch[:,:3]
            rotmat_batch = quat_to_rotmat(target_pose_batch[:,3:])

        homo_pos_batch = torch.cat(
                (pos_batch, torch.ones((B,1), device=self.device)), dim=1
            )
        tmat_batch = torch.zeros((B,4,4), device=self.device)
        tmat_batch[:,:3,:3] = rotmat_batch
        tmat_batch[:,:,3] = homo_pos_batch

        ee_pcd = self.ee_points.detach().clone()
        ee_pts_batch = transform_point_cloud_torch(
            ee_pcd.repeat((B, 1, 1)),
            tmat_batch,
            in_place=True,
        )

        if separated:
            ee_indices = self.link_sampling_indices[self.DOF] - self.ee_start_idx
            ee_pts_batch = ee_pts_batch[:,ee_indices,:3]
        else:
            if label is not None:
                ee_pts_batch[:,:,3] = label
        return ee_pts_batch
    
    def get_link_poses(self, joint_config_batch, nine_d=False):
        assert isinstance(joint_config_batch, torch.Tensor)
        assert self.pose_skip_links is not None
        fk_dict = self.urdf_handle.link_fk_batch(cfgs=joint_config_batch, 
                                                 use_names=True)
        link_poses = OrderedDict()
        for name in self.points:
            if self.semantic_map[name] not in self.ee_links and name not in self.pose_skip_links:
                link_pose = fk_dict[name]
                link_pose = link_pose @ \
                    self.urdf_handle._link_map[name].visuals[0].origin.type_as(link_pose)
                link_poses[name] = link_pose
        link_poses['gripper_base_target'] = fk_dict['gripper_base_target']
        if nine_d:
            for key, link_pose in link_poses.items():
                link_poses[key] = torch.cat((link_pose[:,:3,3], 
                                             link_pose[:,:3,0], 
                                             link_pose[:,:3,1]), dim=-1)
        return link_poses
