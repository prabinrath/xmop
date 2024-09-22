import torch
import torch.nn as nn
import numpy as np
from tracikpy import TracIKSolver
import torch.nn.functional as F
from common.rotation_torch import quat_to_rotmat


class XCoDTracIK:
    def __init__(self, coll_model, robot_point_sampler, urdf_path, coll_cost_thr=0.01, ik_cost_thr=0.01):
        assert 'sample.urdf' in urdf_path # ensure modified urdf
        self.ik_solver = TracIKSolver(
            urdf_path,
            "base_link",
            "gripper_base_target"
        )
        self.coll_cost_thr = coll_cost_thr
        self.ik_cost_thr = ik_cost_thr
        self.coll_model = coll_model
        self.robot_point_sampler = robot_point_sampler
        self.ee_match_loss_fun = nn.L1Loss(reduction='none')
        self.DOF = robot_point_sampler.DOF
    
    def collision_avoid_obj(self, js_th, obstacle_surface_pts, check_ee):
        manip_surface_pts = self.robot_point_sampler.sample_robot_points_batch(js_th)
        surface_pts = torch.cat((manip_surface_pts, obstacle_surface_pts), dim=1)
        B, N = surface_pts.shape[:2]
        batch_coord = surface_pts[:,:,:3].view(-1,3)
        batch_feat = surface_pts.view(-1,4)
        input_dict = dict(
            coord=batch_coord,
            feat=batch_feat,
            batch=torch.arange(B).repeat_interleave(N).to('cuda'),
            grid_size=0.01
        )
        output_dict = self.coll_model(input_dict)
        query_indices = surface_pts[:,:,3]!=0
        if not check_ee:
            for link_id in self.robot_point_sampler.ee_links:
                query_indices = torch.logical_and(query_indices, surface_pts[:,:,3]!=link_id)
        seg_logits = output_dict['feat'].view(B,N,2)
        query_logits = seg_logits[query_indices]
        labels = torch.zeros((query_logits.shape[0],), device='cuda', dtype=torch.int64)
        collision_loss = F.cross_entropy(query_logits, labels, ignore_index=-1, reduction='none')
        collision_loss = collision_loss.view(B,-1,1).mean(dim=1)
        return collision_loss
    
    def collision_free_ik(self, target_ee_pose, obstacle_surface_pts, check_ee=False, gitr=5, num_samples=32):
        assert isinstance(target_ee_pose, torch.Tensor)
        assert isinstance(obstacle_surface_pts, torch.Tensor)
        obstacle_surface_pts = obstacle_surface_pts.repeat((num_samples,1,1))
        tmat = np.zeros((4,4))
        rotm = quat_to_rotmat(target_ee_pose[3:].unsqueeze(0))
        tmat[:3,:3] = rotm.cpu().numpy()
        tmat[:3,3] = target_ee_pose[:3].cpu().numpy()
        tmat[3,3] = 1.0
        for _ in range(gitr):
            samples = []
            itr = 0
            while len(samples) < num_samples and itr < 1000:
                qout = self.ik_solver.ik(tmat)
                if qout is not None:
                    samples.append(qout)
                itr+=1
            if len(samples)==0:
                print('TracIK failed to solve with the provided URDF')
                break
            samples = np.vstack(samples)
            js_th = torch.tensor(samples, dtype=torch.float64, device='cuda', requires_grad=True)     

            with torch.no_grad():
                obj_coll_avoid = self.collision_avoid_obj(js_th, obstacle_surface_pts, check_ee)       
                best_idx = obj_coll_avoid.argmin()
                if obj_coll_avoid[best_idx] < self.coll_cost_thr:
                    return js_th[best_idx].detach()
            
        return None
