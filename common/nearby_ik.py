from tracikpy import TracIKSolver
import numpy as np
from common.rotation_torch import quat_to_rotmat
from urdfpy import URDF


class NearbyIK():
    def __init__(self, urdf_path, home_config, limit_factor=0.95):
        self.ik_solver = TracIKSolver(
            urdf_path,
            "base_link",
            "gripper_base_target"
        )
        self.home_config = np.array(home_config)
        
        urdf_handle = URDF.load(urdf_path)
        jl_limits, ju_limits = [], []
        for i in range(len(urdf_handle.actuated_joints)):
            limit = urdf_handle.actuated_joints[i].limit
            jl_limits.append(limit.lower)
            ju_limits.append(limit.upper)
        self.jl_limits, self.ju_limits = \
            np.asarray(jl_limits) * limit_factor, np.asarray(ju_limits) * limit_factor

    def within_limits(self, js):
        return np.all(js>self.jl_limits) and np.all(js<self.ju_limits)
    
    def get_tmat(self, pose):
        tmat = np.zeros((4,4))
        rotm = quat_to_rotmat(pose[3:].unsqueeze(0))
        tmat[:3,:3] = rotm.numpy()
        tmat[:3,3] = pose[:3].numpy()
        tmat[3,3] = 1.0        
        return tmat
    
    def solve_nearby_pair(self, start_ee_pose, goal_ee_pose, sim_handle=None, iter=100):
        start_tmat = self.get_tmat(start_ee_pose)
        goal_tmat = self.get_tmat(goal_ee_pose)

        start_solns, goal_solns = [], []
        for _ in range(iter):
            start_js = self.ik_solver.ik(start_tmat, qinit=self.home_config)
            goal_js = self.ik_solver.ik(goal_tmat, qinit=self.home_config)
            if start_js is not None:
                start_solns.append(start_js)
            if goal_js is not None:
                goal_solns.append(goal_js)
        
        if len(start_solns) == 0 or len(goal_solns) == 0:
            return None, None
        
        start_solns = np.vstack(start_solns)
        goal_solns = np.vstack(goal_solns)
        cost = np.linalg.norm(goal_solns[:, None, :] - \
                              start_solns[None, :, :], axis=2)
        flat_indices = np.argsort(cost, axis=None)
        sort_index_2d = np.unravel_index(flat_indices, cost.shape)
        for i, j in zip(*sort_index_2d):
            start_js, goal_js = start_solns[j], goal_solns[i]
            if sim_handle is not None:
                incollision = sim_handle.in_collision(start_js) or sim_handle.in_collision(goal_js)
            else:
                incollision = False
            if not incollision and self.within_limits(start_js) and self.within_limits(goal_js):
                return start_js, goal_js
        return None, None
    
    def solve_nearby(self, ee_pose, reference_js=None, sim_handle=None, iter=100):
        reference_js = self.home_config if reference_js is None else reference_js
        tmat = self.get_tmat(ee_pose)
        solns_costs = []
        for _ in range(iter):
            js = self.ik_solver.ik(tmat, qinit=self.home_config)
            if js is not None:
                solns_costs.append((js, np.linalg.norm(reference_js-js)))

        if len(solns_costs) == 0:
            return None
        
        solns_costs.sort(key=lambda x: x[1])
        for i in range(len(solns_costs)):
            js = solns_costs[i][0]
            if sim_handle is not None:
                incollision = sim_handle.in_collision(js)
            else:
                incollision = False
            if not incollision and self.within_limits(js):
                return js
        return None
