from scipy.optimize import minimize
import numpy as np
from pyquaternion import Quaternion


class CollisionIK:
    def __init__(self, fk_handle, sim_handle, collision_inflation=0.01):
        self.fk_handle = fk_handle
        self.sim_handle = sim_handle
        self.DOF = self.sim_handle.DOF
        self.collision_inflation = collision_inflation

    def groove(self, x, n, s, c, r):
        return (-1)**n * np.exp(-(x-s)**2 / (2*c**2)) + r*(x-s)**4

    def ee_match_obj(self, pg, qg, js):
        eef_pose = self.fk_handle.link_fk(js, link=f'link_g{self.DOF}1')

        pos_err = np.linalg.norm(eef_pose[:3,3]-pg)
        pos_obj = self.groove(x=pos_err, n=1, s=0, c=0.2, r=5.0)

        qg = Quaternion(qg).unit
        qt = Quaternion(matrix=eef_pose[:3,:3]).unit
        ori_err = Quaternion.absolute_distance(qg, qt)
        ori_obj = self.groove(x=ori_err, n=1, s=0, c=0.2, r=5.0)

        return pos_obj + ori_obj
    
    def collision_avoid_obj(self, js):
        curr_self_dist = self.sim_handle.self_collision_distance(collision_radius=0.1)
        curr_env_dist = self.sim_handle.env_collision_distance(collision_radius=0.3)

        off = self.collision_inflation
        self_coll_obj = sum([1.8**(-dist+off) if (dist-off)<0 else 0 for dist in curr_self_dist])
        env_coll_obj = sum([1.8**(-dist+off) if (dist-off)<0 else 0 for dist in curr_env_dist])
        
        return self_coll_obj + env_coll_obj
    
    def objective(self, pg, qg, js, disp=False):
        self.sim_handle.marionette_robot(js)
        obj_ee_match = self.ee_match_obj(pg, qg, js)
        obj_coll_avoid = self.collision_avoid_obj(js)
        net_obj = obj_ee_match + obj_coll_avoid
        # if disp:
        #     print('--------------------------------')
        #     print(f'ee_match_obj: {obj_ee_match}')
        #     print(f'collision_avoid_obj: {obj_coll_avoid}')
        #     print(f'net_obj: {net_obj}')
        #     print('--------------------------------')

        return net_obj
    
    def refine_ik(self, pg, qg, init_js, max_gitr=5):
        objective = lambda js: self.objective(pg, qg, js)
        bounds = np.asarray([(joint.limit.lower, joint.limit.upper) for joint in self.fk_handle.actuated_joints])
        options = {'maxiter':500, 'disp':False}

        for gitr in range(max_gitr):
            print(f'Iteration: {gitr+1}')
            res = minimize(objective, init_js, method='SLSQP', bounds=bounds.tolist(), constraints=None, options=options)
            soln_js = res.x
            cost = self.objective(pg, qg, soln_js)
            # -2 is the groove obj for perfect ee_match, 0 is the obj for any collision_avoid state, so best obj is -2
            if cost < -1.9: 
                return soln_js
            # if initial ik returns a bad guess then randomize init_js within bounds and retry
            init_js = np.random.uniform(bounds[:,0], bounds[:,1])
        # no solution could be found, probably the sampled robot is a bad one
        return None

