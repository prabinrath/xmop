from urdfpy import URDF
import numpy as np
from scipy.optimize import minimize
import time


class JointStateRetriever():
    """
    retrieves joint configuration for a robotic manipulator using full-body IK

    :urdf_handle: URDF handle from urdfpy
    :solver_options: solver options for SLSQP from scipy
    """
    def __init__(self, urdf_handle, solver_options=None, cost_th=0.1, max_itrs=10, limit_factor=0.95):
        if solver_options is not None:
            self.solver_options = solver_options
        else:
            self.solver_options = {'maxiter':100, 'disp':False}
        self.urdf_handle = urdf_handle
        self.bounds = np.asarray([(joint.limit.lower, joint.limit.upper) 
                                  for joint in self.urdf_handle.actuated_joints]) * limit_factor
        self.cost_th = cost_th
        self.max_itrs = max_itrs

    def _match_obj(self, js, target_poses_9d):
        fk_dict = self.urdf_handle.visual_geometry_fk(js)
        net_error = 0
        for key, target_pose_9d in target_poses_9d.items():
            pose = fk_dict[key]
            pose_9d = np.concatenate((pose[:3,3], pose[:3,0], pose[:3,1]))
            net_error += np.linalg.norm(target_pose_9d-pose_9d)
        # print(net_error)
        return net_error
    
    def retrieve_js(self, target_poses_9d, init_js):
        objective = lambda js: self._match_obj(js, target_poses_9d)
        soln_js = None
        init_js_bkp = np.copy(init_js)
        for _ in range(self.max_itrs):         
            res = minimize(objective, init_js, method='SLSQP', 
                           bounds=self.bounds.tolist(), 
                           constraints=None, 
                           options=self.solver_options)
            if res.fun < self.cost_th:
                # print(res.fun)
                soln_js = res.x
                break
            # if IK fails then sample another configuration 
            # from the vicinity of last init_js and retry
            init_js = np.random.normal(init_js, 0.1*np.ones((init_js.shape[0],))) 
        if soln_js is None:
            print('Joint state retrieval failed')
            soln_js = init_js_bkp
        return soln_js


if __name__=='__main__':
    # URDF_PATH = 'urdf/sawyer/sawyer_sample.urdf'
    URDF_PATH = 'urdf/franka_panda/panda_sample.urdf'
    # URDF_PATH = 'urdf/ur_robots/ur10_sample.urdf'
    # URDF_PATH = 'urdf/kuka_iiwa14/iiwa14_sample.urdf'
    # URDF_PATH = 'urdf/kinova/kinova_6dof_sample.urdf'

    urdf_handle = URDF.load(URDF_PATH)
    js_ret = JointStateRetriever(urdf_handle=urdf_handle)

    duration_list = []
    for itr in range(100):
        js_gt = np.random.uniform(js_ret.bounds[:,0], js_ret.bounds[:,1])
        fk_dict_gt = urdf_handle.visual_geometry_fk(js_gt)
        target_poses_9d = {}
        for key, pose in fk_dict_gt.items():
            target_poses_9d[key] = np.concatenate((pose[:3,3], pose[:3,0], pose[:3,1]))

        # init js in the vicinity current js
        init_js = np.random.normal(js_gt, 0.1*np.ones((js_gt.shape[0],))) 
        start = time.perf_counter()
        soln_js = js_ret.retrieve_js(target_poses_9d, init_js)
        duration_list.append(time.perf_counter()-start)
        js_error = np.linalg.norm(soln_js-js_gt)
        if js_error>1e-3:
            print('HIGH_ERROR')
        print('----------------------------------------------')
        print(f'found soln {itr} in {duration_list[-1]}s')
        print(f'js error is {js_error}')

    print(f'avg time: {sum(duration_list)/len(duration_list)}')
    print(f'max time: {max(duration_list)}')
    print(f'min time: {min(duration_list)}')