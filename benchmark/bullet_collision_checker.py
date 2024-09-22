from common import BulletRobotEnv


class BulletCollisionChecker(BulletRobotEnv):
    def __init__(self, urdf_path, gui=False):
        super().__init__(gui=gui, base_link='base_link')
        assert 'sample.urdf' in urdf_path # ensure modified urdf
        self.load_robot(urdf_path)
    
    def in_collision(self, js):
        assert js.shape[0] == self.DOF
        self.marionette_robot(js)
        self.perform_collision_check()   
        env_collision = False    
        if self.obstacle_ids is not None:
            env_collision = self.env_collision_check() 
        return self.self_collision_check() or env_collision
    
    def in_self_collision(self, js):
        assert js.shape[0] == self.DOF
        self.marionette_robot(js)
        self.perform_collision_check()
        return self.self_collision_check()
    
    def in_env_collision(self, js):
        assert js.shape[0] == self.DOF
        self.marionette_robot(js)
        self.perform_collision_check()
        return self.env_collision_check() 
    
    def in_collision_complete(self, js):
        assert js.shape[0] == self.DOF
        self.marionette_robot(js)
        self.perform_collision_check()        
        return self.complete_collision_check()  
        

if __name__=='__main__':
    from training import MpiNetDataset
    from urdfpy import URDF
    import numpy as np
    
    # URDF_PATH = 'urdf/sawyer/sawyer_sample.urdf'
    URDF_PATH = 'urdf/franka_panda/panda_motion_bench_sample.urdf'
    # URDF_PATH = 'urdf/ur_robots/ur5_sample.urdf'
    # URDF_PATH = 'urdf/kuka_iiwa14/iiwa14_sample.urdf'
    # URDF_PATH = 'urdf/kinova/kinova_6dof_sample.urdf'

    sim_handle = BulletCollisionChecker(URDF_PATH, gui=True)

    urdf_handle = URDF.load(URDF_PATH)
    jl_limits, ju_limits = [], []
    for i in range(sim_handle.DOF):
        limit = urdf_handle.actuated_joints[i].limit
        jl_limits.append(limit.lower)
        ju_limits.append(limit.upper)
    jl_limits = np.asarray(jl_limits)
    ju_limits = np.asarray(ju_limits)
    
    mpinet_dataset = MpiNetDataset('global_solutions', 
                               'resources/datasets/mpinet_dataset/train.hdf5', 
                               'urdf/franka_panda/panda.urdf',
                               sample_color=True)
    random_indices = np.random.choice(len(mpinet_dataset), 10, replace=False)

    for idx in random_indices:
        (_, _), obstacle_config, _ = mpinet_dataset[idx]
        sim_handle.load_obstacles(obstacle_config)

        js = np.random.uniform(jl_limits, ju_limits)

        print(f'hard: {sim_handle.in_collision(js)}')
        print(list(sim_handle.in_collision_complete(js)))

        sim_handle.remove_obstacles()