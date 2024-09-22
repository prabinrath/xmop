import os
import pickle
from urdfpy import URDF
from common.sample_surface_points import sample_surface_points
from geometrout.primitive import Cuboid, Cylinder
from pyquaternion import Quaternion


class PlanningBenchmarker():
    def __init__(self, robot_name='franka', num_obstacle_points=None):
        robot_urdfs = {
            'sawyer': 'urdf/sawyer/sawyer_sample.urdf',
            'franka': 'urdf/franka_panda/panda_mpinet_bench_sample.urdf',
            'ur5': 'urdf/ur_robots/ur5_sample.urdf',
            'ur10': 'urdf/ur_robots/ur10_sample.urdf',
            'kuka': 'urdf/kuka_iiwa14/iiwa14_motion_bench_sample.urdf',
            'kinova6': 'urdf/kinova/kinova_6dof_sample.urdf',
            'kinova7': 'urdf/kinova/kinova_7dof_sample.urdf'
        }
        self.urdf_path = robot_urdfs[robot_name]
        self.urdf_handle = URDF.load(self.urdf_path)
        self.num_obstacle_points = num_obstacle_points
        self.planning_problems = []

        if os.path.exists(f'resources/benchmarks/mpinet_bench_marker/{robot_name}_mpinet_benchmarker_problems.pkl'):
            with open(f'resources/benchmarks/mpinet_bench_marker/{robot_name}_mpinet_benchmarker_problems.pkl', 'rb') as file:
                self.planning_problems = pickle.load(file)
        else:
            raise Exception('Benchmark Not Found')
        
        assert len(self.planning_problems) > 0, 'Loading Benchmark Failed'
    
    def get_problem(self, problem_idx, ee_pose=False):
        problem = self.planning_problems[problem_idx]

        if ee_pose:
            start_pose = self.urdf_handle.link_fk(cfg=problem['start_js'], 
                                                link='gripper_base_target', use_names=True)
            start_position = start_pose[:3,3]
            qt = Quaternion(matrix=start_pose[:3,:3])
            start_pose = (start_position, qt.q)

            goal_pose = self.urdf_handle.link_fk(cfg=problem['goal_js'], 
                                                link='gripper_base_target', use_names=True)
            goal_position = goal_pose[:3,3]
            qt = Quaternion(matrix=goal_pose[:3,:3])
            goal_pose = (goal_position, qt.q)
        else:
            start_pose, goal_pose = None, None

        obstacle_primitives = []
        for obstacle in problem['obstacle_config']:
            if obstacle['type'] == 'cuboid':
                obstacle_primitives.append(Cuboid(center=obstacle['translation'], 
                                                  dims=obstacle['scale'], 
                                                  quaternion=obstacle['orientation']))
            elif obstacle['type'] == 'cylinder':
                obstacle_primitives.append(Cylinder(center=obstacle['translation'], 
                                                    radius=obstacle['radius'], 
                                                    height=obstacle['height'], 
                                                    quaternion=obstacle['orientation']))
                
        if self.num_obstacle_points is not None:
            obstacle_surface_points, _ = sample_surface_points(obstacle_primitives, self.num_obstacle_points)
        else: 
            obstacle_surface_points = None
                
        return obstacle_surface_points, problem['obstacle_config'], \
            problem['start_js'], problem['goal_js'], start_pose, goal_pose


if __name__=='__main__':
    from benchmark import BulletCollisionChecker
    import time

    problem_handle = PlanningBenchmarker(robot_name='ur10')
    sim_handle = BulletCollisionChecker(problem_handle.urdf_path, gui=True)

    for idx in range(len(problem_handle)):
        _, obstacle_config, start_js, goal_js, start_pose, goal_pose = \
            problem_handle.get_problem(idx, ee_pose=True)
        sim_handle.load_obstacles(obstacle_config)
        assert not sim_handle.in_collision(start_js)
        sim_handle.set_dummy_state(start_pose[0], start_pose[1])
        time.sleep(0.5)
        assert not sim_handle.in_collision(goal_js)
        sim_handle.set_dummy_state(goal_pose[0], goal_pose[1])
        time.sleep(0.5)
        sim_handle.remove_obstacles()