# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import numpy as np
import pickle
from pathlib import Path
from pyquaternion import Quaternion
from urdfpy import URDF
from benchmark.bullet_collision_checker import BulletCollisionChecker
from common import TrajDataManager
from pathlib import Path
from typing import Sequence, Tuple, Dict
from .sparc_smoothness_metric import sparc


class MetricsEvaluator:
    """
    This class can be used to evaluate a whole set of environments and data
    """

    def __init__(self, urdf_path: str, 
                 experiment_name: str = 'evaluate', 
                 result_dir: str = 'benchmark/results', 
                 sim_handle=None,
                 urdf_handle=None):
        """
        Initializes the evaluator class
        """
        assert 'sample.urdf' in urdf_path # ensure modified urdf
        self.result_dir = os.path.join(result_dir, experiment_name)
        self.experiment_name = experiment_name
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        self.sim_handle = BulletCollisionChecker(urdf_path) if sim_handle is None else sim_handle
        self.urdf_handle = URDF.load(urdf_path) if urdf_handle is None else urdf_handle
        self.metadata = {'experiment_name': experiment_name, 'urdf_path': urdf_path, 'problems': {}}
        self.traj_mgr = None
    
    def setup_trajectory_manager(self, start_idx, end_idx):
        if os.path.exists(os.path.join(self.result_dir, f'traj_{start_idx}_{end_idx}.h5')):
            self.traj_mgr = TrajDataManager(self.result_dir, start_idx, end_idx)
        else:
            self.traj_mgr = TrajDataManager(self.result_dir, start_idx, end_idx, mode='w')
    
    def save(self, start_idx, end_idx):
        with open(os.path.join(self.result_dir, f'{self.experiment_name}_{start_idx}_{end_idx}.pkl'), 'wb') as file:
            pickle.dump(self.metadata, file)
    
    def load(self, start_idx, end_idx):
        try:
            with open(os.path.join(self.result_dir, f'{self.experiment_name}_{start_idx}_{end_idx}.pkl'), 'rb') as file:
                self.metadata = pickle.load(file)
        except:
            return False
        return True

    def has_self_collision(self, trajectory) -> bool:
        """
        Checks whether there is a self collision (using OR between different methods)

        :param trajectory : The trajectory
        :rtype bool: Whether there is a self collision
        """
        assert isinstance(trajectory, np.ndarray)
        for js in trajectory:
            if self.sim_handle.in_self_collision(js):
                return True
        return False
    
    def has_env_collision(
        self, trajectory
    ) -> bool:
        """
        Checks whether there is environment collision

        :param trajectory : The trajectory
        :param obstacles : The obstacles in the scene
        :rtype bool: Whether there is environment collision
        """
        if self.sim_handle.obstacle_ids is None:
            return False
        
        assert isinstance(trajectory, np.ndarray)
        for js in trajectory:
            if self.sim_handle.in_env_collision(js):
                return True
        return False

    def violates_joint_limits(self, trajectory) -> bool:
        """
        Checks whether any configuration in the trajectory violates joint limits

        :param trajectory : The trajectory
        :rtype bool: Whether there is a joint limit violation
        """
        jl_limits, ju_limits = [], []
        for i in range(self.sim_handle.DOF):
            limit = self.urdf_handle.actuated_joints[i].limit
            jl_limits.append(limit.lower)
            ju_limits.append(limit.upper)
        
        return np.any(trajectory < np.asarray(jl_limits)) or \
            np.any(trajectory > np.asarray(ju_limits))

    def check_final_position(self, final_pose, goal_pose) -> float:
        """
        Gets the number of centimeters between the final pose and target

        :param final_pose : The final pose of the trajectory
        :param target : The target pose
        :rtype float: The distance in centimeters
        """
        return 100 * np.linalg.norm(final_pose[:3,3] - goal_pose[:3,3])

    def check_final_orientation(self, final_pose, goal_pose) -> float:
        """
        Gets the number of degrees between the final orientation and the target orientation

        :param final_orientation : The final orientation
        :param target : The final target orientation
        :rtype float: The rotational distance in degrees
        """
        final_orientation = Quaternion(matrix=final_pose[:3,:3]).unit
        goal_orientation = Quaternion(matrix=goal_pose[:3,:3]).unit
        return np.abs((final_orientation * goal_orientation.conjugate).radians * 180 / np.pi)

    def calculate_smoothness(self, trajectory, dt: float) -> Tuple[float, float]:
        """
        Calculate trajectory smoothness using SPARC

        :param trajectory : The trajectory
        :param dt float: The timestep in between consecutive steps of the trajectory
        :rtype Tuple[float, float]: The SPARC in configuration space and end effector space
        """
        configs = np.asarray(trajectory)
        assert configs.ndim == 2 and configs.shape[1] == self.sim_handle.DOF
        config_movement = np.linalg.norm(np.diff(configs, 1, axis=0) / dt, axis=1)
        assert len(config_movement) == len(configs) - 1
        config_sparc, _, _ = sparc(config_movement, 1.0 / dt)
        
        eff_positions = self.urdf_handle.link_fk_batch(cfgs=trajectory, link='gripper_base_target')[:,:3,3]
        assert eff_positions.ndim == 2 and eff_positions.shape[1] == 3
        eff_movement = np.linalg.norm(np.diff(eff_positions, 1, axis=0) / dt, axis=1)
        assert len(eff_movement) == len(eff_positions) - 1
        eff_sparc, _, _ = sparc(eff_movement, 1.0 / dt)

        return config_sparc, eff_sparc
    
    def calculate_config_path_length(self, trajectory) -> float:
        return sum(np.linalg.norm(np.diff(trajectory, 1, axis=0), axis=1))

    def calculate_eff_path_lengths(self, trajectory) -> Tuple[float, float]:
        """
        Calculate the end effector path lengths (position and orientation).
        Orientation is in degrees.

        :param trajectory : The trajectory
        :rtype Tuple[float, float]: The path lengths (position, orientation)
        """
        eff_poses = self.urdf_handle.link_fk_batch(cfgs=trajectory, link='gripper_base_target')

        eff_positions = eff_poses[:,:3,3]
        assert eff_positions.ndim == 2 and eff_positions.shape[1] == 3
        position_step_lengths = np.linalg.norm(
            np.diff(eff_positions, 1, axis=0), axis=1
        )
        eff_position_path_length = sum(position_step_lengths)

        eff_quaternions = [Quaternion(matrix=pose[:3,:3]) for pose in eff_poses]
        eff_orientation_path_length = 0
        for qi, qj in zip(eff_quaternions[:-1], eff_quaternions[1:]):
            eff_orientation_path_length += np.abs(
                np.degrees((qj * qi.conjugate).radians)
            )
        return eff_position_path_length, eff_orientation_path_length

    def percent_true(self, arr: Sequence) -> float:
        """
        Returns the percent true of a boolean sequence or the percent nonzero of a numerical sequence

        :param arr Sequence: The input sequence
        :rtype float: The percent
        """
        return 100 * np.count_nonzero(arr) / len(arr)

    def metric_array(self, key, indices=None):
        metrics = []
        problem_indices = self.metadata['problems'].keys() if indices is None else indices
        for problem_idx in problem_indices:
            if key in self.metadata['problems'][problem_idx]:
                metrics.append(self.metadata['problems'][problem_idx][key])
        return np.asarray(metrics)

    def evaluate_trajectory(
        self,
        problem_idx,
        trajectory,
        dt,
        goal_pose,
        time: float,
        position_error_thresh=1,
        orientation_error_thresh=5,
        sparc_smoothness_thresh=-1.6,
        skip_metrics: bool = False,
        save_traj: bool = True
    ):
        """
        Evaluates a single trajectory and stores the metrics in metadata.
        Will visualize and print relevant info if `self.gui` is `True`

        :param trajectory : The trajectory
        :param dt float: The time step for the trajectory
        :param target : The target pose
        :param obstacle_config : The obstacles in the scene
        :param time float: The time taken to calculate the trajectory
        :param skip_metrics bool: Whether to skip the path metrics (for example if it's a feasibility planner that failed)
        """
        # print(f'Evaluating problem {problem_idx}')
        self.metadata['problems'][problem_idx] = {}
        
        def add_metric(key, value):
            self.metadata['problems'][problem_idx][key] = value
        
        def get_metric(key):
            return self.metadata['problems'][problem_idx][key]

        if skip_metrics:
            add_metric("success", False)
            add_metric("eff_position_path_length", np.inf)
            add_metric("eff_orientation_path_length", np.inf)
            add_metric("config_path_length", np.inf)
            add_metric("num_steps", np.inf)
            add_metric("time", np.inf)
            return
        
        if self.traj_mgr is not None and save_traj:
            try:
                self.traj_mgr.save_trajectory(problem_idx, 
                                            np.zeros((trajectory.shape[1], 10)), 
                                            np.zeros((trajectory.shape[1], 7)), 
                                        trajectory)
            except:
                print('Skipping trajectory save operation')

        add_metric("joint_limit_violation", self.violates_joint_limits(trajectory))
        add_metric("self_collision", self.has_self_collision(trajectory))
        add_metric("env_collision", self.has_env_collision(trajectory))

        final_pose = self.urdf_handle.link_fk(trajectory[-1], link="gripper_base_target")
        position_error = self.check_final_position(final_pose, goal_pose)
        add_metric("position_error", position_error)
        orientation_error = self.check_final_orientation(final_pose, goal_pose)
        add_metric("orientation_error", orientation_error)

        config_smoothness, eff_smoothness = self.calculate_smoothness(trajectory, dt)
        add_metric("config_smoothness", config_smoothness)
        add_metric("eff_smoothness", eff_smoothness)

        (
            eff_position_path_length,
            eff_orientation_path_length,
        ) = self.calculate_eff_path_lengths(trajectory)
        add_metric("eff_position_path_length", eff_position_path_length)
        add_metric("eff_orientation_path_length", eff_orientation_path_length)
        config_path_length = self.calculate_config_path_length(trajectory)
        add_metric("config_path_length", config_path_length)

        physical_violation = get_metric('joint_limit_violation') or \
                                  get_metric('self_collision') or \
                                  get_metric('env_collision')
        success = (
            position_error < position_error_thresh
            and orientation_error < orientation_error_thresh
            and not physical_violation
        )

        add_metric("success", success)
        add_metric("time", time)
        add_metric("num_steps", trajectory.shape[0])

    def metrics(self, indices=None) -> Dict[str, float]:
        """
        Calculates the metrics for the experiment
        :rtype Dict[str, float]: The metrics
        """
        success = self.percent_true(self.metric_array("success",indices))
        one_cm = self.percent_true(self.metric_array("position_error",indices) < 1)
        three_cm = self.percent_true(self.metric_array("position_error",indices) < 3)
        five_cm = self.percent_true(self.metric_array("position_error",indices) < 5)
        one_deg = self.percent_true(self.metric_array("orientation_error",indices) < 1)
        three_deg = self.percent_true(self.metric_array("orientation_error",indices) < 3)
        five_deg = self.percent_true(self.metric_array("orientation_error",indices) < 5)

        config_smoothness = np.mean(self.metric_array("config_smoothness",indices))
        eff_smoothness = np.mean(self.metric_array("eff_smoothness",indices))
        all_eff_position_path_lengths = self.metric_array("eff_position_path_length",indices)
        all_eff_orientation_path_lengths = self.metric_array("eff_orientation_path_length",indices)
        all_config_path_lengths = self.metric_array("config_path_length",indices)
        all_times = self.metric_array("time",indices)

        is_smooth = self.percent_true(
            np.logical_and(
                self.metric_array("config_smoothness",indices) < -1.6,
                self.metric_array("eff_smoothness",indices) < -1.6,
            )
        )

        successes = self.metric_array("success",indices)
        success_position_path_lengths = all_eff_position_path_lengths[successes]
        success_orientation_path_lengths = all_eff_orientation_path_lengths[successes]
        success_config_path_lengths = all_config_path_lengths[successes]
        eff_position_path_length = (
            np.mean(success_position_path_lengths),
            np.std(success_position_path_lengths),
        )
        eff_orientation_path_length = (
            np.mean(success_orientation_path_lengths),
            np.std(success_orientation_path_lengths),
        )
        config_path_length = (
            np.mean(success_config_path_lengths),
            np.std(success_config_path_lengths),
        )
        success_times = all_times[successes]
        time = (
            np.mean(success_times),
            np.std(success_times),
        )
        
        self_collision_array, env_collision_array, joint_limit_array = \
            self.metric_array("self_collision",indices), \
            self.metric_array("env_collision",indices), \
            self.metric_array("joint_limit_violation",indices)
        self_collision = self.percent_true(self_collision_array)
        env_collision = self.percent_true(env_collision_array)
        joint_limit = self.percent_true(joint_limit_array)
        physical_violation = self.percent_true(np.any(np.vstack([self_collision_array, 
                                                                 env_collision_array, 
                                                                 joint_limit_array]), axis=0))

        all_num_steps = self.metric_array("num_steps",indices)
        success_num_steps = all_num_steps[successes]
        step_time = (
            np.mean(success_times / success_num_steps),
            np.std(success_times / success_num_steps),
        )
        return {
            "success": success,
            "total": successes.shape[0],
            "time": time,
            "step time": step_time,
            "env collision": env_collision,
            "self collision": self_collision,
            "joint violation": joint_limit,
            "physical violation": physical_violation,
            "1 cm": one_cm,
            "3 cm": three_cm,
            "5 cm": five_cm,
            "1 deg": one_deg,
            "3 deg": three_deg,
            "5 deg": five_deg,            
            "is smooth": is_smooth,
            "average config sparc": config_smoothness,
            "average eff sparc": eff_smoothness,
            "eff position path length": eff_position_path_length,
            "eff orientation path length": eff_orientation_path_length,
            "config path length": config_path_length
        }

    def print_metrics(self, metrics):
        """
        Prints the metrics in an easy to read format
        """
        print(f"Total problems: {metrics['total']}")
        print(f"% Success: {metrics['success']:4.2f}")
        print(f"% Within 1cm: {metrics['1 cm']:4.2f}")
        print(f"% Within 3cm: {metrics['3 cm']:4.2f}")
        print(f"% Within 5cm: {metrics['5 cm']:4.2f}")
        print(f"% Within 1deg: {metrics['1 deg']:4.2f}")
        print(f"% Within 3deg: {metrics['3 deg']:4.2f}")
        print(f"% Within 5deg: {metrics['5 deg']:4.2f}")
        print(f"% With Environment Collision: {metrics['env collision']:4.2f}")
        print(f"% With Self Collision: {metrics['self collision']:4.2f}")
        print(f"% With Joint Limit Violations: {metrics['joint violation']:4.2f}")
        print(f"% Physical Violations: {metrics['physical violation']:4.2f}")
        print(f"Average Config SPARC: {metrics['average config sparc']:4.2f}")
        print(f"Average End Eff SPARC: {metrics['average eff sparc']:4.2f}")
        print(f"% Smooth: {metrics['is smooth']:4.2f}")
        print(
            "Average End Eff Position Path Length:"
            f" {metrics['eff position path length'][0]:4.2f}"
            f" ± {metrics['eff position path length'][1]:4.2f}"
        )
        print(
            "Average End Eff Orientation Path Length:"
            f" {metrics['eff orientation path length'][0]:4.2f}"
            f" ± {metrics['eff orientation path length'][1]:4.2f}"
        )
        print(
            "Average Config Path Length:"
            f" {metrics['config path length'][0]:4.2f}"
            f" ± {metrics['config path length'][1]:4.2f}"
        )
        print(f"Average Time: {metrics['time'][0]:4.2f} ± {metrics['time'][1]:4.2f}")
        print(
            "Average Time Per Step (Not Always Valuable):"
            f" {metrics['step time'][0]:4.6f}"
            f" ± {metrics['step time'][1]:4.6f}"
        )
