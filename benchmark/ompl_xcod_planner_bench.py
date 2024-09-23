from ompl import base as ob
from ompl import geometric as og
from ompl.util import noOutputHandler
import numpy as np
from xcod import XCoD
from benchmark import PlanningBenchmarker, MetricsEvaluator, BulletCollisionChecker
from common import RealRobotPointSampler, RedirectStream
import datetime
import argparse
import torch
import time
import yaml


class OmplXCoDPlanner():
    # Partly borrowed from https://github.com/fishbotics/atob
    def __init__(self, coll_model, robot_point_sampler, 
                 valid_thr=0.001, 
                 path_cost_threshold=10.0, 
                 min_search_time=1,
                 timeout=100,
                 verbose=True):
        self.valid_thr = valid_thr
        self.coll_model = coll_model
        self.path_cost_threshold = path_cost_threshold
        self.robot_point_sampler = robot_point_sampler
        self.DOF = self.robot_point_sampler.DOF
        self.jl_limits = self.robot_point_sampler.jl_limits
        self.ju_limits = self.robot_point_sampler.ju_limits
        assert min_search_time <= timeout
        self.timeout = timeout
        self.min_search_time = min_search_time
        if not verbose:
            noOutputHandler()

    def _js_validity_check(self, js, obstacle_surface_pts, ompl=True):
        if ompl:
            js_np = []
            for i in range(self.DOF):
                js_np.append(js[i])
            js_np = np.asarray(js_np, dtype=np.float32)
        else:
            js_np = np.asarray(js, dtype=np.float32)

        js_th = torch.as_tensor(js_np, device='cuda')
        manip_surface_pts = self.robot_point_sampler.sample_robot_points_batch(js_th.unsqueeze(0))
        surface_pts = torch.cat((manip_surface_pts.squeeze(), obstacle_surface_pts), dim=0)
        query_indices = torch.zeros((surface_pts.shape[0],)).bool()
        query_indices[surface_pts[:,3]!=0] = True
        B, N = 1, surface_pts.shape[0]
        batch_coord = surface_pts[:,:3]
        batch_feat = surface_pts
        input_dict = dict(
            coord=batch_coord,
            feat=batch_feat,
            batch=torch.arange(B).repeat_interleave(N).to('cuda'),
            grid_size=0.01
        )
        with torch.no_grad():
            output_dict = self.coll_model(input_dict)
        seg_logits = output_dict['feat']
        query_logits = seg_logits[query_indices]
        cost = torch.sum(torch.argmax(query_logits, dim=1)).float().mean()
        return cost.item() < self.valid_thr

    def _setup_problem(self, start_js, goal_js, obstacle_surface_pts):
        # define the state space
        space = ob.RealVectorStateSpace(self.DOF)

        # set the boundaries on the state space from the joint limits
        bounds = ob.RealVectorBounds(self.DOF)
        for i in range(self.DOF):
            bounds.setLow(i, float(self.jl_limits[i]))
            bounds.setHigh(i, float(self.ju_limits[i]))
        space.setBounds(bounds)

        # space information is an object that wraps the planning space itself, as well as
        # other properties about the space. Most notably, it has a reference to
        # a collision checker able to verify a specific configuration
        space_information = ob.SpaceInformation(space)
        space_information.setStateValidityChecker(
            ob.StateValidityCheckerFn(lambda js: self._js_validity_check(js, obstacle_surface_pts))
        )
        space_information.setup()

        # define a planning problem on the planning space with the given collision checker
        pdef = ob.ProblemDefinition(space_information)

        # copy the start and goal states into the OMPL representation
        start_state = ob.State(space)
        for i in range(self.DOF):
            start_state[i] = start_js[i]
        pdef.addStartState(start_state)
        goal_state = ob.GoalState(space_information)
        gstate = ob.State(space)
        for i in range(self.DOF):
            gstate[i] = goal_js[i]
        goal_state.setState(gstate)
        pdef.setGoal(goal_state)

        return space_information, pdef
    
    def _plan(self, space_information, pdef, planner='rrt_connect'):
        # planning problem needs to know what it's optimizing for
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(space_information)
        )

        # set up the actual planner and give it the problem
        if planner == 'ait_star':
            optimizing_planner = og.AITstar(space_information)
            optimizing_planner.setProblemDefinition(pdef)
            optimizing_planner.setup()

            # solve planning problem
            solved = optimizing_planner.solve(
                    ob.plannerOrTerminationCondition(
                        ob.plannerOrTerminationCondition(
                            ob.plannerAndTerminationCondition(
                                # min solution time
                                ob.timedPlannerTerminationCondition(self.min_search_time),
                                ob.exactSolnPlannerTerminationCondition(pdef),
                            ),
                            # max solution runtime
                            ob.timedPlannerTerminationCondition(self.timeout),
                        ),
                        ob.CostConvergenceTerminationCondition(pdef),
                    )
                )
        elif planner == 'rrt_connect':
            optimizing_planner = og.RRTConnect(space_information)
            optimizing_planner.setProblemDefinition(pdef)
            optimizing_planner.setup()
            
            # solve planning problem
            solved = optimizing_planner.solve(
                    ob.plannerOrTerminationCondition(
                        ob.plannerOrTerminationCondition(
                            ob.exactSolnPlannerTerminationCondition(pdef),
                            # max solution runtime
                            ob.timedPlannerTerminationCondition(self.timeout),
                        ),
                        ob.CostConvergenceTerminationCondition(pdef),
                    )
                )
        else:
            raise NotImplementedError()
        
        path = pdef.getSolutionPath()
        if solved.asString() == 'Exact solution':
            # simplify path
            simplifier = og.PathSimplifier(space_information)
            try:
                simplifier.shortcutPath(path)
            except:
                # new in OMPL latest version
                simplifier.partialShortcutPath(path)
            simplifier.smoothBSpline(path)
            path.interpolate()
            path_cost = path.cost(ob.PathLengthOptimizationObjective(space_information)).value()
            if path_cost < self.path_cost_threshold:
                path_length = path.getStateCount()
                return [[path.getState(i)[j] for j in range(self.DOF)] 
                                for i in range(path_length)], path_cost
            else:
                print('Too costly path')

        return None, None 
    
    def plan(self, start_js, goal_js, obstacle_surface_pts, validate_plan=False, planner='rrt_connect'):                            
        space_information, pdef = self._setup_problem(start_js, goal_js, obstacle_surface_pts)
        path, cost = self._plan(space_information, pdef, planner)
        if path is None:
            return None, None
        
        path = np.asarray(path, dtype=np.float32)

        if validate_plan:
            # make sure none of the waypoints are in collision
            for wp in path:
                for idx in range(self.DOF):
                    if wp[idx] < self.jl_limits[idx] or wp[idx] > self.ju_limits[idx]:
                        return None, None
                if not self._js_validity_check(wp, obstacle_surface_pts, ompl=False):
                    return None, None
        return path, cost


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Classical+XCoD')
    parser.add_argument('--robot_name', default='franka', type=str, help='Robot for benchmarking')
    parser.add_argument('--planner', default='ait_star', type=str, help='Classical planner')
    parser.add_argument('--visualize', default=False, type=bool, help='Whether to visualize')
    parser.add_argument('--timeout', default=100, type=int, help='Planning time')
    args = parser.parse_args()
    print(args)

    experiment_name = f'{args.planner}-xcod-{args.robot_name}-mpinet-obs'
    print(experiment_name)

    problem_handle = PlanningBenchmarker(robot_name=args.robot_name, num_obstacle_points=4096*4)

    with open("config/robot_point_sampler.yaml") as file:
        robot_point_sampler = RealRobotPointSampler(
            urdf_path=problem_handle.urdf_path, 
            config=yaml.safe_load(file)['ompl_xcod_planning'],
            num_robot_points=4096,
            device='cuda')
        
    coll_model = XCoD(
        stride=[2, 2],
        enc_depths=[2, 4, 2],
        enc_channels=[32, 64, 128],
        enc_num_head=[2, 4, 8],
        enc_patch_size=[256, 256, 256],
        dec_depths=[2, 2],
        dec_channels=[32, 64],
        dec_num_head=[4, 8],
        dec_patch_size=[256, 256],
        checkpoint_path='checkpoints/ptv3_semantic_mini_best.pth',
        ).to('cuda')
    coll_model.eval()

    sim_handle = BulletCollisionChecker(problem_handle.urdf_path, gui=args.visualize)

    nc_classical_mp = OmplXCoDPlanner(
                                coll_model=coll_model,
                                robot_point_sampler=robot_point_sampler,
                                path_cost_threshold=20,
                                min_search_time=args.timeout/5, 
                                timeout=args.timeout,
                                verbose=False)

    metrics = MetricsEvaluator(problem_handle.urdf_path,
                                experiment_name=experiment_name,
                                sim_handle=sim_handle,
                                urdf_handle=problem_handle.urdf_handle)
    metrics.setup_trajectory_manager(0, len(problem_handle))

    loaded_metrics = metrics.load(0, len(problem_handle))
    if not loaded_metrics:
        net_start_time = time.perf_counter()
        with RedirectStream(f'log/{experiment_name}.log'):
            for idx in range(len(problem_handle)):
                obstacle_surface_pts, obstacle_config, start_js, goal_js, _, goal_pose = \
                    problem_handle.get_problem(idx, ee_pose=args.visualize)
                obstacle_surface_pts = torch.as_tensor(obstacle_surface_pts, dtype=torch.float32, device='cuda')
                sim_handle.load_obstacles(obstacle_config)
                
                if args.visualize:
                    sim_handle.set_dummy_state(goal_pose[0], goal_pose[1])

                start_time = time.perf_counter()
                traj, _ = nc_classical_mp.plan(start_js, goal_js, obstacle_surface_pts, planner=args.planner)
                duration = time.perf_counter() - start_time

                if traj is None:
                    print(f'Evaluation Failed for {idx}')
                    metrics.evaluate_trajectory(
                        idx, None, None, None, None, skip_metrics=True)
                else:
                    print(f'Evaluation Success for {idx}')
                    if args.visualize:
                        # visualize n_dof plan
                        time.sleep(0.5)
                        for js in traj:
                            sim_handle.marionette_robot(js)
                            time.sleep(0.05)
                        time.sleep(0.5)
                    metrics.evaluate_trajectory(
                        idx,
                        traj,
                        0.1,
                        problem_handle.urdf_handle.link_fk(cfg=goal_js, 
                                link="gripper_base_target", use_names=True),
                        duration
                    )
                
                sim_handle.remove_obstacles()
        print(f'workhorse completed in {datetime.timedelta(seconds=time.perf_counter()-net_start_time)}')

    metrics_evaluation = metrics.metrics()
    metrics.print_metrics(metrics_evaluation)
    if not loaded_metrics:
        metrics.save(0, len(problem_handle))