from ompl import base as ob
from ompl import geometric as og
from ompl.util import noOutputHandler
import numpy as np


class OmplBulletPlanner():
    # Partly borrowed from https://github.com/fishbotics/atob
    def __init__(self, min_search_time=1, timeout=100, path_cost_threshold=10.0, verbose=True):
        assert min_search_time <= timeout
        self.timeout = timeout
        self.min_search_time = min_search_time
        self.path_cost_threshold = path_cost_threshold
        if not verbose:
            noOutputHandler()

    def _js_validity_check(self, sim_handle, js, ompl=True):
        if ompl:
            js_np = []
            for i in range(sim_handle.DOF):
                js_np.append(js[i])
            js_np = np.asarray(js_np, dtype=np.float32)
        else:
            js_np = np.asarray(js, dtype=np.float32)
        
        return not sim_handle.in_collision(js_np)

    def _setup_problem(self, start_js, goal_js, sim_handle, urdf_handle):
        # define the state space
        space = ob.RealVectorStateSpace(sim_handle.DOF)

        # set the boundaries on the state space from the joint limits
        bounds = ob.RealVectorBounds(sim_handle.DOF)
        for i in range(sim_handle.DOF):
            limit = urdf_handle.actuated_joints[i].limit
            low, high = limit.lower, limit.upper
            bounds.setLow(i, low)
            bounds.setHigh(i, high)
        space.setBounds(bounds)

        # space information is an object that wraps the planning space itself, as well as
        # other properties about the space. Most notably, it has a reference to
        # a collision checker able to verify a specific configuration
        space_information = ob.SpaceInformation(space)
        space_information.setStateValidityChecker(
            ob.StateValidityCheckerFn(lambda js: self._js_validity_check(sim_handle, js))
        )
        space_information.setup()

        # define a planning problem on the planning space with the given collision checker
        pdef = ob.ProblemDefinition(space_information)

        # copy the start and goal states into the OMPL representation
        start_state = ob.State(space)
        for i in range(sim_handle.DOF):
            start_state[i] = start_js[i]
        pdef.addStartState(start_state)
        goal_state = ob.GoalState(space_information)
        gstate = ob.State(space)
        for i in range(sim_handle.DOF):
            gstate[i] = goal_js[i]
        goal_state.setState(gstate)
        pdef.setGoal(goal_state)

        return space_information, pdef
    
    def _plan(self, space_information, pdef, DOF, planner='rrt_connect'):
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
                return [[path.getState(i)[j] for j in range(DOF)] 
                                for i in range(path_length)], path_cost
            else:
                print('Too costly path')

        return None, None 
    
    def plan(self, start_js, goal_js, sim_handle, urdf_handle, validate_plan=False, planner='rrt_connect'):
        space_information, pdef = self._setup_problem(start_js, goal_js, sim_handle, urdf_handle)
        path, cost = self._plan(space_information, pdef, sim_handle.DOF, planner)
        if path is None:
            return None, None

        path = np.asarray(path, dtype=np.float32)

        if validate_plan:
            # make sure none of the waypoints are in collision using a hard collision check
            for wp in path:
                for idx in range(sim_handle.DOF):
                    limit = urdf_handle.actuated_joints[idx].limit
                    if wp[idx] < limit.lower or wp[idx] > limit.upper:
                        return None, None
                if not self._js_validity_check(sim_handle, wp, ompl=False):
                    return None, None
        return path, cost
    

if __name__=='__main__':
    from benchmark import BulletCollisionChecker, PlanningBenchmarker, MetricsEvaluator
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Classical+Bullet')
    parser.add_argument('--robot_name', default='franka', type=str, help='Robot for benchmarking')
    parser.add_argument('--planner', default='ait_star', type=str, help='Classical planner')
    parser.add_argument('--visualize', default=False, type=bool, help='Whether to visualize')
    parser.add_argument('--timeout', default=20, type=int, help='Planning time')
    parser.add_argument('--obstacles', default='obs', type=str, help='Spawn obstacles')
    args = parser.parse_args()
    print(args)

    experiment_name = f'{args.planner}-bullet-{args.robot_name}-mpinet-{args.obstacles}'
    print(experiment_name)
    problem_handle = PlanningBenchmarker(robot_name=args.robot_name)
    sim_handle = BulletCollisionChecker(problem_handle.urdf_path, gui=args.visualize)
    classical_mp = OmplBulletPlanner(verbose=args.visualize, 
                               min_search_time=args.timeout/5, 
                               timeout=args.timeout, 
                               path_cost_threshold=20)
    metrics = MetricsEvaluator(problem_handle.urdf_path, 
                               experiment_name=experiment_name,
                               sim_handle=sim_handle,
                               urdf_handle=problem_handle.urdf_handle)
    metrics.setup_trajectory_manager(0, len(problem_handle))

    loaded_metrics = metrics.load(0, len(problem_handle))
    if not loaded_metrics:
        for idx in range(len(problem_handle)):
            _, obstacle_config, start_js, goal_js, _, goal_pose = problem_handle.get_problem(idx, ee_pose=args.visualize)
            if args.obstacles == 'obs':
                sim_handle.load_obstacles(obstacle_config)
            if args.visualize:
                sim_handle.set_dummy_state(goal_pose[0], goal_pose[1])

            start_time = time.perf_counter()
            traj, _ = classical_mp.plan(start_js, goal_js, sim_handle, problem_handle.urdf_handle, 
                                            planner=args.planner)
            duration = time.perf_counter() - start_time

            if traj is None:
                print(f'Evaluation Failed for {idx}')
                metrics.evaluate_trajectory(
                    idx, None, None, None, None, skip_metrics=True)
            else:
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

    metrics_evaluation = metrics.metrics()
    metrics.print_metrics(metrics_evaluation)
    if not loaded_metrics:
        metrics.save(0, len(problem_handle))
