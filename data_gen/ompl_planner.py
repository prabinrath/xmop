from ompl import base as ob
from ompl import geometric as og
from ompl.util import noOutputHandler
from data_gen.path_interpolation import smooth_cubic
import numpy as np
from data_gen.ik_optim import CollisionIK


class OmplPlanner():
    # Partly borrowed from https://github.com/fishbotics/atob
    def __init__(self, collision_inflation=0.01, path_cost_threshold=10.0, verbose=True):
        self.collision_inflation = collision_inflation
        self.path_cost_threshold = path_cost_threshold
        if not verbose:
            noOutputHandler()

    def _js_validity_check(self, sim_handle, js, ompl=True, hard=False):
        if ompl:
            js_np = []
            for i in range(sim_handle.DOF):
                js_np.append(js[i])
            js_np = np.asarray(js_np, dtype=np.float32)
        else:
            js_np = np.asarray(js, dtype=np.float32)
        sim_handle.marionette_robot(js_np)

        if hard:
            sim_handle.perform_collision_check()
            validity = (not sim_handle.self_collision_check()) and (not sim_handle.env_collision_check())
        else:
            off = self.collision_inflation
            self_collision_min = min(sim_handle.self_collision_distance()+[float('inf')])
            env_collision_min = min(sim_handle.env_collision_distance()+[float('inf')])
            validity = (not self_collision_min < off) and (not env_collision_min < off)

        return validity

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
    
    def _ait_star_plan(self, space_information, pdef, DOF):
        # planning problem needs to know what it's optimizing for
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(space_information)
        )

        # set up the actual planner and give it the problem
        optimizing_planner = og.AITstar(space_information)
        optimizing_planner.setProblemDefinition(pdef)
        optimizing_planner.setup()

        # solve planning problem
        solved = optimizing_planner.solve(
                ob.plannerOrTerminationCondition(
                    ob.plannerOrTerminationCondition(
                        ob.plannerAndTerminationCondition(
                            # min solution time
                            ob.timedPlannerTerminationCondition(15),
                            ob.exactSolnPlannerTerminationCondition(pdef),
                        ),
                        # max solution runtime
                        ob.timedPlannerTerminationCondition(20),
                    ),
                    ob.CostConvergenceTerminationCondition(pdef),
                )
            )
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
    
    def _retime_path(self, path, sim_handle, urdf_handle, traj_resolution):
        vel_limits = [joint.limit.velocity for joint in urdf_handle.actuated_joints]
        acc_limits = [10.0,]*sim_handle.DOF
        curve = smooth_cubic(
            path,
            lambda js: self._js_validity_check(sim_handle, js, ompl=False, hard=True),
            traj_resolution * np.ones(sim_handle.DOF),
            vel_limits,
            acc_limits,
            # max_iterations=500,
        )
        # constrains max configuration space change to traj_resolution
        path, t = [], 0
        last_config = curve(t)
        path.append(last_config)
        while t<curve.x[-1]:
            curr_config = curve(t)
            # TODO: improve success rate for lower resolutions by checking for valid 
            # configurations in the neighbourhood of local_config
            if np.max(np.abs(curr_config-last_config)) > traj_resolution:
                last_config = curr_config
                path.append(last_config)
            t += curve.x[-1]/500
        path.append(curve(curve.x[-1]))
        path = np.asarray(path, dtype=np.float32)
        return path
    
    def plan(self, sim_handle, urdf_handle, eef_plan, traj_resolution=None):
        coll_ik = CollisionIK(fk_handle=urdf_handle, sim_handle=sim_handle, 
                              collision_inflation=self.collision_inflation)                              
        source, target = eef_plan[0], eef_plan[-1]
        init_js = sim_handle.robot_ik(source[0], source[1])
        start_js = coll_ik.refine_ik(source[0], source[1], init_js, max_gitr=10)
        if start_js is None:
            return None, None
        init_js = sim_handle.robot_ik(target[0], target[1])
        goal_js = coll_ik.refine_ik(target[0], target[1], init_js, max_gitr=10)
        if goal_js is None:
            return None, None

        space_information, pdef = self._setup_problem(start_js, goal_js, sim_handle, urdf_handle)
        path, cost = self._ait_star_plan(space_information, pdef, DOF=sim_handle.DOF)
        if path is None:
            return None, None

        if traj_resolution is not None:
            path = self._retime_path(path, sim_handle, urdf_handle, traj_resolution)
        else:
            path = np.asarray(path, dtype=np.float32)

        # make sure none of the waypoints are in collision using a hard collision check
        for wp in path:
            for idx in range(sim_handle.DOF):
                limit = urdf_handle.actuated_joints[idx].limit
                if wp[idx] < limit.lower or wp[idx] > limit.upper:
                    return None, None
            if not self._js_validity_check(sim_handle, wp, ompl=False, hard=True):
                return None, None
        return path, cost