from xmop import XMoP
from benchmark import PlanningBenchmarker, MetricsEvaluator, BulletCollisionChecker
from common import RedirectStream
import datetime
import argparse
import time


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='XMoP+XCoD')
    parser.add_argument('--robot_name', default='franka', type=str, help='Robot for benchmarking')
    parser.add_argument('--visualize', default=False, type=bool, help='Whether to visualize')
    parser.add_argument('--max_rollout_steps', default=200, type=int, help='Planning time')
    parser.add_argument('--gpu', default=0, type=int, help='Gpu to use')
    args = parser.parse_args()
    print(args)

    experiment_name = f'multistep-xmop-{args.robot_name}-mpinet-obs'
    print(experiment_name)

    problem_handle = PlanningBenchmarker(robot_name=args.robot_name, num_obstacle_points=4096*2)
    sim_handle = BulletCollisionChecker(problem_handle.urdf_path, gui=args.visualize)

    planner_config = dict(
        mode='multistep',
        urdf_path=problem_handle.urdf_path,
        config_folder='config/',
        ckpt_path='checkpoints/',
        smoothing_factor=0.25
    )
    neural_planner = XMoP(planner_config, device=f'cuda:{args.gpu}')

    metrics = MetricsEvaluator(problem_handle.urdf_path,
                                experiment_name=experiment_name,
                                sim_handle=sim_handle,
                                urdf_handle=problem_handle.urdf_handle)
    loaded_metrics = metrics.load(0, len(problem_handle))
    if not loaded_metrics:
        net_start_time = time.perf_counter()
        with RedirectStream(f'log/{experiment_name}.log'):
            for idx in range(len(problem_handle)):
                obstacle_surface_pts, obstacle_config, start_js, goal_js, _, goal_pose = \
                    problem_handle.get_problem(idx, ee_pose=True)
                sim_handle.load_obstacles(obstacle_config)
                
                if args.visualize:
                    sim_handle.set_dummy_state(goal_pose[0], goal_pose[1])

                observation = dict(
                    curr_js=start_js,
                    obstacle_surface_pts=obstacle_surface_pts,
                    goal_eef=goal_pose,
                )
                start_time = time.perf_counter()
                traj = neural_planner.plan_offline(observation, args.max_rollout_steps, goal_thresh=0.01, exact=False)
                duration = time.perf_counter() - start_time

                if traj is None:
                    print(f'Planning Failed for {idx}')
                    metrics.evaluate_trajectory(
                        idx, None, None, None, None, skip_metrics=True)
                else:
                    print(f'Planning Success for {idx}')
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