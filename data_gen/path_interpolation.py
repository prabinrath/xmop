import numpy as np
from scipy.interpolate import CubicHermiteSpline, PPoly
import time
from collections import namedtuple, deque
import math
import random

# Code is taken from here: https://github.com/caelan/motion-planners/blob/e15c6b0b6a2b92d781bb3a436b724bb9fddc0e23/motion_planners/trajectory/smooth.py

Interval = namedtuple("Interval", ["lower", "upper"])  # AABB
EPSILON = 1e-6


def find(test, sequence):
    for item in sequence:
        if test(item):
            return item
    raise RuntimeError()


def find_lower_bound(x1, x2, v1=None, v2=None, v_max=None, a_max=None, ord=np.inf):
    d = len(x1)
    if v_max is None:
        v_max = np.full(d, np.inf)
    if a_max is None:
        a_max = np.full(d, np.inf)
    lower_bounds = [
        # Instantaneously accelerate
        np.linalg.norm(
            np.divide(np.subtract(x2, x1), v_max), ord=ord
        ),  # quickest_inf_accel
    ]
    if (v1 is not None) and (v2 is not None):
        lower_bounds.extend(
            [
                np.linalg.norm(np.divide(np.subtract(v2, v1), a_max), ord=ord),
            ]
        )
    return max(lower_bounds)


def elapsed_time(start_time):
    return time.time() - start_time


def get_pairs(sequence):
    sequence = list(sequence)
    return list(zip(sequence[:-1], sequence[1:]))


def check_time(t):
    return not isinstance(t, complex) and (t >= 0.0)


def get_sign(x):
    if x > 0:
        return +1
    if x < 0:
        return -1
    return x


def get_delta(q1, q2):
    return np.array(q2) - np.array(q1)


def get_difference(q2, q1):
    return get_delta(q1, q2)


def remove_redundant(path, tolerance=1e-3):
    assert path
    new_path = [path[0]]
    for conf in path[1:]:
        difference = get_difference(new_path[-1], np.array(conf))
        if not np.allclose(
            np.zeros(len(difference)), difference, atol=tolerance, rtol=0
        ):
            new_path.append(conf)
    return new_path


def get_unit_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return np.array(vec) / norm


def waypoints_from_path(path, difference_fn=None, tolerance=1e-3):
    if difference_fn is None:
        difference_fn = get_difference
    path = remove_redundant(path, tolerance=tolerance)
    if len(path) < 2:
        return path
    waypoints = [path[0]]
    last_conf = path[1]
    last_difference = get_unit_vector(difference_fn(last_conf, waypoints[-1]))
    for conf in path[2:]:
        difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        if not np.allclose(last_difference, difference, atol=tolerance, rtol=0):
            waypoints.append(last_conf)
            difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        last_conf = conf
        last_difference = difference
    waypoints.append(last_conf)
    return waypoints


def bisect(sequence):
    sequence = list(sequence)
    indices = set()
    queue = deque([(0, len(sequence) - 1)])
    while queue:
        lower, higher = queue.popleft()
        if lower > higher:
            continue
        index = int((lower + higher) / 2.0)
        assert index not in indices
        # if is_even(higher - lower):
        yield sequence[index]
        queue.extend(
            [
                (lower, index - 1),
                (index + 1, higher),
            ]
        )


def bisect_selector(path):
    return bisect(path)


default_selector = bisect_selector  # random_selector


def get_times(curve):
    # TODO: rename these
    return curve.x


def spline_start(spline):
    return get_times(spline)[0]


def spline_end(spline):
    return get_times(spline)[-1]


def spline_duration(spline):
    return spline_end(spline) - spline_start(spline)


def get_interval(curve, start_t=None, end_t=None):
    if start_t is None:
        start_t = spline_start(curve)
    if end_t is None:
        end_t = spline_end(curve)
    start_t = max(start_t, spline_start(curve))
    end_t = min(end_t, spline_end(curve))
    assert start_t < end_t
    return Interval(start_t, end_t)


def even_space(start, stop, step=1, endpoint=True):
    sequence = np.arange(start, stop, step=step)
    if not endpoint:
        return sequence
    return np.append(sequence, [stop])


def sample_discretize_curve(positions_curve, resolutions, time_step=1e-2, **kwargs):
    start_t, end_t = get_interval(positions_curve, **kwargs)
    times = [start_t]
    samples = [positions_curve(start_t)]
    for t in even_space(start_t, end_t, step=time_step):
        positions = positions_curve(t)
        if (
            np.less_equal(samples[-1] - resolutions, positions).all()
            and np.less_equal(positions, samples[-1] + resolutions).all()
        ):
            continue
        times.append(t)
        samples.append(positions)
    return times, samples


def check_curve(p_curve, x1, x2, v1, v2, T, v_max=np.inf, a_max=np.inf):
    assert p_curve is not None
    end_times = np.append(p_curve.x[:1], p_curve.x[-1:])
    v_curve = p_curve.derivative()
    # print()
    # print(x1, x2, v1, v2, T, v_max, a_max)
    # print([x1, x2], [float(p_curve(t)) for t in end_times])
    # print([v1, v2], [float(v_curve(t)) for t in end_times])
    if not np.allclose([0.0, T], end_times):
        raise RuntimeError([0.0, T], end_times)
    if not np.allclose([x1, x2], [float(p_curve(t)) for t in end_times]):
        raise RuntimeError([x1, x2], [float(p_curve(t)) for t in end_times])
    if not np.allclose([v1, v2], [float(v_curve(t)) for t in end_times]):
        raise RuntimeError([v1, v2], [float(v_curve(t)) for t in end_times])
    all_times = p_curve.x
    if not all(abs(v_curve(t)) <= abs(v_max) + EPSILON for t in all_times):
        raise RuntimeError(abs(v_max), [abs(v_curve(t)) for t in all_times])
    a_curve = v_curve.derivative()
    if not all(abs(a_curve(t)) <= abs(a_max) + EPSILON for t in all_times):
        raise RuntimeError(abs(a_max), [abs(a_curve(t)) for t in all_times])
    # TODO: check continuity
    return


def curve_from_controls(durations, accels, t0=0.0, x0=0.0, v0=0.0):
    assert len(durations) == len(accels)
    # from numpy.polynomial import Polynomial
    # t = Polynomial.identity()
    times = [t0]
    positions = [x0]
    velocities = [v0]
    coeffs = []
    for duration, accel in zip(durations, accels):
        assert duration >= 0.0
        coeff = [0.5 * accel, 1.0 * velocities[-1], positions[-1]]  # 0. jerk
        coeffs.append(coeff)
        times.append(times[-1] + duration)
        p_curve = np.poly1d(coeff)  # Not centered
        positions.append(p_curve(duration))
        v_curve = p_curve.deriv()  # Not centered
        velocities.append(v_curve(duration))
    # print(positions)
    # print(velocities)

    # np.piecewise
    # max_order = max(p_curve.order for p_curve in p_curves)
    # coeffs = np.zeros([max_order + 1, len(p_curves), 1])
    # for i, p_curve in enumerate(p_curves):
    #     # TODO: need to center
    #     for k, c in iterate_poly1d(p_curve):
    #         coeffs[max_order - k, i] = c
    # TODO: check continuity

    # from scipy.interpolate import CubicHermiteSpline
    return PPoly(c=np.array(coeffs).T, x=times)  # TODO: spline.extend


def zero_one_fixed(x1, x2, T, v_max=np.inf, dt=EPSILON):
    # assert 0 < v_max < INF
    sign = get_sign(x2 - x1)
    d = abs(x2 - x1)
    v = d / T
    if v > (v_max + EPSILON):
        return None
    t_hold = T - 2 * dt
    assert t_hold > 0
    v = d / t_hold
    # return zero_three_stage(x1, x2, T, v_max=v_max, a_max=1e6) # NOTE: approximation
    coeffs = [[0.0, x1], [sign * v, x1], [0.0, x2]]
    times = [0.0, dt, dt + t_hold, T]
    p_curve = PPoly(c=np.array(coeffs).T, x=times)  # Not differentiable
    return p_curve


def zero_two_ramp(x1, x2, T, v_max=np.inf, a_max=np.inf):
    sign = get_sign(x2 - x1)
    d = abs(x2 - x1)
    t_accel = T / 2.0
    a = d / t_accel ** 2  # Lower accel
    if a > (a_max + EPSILON):
        return None
    if a * t_accel > (v_max + EPSILON):
        return None
    a = min(a, a_max)  # Numerical error
    durations = [t_accel, t_accel]
    accels = [sign * a, -sign * a]
    p_curve = curve_from_controls(durations, accels, x0=x1)
    return p_curve


def parabolic_val(t=0.0, t0=0.0, x0=0.0, v0=0.0, a=0.0):
    return x0 + v0 * (t - t0) + 1 / 2.0 * a * (t - t0) ** 2


def zero_three_stage(x1, x2, T, v_max=np.inf, a_max=np.inf):
    sign = get_sign(x2 - x1)
    d = abs(x2 - x1)
    solutions = np.roots(
        [
            a_max,
            -a_max * T,
            d,
        ]
    )
    solutions = filter(check_time, solutions)
    solutions = [t for t in solutions if (T - 2 * t) >= 0]
    if not solutions:
        return None
    t1 = min(solutions)
    if t1 * a_max > (v_max + EPSILON):
        return None
    # t1 = min(t1, v_max / a_max)
    t3 = t1
    t2 = T - t1 - t3  # Lower velocity
    durations = [t1, t2, t3]
    accels = [sign * a_max, 0.0, -sign * a_max]
    p_curve = curve_from_controls(durations, accels, x0=x1)
    return p_curve


def opt_straight_line(
    x1, x2, v_max=np.inf, a_max=np.inf, t_min=1e-3, only_duration=False
):
    # TODO: solve for a given T which is higher than the min T
    # TODO: solve for all joints at once using a linear interpolator
    # Can always rest at the start of the trajectory if need be
    # Exploits symmetry
    assert (v_max > 0.0) and (a_max > 0.0)
    # assert (v_max < INF) or (a_max < INF) or (t_min > 0)
    # v_max = abs(x2 - x1) / abs(v_max)
    d = abs(x2 - x1)
    # assert d > 0
    # if v_max == INF:
    #     raise NotImplementedError()
    # TODO: more efficient version

    if np.isinf(a_max):
        T = d / v_max
        T += 2 * EPSILON
        T = max(t_min, T)
        if only_duration:
            return T
        p_curve = zero_one_fixed(x1, x2, T, v_max=v_max)
        check_curve(p_curve, x1, x2, v1=0.0, v2=0.0, T=T, v_max=v_max, a_max=a_max)
        return p_curve

    t_accel = math.sqrt(d / a_max)  # 1/2.*a*t**2 = d/2.
    if a_max * t_accel <= v_max:
        T = 2.0 * t_accel
        # a = a_max
        assert t_min <= T
        T = max(t_min, T)
        if only_duration:
            return T
        p_curve = zero_two_ramp(x1, x2, T, v_max, a_max)
        check_curve(p_curve, x1, x2, v1=0.0, v2=0.0, T=T, v_max=v_max, a_max=a_max)
        return p_curve

    t1 = t3 = (v_max - 0.0) / a_max
    t2 = (d - 2 * parabolic_val(t1, a=a_max)) / v_max
    T = t1 + t2 + t3

    assert t_min <= T
    T = max(t_min, T)
    if only_duration:
        return T
    p_curve = zero_three_stage(x1, x2, T, v_max=v_max, a_max=a_max)
    check_curve(p_curve, x1, x2, v1=0.0, v2=0.0, T=T, v_max=v_max, a_max=a_max)
    return p_curve


def solve_linear(difference, v_max, a_max, **kwargs):
    # TODO: careful with circular joints
    # TODO: careful if difference is zero
    unit_v_max = min(np.divide(v_max, np.absolute(difference)))
    unit_a_max = min(np.divide(a_max, np.absolute(difference)))
    return opt_straight_line(
        x1=0.0, x2=1.0, v_max=unit_v_max, a_max=unit_a_max, **kwargs
    )


def quickest_two_ramp(x1, x2, v1, v2, a_max, v_max=np.inf):
    # optimize_two_ramp(x1, x2, v1, v2, a_max)
    solutions = np.roots(
        [
            a_max,  # t**2
            2 * v1,  # t
            (v1 ** 2 - v2 ** 2) / (2 * a_max) + (x1 - x2),  # 1
        ]
    )
    solutions = [t for t in solutions if check_time(t)]
    # solutions = [t for t in solutions if t <= abs(v2 - v1) / abs(a_max)] # TODO: omitting this constraint from Hauser (abs(v2 - v1) might be 0)
    solutions = [
        t for t in solutions if t <= abs(v_max) / abs(a_max)
    ]  # Maybe this is what Hauser meant?
    solutions = [
        t for t in solutions if abs(v1 + t * a_max) <= abs(v_max) + EPSILON
    ]  # TODO: check v2
    if not solutions:
        return None
    t = min(solutions)
    T = 2 * t + (v1 - v2) / a_max
    if T < 0:
        return None
    # min_two_ramp(x1, x2, v1, v2, T, a_max, v_max=v_max)
    return T


def solve_three_stage(x1, x2, v1, v2, v_max, a):
    tp1 = (v_max - v1) / a
    tl = (v2 ** 2 + v1 ** 2 - 2 * v_max ** 2) / (2 * v_max * a) + (x2 - x1) / v_max
    tp2 = (v2 - v_max) / -a  # Difference from Hauser
    return tp1, tl, tp2


def quickest_three_stage(x1, x2, v1, v2, v_max, a_max):
    # http://motion.pratt.duke.edu/papers/icra10-smoothing.pdf
    # https://github.com/Puttichai/parabint/blob/2662d4bf0fbd831cdefca48863b00d1ae087457a/parabint/optimization.py
    # TODO: minimum-switch-time constraint
    # assert np.positive(v_max).all() and np.positive(a_max).all()
    # P+L+P-
    ts = solve_three_stage(x1, x2, v1, v2, v_max, a_max)
    if any(t < 0 for t in ts):
        return None
    T = sum(ts)
    # min_three_ramp(x1, x2, v1, v2, v_max, a_max, T)
    return T


def quickest_inf_accel(x1, x2, v_max=np.inf):
    # return solve_zero_ramp(x1, x2, v_max=np.inf)
    return abs(x2 - x1) / abs(v_max)


def quickest_stage(x1, x2, v1, v2, v_max=np.inf, a_max=np.inf, min_t=1e-3):
    # TODO: handle infinite acceleration
    assert (v_max > 0.0) and (a_max > 0.0)
    assert all(abs(v) <= (v_max + EPSILON) for v in [v1, v2])
    if (v_max == np.inf) and (a_max == np.inf):
        T = 0
        return max(min_t, T)  # TODO: throw an error
    if np.isinf(a_max):
        T = quickest_inf_accel(x1, x2, v_max=v_max)
        return max(min_t, T)

    # if (v1 == 0.) and (v2 == 0.):
    #     candidates = [opt_straight_line(x1, x2, v_max=v_max, a_max=a_max).x]

    candidates = [
        quickest_two_ramp(x1, x2, v1, v2, a_max, v_max=v_max),
        quickest_two_ramp(x1, x2, v1, v2, -a_max, v_max=-v_max),
    ]
    # if v_max != INF:
    candidates.extend(
        [
            quickest_three_stage(x1, x2, v1, v2, v_max, a_max),
            quickest_three_stage(x1, x2, v1, v2, -v_max, -a_max),
        ]
    )
    candidates = [t for t in candidates if t is not None]
    if not candidates:
        return None
    T = min(t for t in candidates)
    return max(min_t, T)


def solve_multivariate_ramp(x1, x2, v1, v2, v_max, a_max):
    d = len(x1)
    durations = [
        quickest_stage(x1[i], x2[i], v1[i], v2[i], v_max[i], a_max[i]) for i in range(d)
    ]
    if any(duration is None for duration in durations):
        return None
    return max(durations)


def smooth_cubic(
    path,
    collision_fn,
    resolutions,
    v_max=None,
    a_max=None,
    time_step=1e-2,
    parabolic=True,
    sample=False,
    intermediate=True,
    max_iterations=1000,
    max_time=np.inf,
    min_improve=0.0,
    verbose=False,
):
    start_time = time.time()
    if path is None:
        return None
    assert (v_max is not None) or (a_max is not None)
    assert path and (max_iterations < np.inf) or (max_time < np.inf)

    def curve_collision_fn(segment, t0=None, t1=None):
        # if not within_dynamical_limits(curve, max_v=v_max, max_a=a_max, start_t=t0, end_t=t1):
        #    return True
        _, samples = sample_discretize_curve(
            segment, resolutions, start_t=t0, end_t=t1, time_step=time_step
        )
        if any(map(collision_fn, default_selector(samples))):
            return True
        return False

    start_positions = waypoints_from_path(
        path
    )  # TODO: ensure following the same path (keep intermediate if need be)
    if len(start_positions) == 1:
        start_positions.append(start_positions[-1])

    start_durations = [0] + [
        solve_linear(np.subtract(p2, p1), v_max, a_max, t_min=1e-3, only_duration=True)
        for p1, p2 in get_pairs(start_positions)
    ]  # TODO: does not assume continuous acceleration
    start_times = np.cumsum(start_durations)  # TODO: dilate times
    start_velocities = [
        np.zeros(len(start_positions[0])) for _ in range(len(start_positions))
    ]
    start_curve = CubicHermiteSpline(
        start_times, start_positions, dydx=start_velocities
    )
    # TODO: directly optimize for shortest spline
    if len(start_positions) <= 2:
        return start_curve

    curve = start_curve
    for iteration in range(max_iterations):
        if elapsed_time(start_time) >= max_time:
            break
        times = curve.x
        durations = [0.0] + [t2 - t1 for t1, t2 in get_pairs(times)]
        positions = [curve(t) for t in times]
        velocities = [curve(t, nu=1) for t in times]

        t1, t2 = np.random.uniform(times[0], times[-1], 2)
        if t1 > t2:
            t1, t2 = t2, t1
        ts = [t1, t2]
        i1 = find(
            lambda i: times[i] <= t1, reversed(range(len(times)))
        )  # index before t1
        i2 = find(lambda i: times[i] >= t2, range(len(times)))  # index after t2
        assert i1 != i2

        local_positions = [curve(t) for t in ts]
        local_velocities = [curve(t, nu=1) for t in ts]
        if not all(
            np.less_equal(np.absolute(v), np.array(v_max) + EPSILON).all()
            for v in local_velocities
        ):
            continue

        x1, x2 = local_positions
        v1, v2 = local_velocities

        current_t = (t2 - t1) - min_improve  # TODO: percent improve
        # min_t = 0
        min_t = find_lower_bound(x1, x2, v1, v2, v_max=v_max, a_max=a_max)
        if parabolic:
            # Softly applies limits
            min_t = solve_multivariate_ramp(
                x1, x2, v1, v2, v_max, a_max
            )  # TODO: might not be feasible (soft constraint)
            if min_t is None:
                continue
        if min_t >= current_t:
            continue
        best_t = random.uniform(min_t, current_t) if sample else min_t

        local_durations = [t1 - times[i1], best_t, times[i2] - t2]
        # local_times = [0, best_t]
        local_times = [
            t1,
            (t1 + best_t),
        ]  # Good if the collision function is time varying

        if intermediate:
            local_curve = CubicHermiteSpline(
                local_times, local_positions, dydx=local_velocities
            )
            if curve_collision_fn(local_curve, t0=None, t1=None):  # check_spline
                continue
            # local_positions = [local_curve(x) for x in local_curve.x]
            # local_velocities = [local_curve(x, nu=1) for x in local_curve.x]
            local_durations = (
                [t1 - times[i1]]
                + [x - local_curve.x[0] for x in local_curve.x[1:]]
                + [times[i2] - t2]
            )

        new_durations = np.concatenate(
            [durations[: i1 + 1], local_durations, durations[i2 + 1 :]]
        )
        new_times = np.cumsum(new_durations)
        new_positions = positions[: i1 + 1] + local_positions + positions[i2:]
        new_velocities = velocities[: i1 + 1] + local_velocities + velocities[i2:]

        new_curve = CubicHermiteSpline(new_times, new_positions, dydx=new_velocities)
        if not intermediate and curve_collision_fn(new_curve, t0=None, t1=None):
            continue
        if verbose:
            print(
                "Iterations: {} | Current time: {:.3f} | New time: {:.3f} | Elapsed time: {:.3f}".format(
                    iteration,
                    spline_duration(curve),
                    spline_duration(new_curve),
                    elapsed_time(start_time),
                )
            )
        curve = new_curve
    if verbose:
        print(
            "Iterations: {} | Start time: {:.3f} | End time: {:.3f} | Elapsed time: {:.3f}".format(
                max_iterations,
                spline_duration(start_curve),
                spline_duration(curve),
                elapsed_time(start_time),
            )
        )
    return curve