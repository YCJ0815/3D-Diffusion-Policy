"""Deterministic joint-space RRT-Connect for the collision-checked approach.

Edges are straight joint-space segments; use scalar quintic timing on each
edge, never a spline across vertices (which could leave the checked path).
"""
import numpy as np


def add_rrt_arguments(parser):
    parser.add_argument("--rrt-seed", type=int, default=0)
    parser.add_argument("--rrt-max-iterations", type=int, default=5000)
    parser.add_argument("--rrt-step-rad", type=float, default=0.25)
    parser.add_argument("--rrt-edge-resolution-rad", type=float, default=0.01)
    parser.add_argument("--rrt-clearance-m", type=float, default=0.002,
                        help="Minimum sampled robot-workpiece clearance during approach.")


def rrt_connect(start, goal, lower, upper, valid, *, seed=0, max_iterations=5000,
                step=0.25, resolution=0.01, shortcut_attempts=100):
    """Plan without angle wrapping; raise on invalid endpoints or failure."""
    start, goal, lower, upper = [np.asarray(x, dtype=float) for x in (start, goal, lower, upper)]
    if (not np.isfinite([step, resolution]).all() or step <= 0 or resolution <= 0
            or max_iterations < 1):
        raise ValueError("RRT step, resolution and iteration count must be positive")
    if (start.shape != goal.shape or lower.shape != start.shape or upper.shape != start.shape
            or not np.isfinite(np.stack([start, goal, lower, upper])).all()
            or np.any(lower >= upper)):
        raise ValueError("Invalid RRT joint bounds or endpoint dimensions")
    checks = 0

    def state_valid(q):
        nonlocal checks
        checks += 1
        return bool(np.all(q >= lower) and np.all(q <= upper) and valid(q))

    def edge_valid(a, b):
        count = max(1, int(np.ceil(np.max(np.abs(b-a)) / resolution)))
        return all(state_valid(a + (b-a)*u) for u in np.linspace(0, 1, count+1))

    for name, q in (("start", start), ("goal", goal)):
        if not state_valid(q):
            raise ValueError(f"RRT {name} violates joint limits, collision or clearance constraints")
    rng = np.random.default_rng(seed)
    iterations = 0
    direct = edge_valid(start, goal)
    path = [start, goal] if direct else None
    trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]

    def extend(tree, target):
        nodes, parents = tree
        nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes)-target, axis=1)))
        delta = target-nodes[nearest]
        distance = float(np.linalg.norm(delta))
        if distance < 1e-12:
            return nearest, True
        candidate = nodes[nearest] + delta*min(1.0, step/distance)
        if not edge_valid(nodes[nearest], candidate):
            return None, False
        nodes.append(candidate)
        parents.append(nearest)
        return len(nodes)-1, distance <= step

    def trace(tree, index):
        nodes, parents = tree
        result = []
        while index >= 0:
            result.append(nodes[index])
            index = parents[index]
        return result[::-1]

    if path is None:
        for iterations in range(1, max_iterations+1):
            active = (iterations-1) % 2
            other = 1-active
            target = trees[other][0][0] if rng.random() < 0.15 else rng.uniform(lower, upper)
            a, _ = extend(trees[active], target)
            if a is None:
                continue
            while True:
                b, reached = extend(trees[other], trees[active][0][a])
                if b is None:
                    break
                if reached:
                    first, second = trace(trees[active], a), trace(trees[other], b)
                    path = first + second[-2::-1]
                    if active == 1:
                        path.reverse()
                    break
            if path is not None:
                break
    if path is None:
        raise RuntimeError(f"RRT-Connect failed after {max_iterations} iterations; export refused")
    for _ in range(shortcut_attempts):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j > i+1 and edge_valid(path[i], path[j]):
            path = path[:i+1] + path[j:]
    # Revalidate all shortcut edges and preserve the exact absolute endpoints.
    path = np.asarray(path)
    path[0], path[-1] = start, goal
    if not all(edge_valid(a, b) for a, b in zip(path[:-1], path[1:])):
        raise RuntimeError("RRT final path validation failed")
    return path, {"planner": "joint_space_rrt_connect", "seed": seed,
                  "iterations": iterations, "direct_edge_valid": direct,
                  "state_checks": checks, "path_vertices": len(path),
                  "edge_resolution_rad": resolution, "step_rad": step}


def plan_pybullet_approach(pb, robot, joints, workpiece, start, goal, client, args):
    if not np.isfinite(args.rrt_clearance_m) or args.rrt_clearance_m < 0:
        raise ValueError("RRT clearance must be finite and nonnegative")
    info = [pb.getJointInfo(robot, j, physicsClientId=client) for j in joints]
    lower, upper = np.array([x[8] for x in info]), np.array([x[9] for x in info])

    def valid(q):
        for joint, angle in zip(joints, q):
            pb.resetJointState(robot, joint, float(angle), physicsClientId=client)
        pb.performCollisionDetection(physicsClientId=client)
        return not (pb.getContactPoints(robot, robot, physicsClientId=client)
                    or pb.getClosestPoints(robot, workpiece, distance=args.rrt_clearance_m,
                                           physicsClientId=client))

    path, report = rrt_connect(start, goal, lower, upper, valid,
                              seed=args.rrt_seed, max_iterations=args.rrt_max_iterations,
                              step=args.rrt_step_rad, resolution=args.rrt_edge_resolution_rad)
    report["robot_workpiece_clearance_m"] = args.rrt_clearance_m
    report["self_collision_clearance_m"] = 0.0
    report["joint_path_rad"] = path.tolist()
    return path, report


def edge_durations(path, minimum_duration):
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = float(lengths.sum())
    return minimum_duration * (lengths/total if total > 0 else np.ones(len(lengths))/len(lengths))
