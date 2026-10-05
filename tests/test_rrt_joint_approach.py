"""Run with unittest; no training dependencies or robot connection needed."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from rrt_joint_approach import rrt_connect
from export_ur5e_servoj import build_parser, smooth_segment
from replay_predicted_trajectory_pybullet import build_parser as replay_parser


class ApproachTests(unittest.TestCase):
    def test_defaults_match_replay(self):
        export = build_parser().parse_args(["--output", "test.script"])
        replay = replay_parser().parse_args([])
        for name in ("default_joints", "workpiece_position", "workpiece_scale", "lift_height",
                     "lift_seconds", "horizontal_seconds", "vertical_yaw_deg", "rrt_seed",
                     "rrt_clearance_m", "rrt_step_rad", "rrt_edge_resolution_rad"):
            self.assertEqual(getattr(export, name), getattr(replay, name), name)

    def test_rrt_avoids_blocked_edge_and_is_deterministic(self):
        # A central rectangle blocks the direct approach, but permits a detour.
        def valid(q):
            return not (abs(q[0]) <= 0.2 and abs(q[1]) <= 0.65)

        args = ([-0.8, 0], [0.8, 0], [-1, -1], [1, 1], valid)
        path, report = rrt_connect(*args, seed=7)
        repeated, _ = rrt_connect(*args, seed=7)
        np.testing.assert_array_equal(path, repeated)
        self.assertFalse(report["direct_edge_valid"])
        np.testing.assert_array_equal(path[0], args[0])
        np.testing.assert_array_equal(path[-1], args[1])
        # Validate much more finely than the planner, including quintic timing.
        for start, goal in zip(path[:-1], path[1:]):
            for u in np.linspace(0, 1, 1001):
                alpha = 10*u**3 - 15*u**4 + 6*u**5
                self.assertTrue(valid(start+(goal-start)*alpha))

    def test_invalid_endpoint_and_disconnected_space_refuse(self):
        with self.assertRaisesRegex(ValueError, "start"):
            rrt_connect([-2, 0], [0.8, 0], [-1, -1], [1, 1], lambda q: True)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            rrt_connect([-0.8, 0], [0.8, 0], [-1, -1], [1, 1],
                        lambda q: abs(q[0]) > 0.2, max_iterations=100)

    def test_servo_timing_keeps_rrt_edge_and_joint_limits(self):
        start, goal = np.zeros(6), np.array([0.9, -0.4, 0.2, 0, 0.1, -0.3])
        points, _, peak_v, peak_a = smooth_segment(start, goal, 0.002, 0.1, 0.5, 1.0)
        np.testing.assert_array_equal(points[0], start)
        np.testing.assert_allclose(points[-1], goal, atol=1e-12)
        # All axes must share one scalar interpolation along the checked edge.
        np.testing.assert_allclose(points, points[:, :1]/goal[0]*goal, atol=1e-12)
        self.assertLessEqual(peak_v, 0.5)
        self.assertLessEqual(peak_a, 1.0)


if __name__ == "__main__":
    unittest.main()
