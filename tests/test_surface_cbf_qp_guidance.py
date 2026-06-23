import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


from diffusion_policy_3d.common.surface_cbf_qp_guidance import (
    build_risk_segments,
    build_segment_window_cbf_constraints,
    build_guidance_target_schedule,
    compute_path_length,
    compute_smoothness,
    interpolate_swept_segments,
    rank_screened_candidates,
    select_segment_window_timesteps,
    select_guidance_candidate_indices,
    summarize_sdf_risk,
)


class SurfaceCBFQPGuidanceHelperTests(unittest.TestCase):
    def test_build_risk_segments_groups_contiguous_trigger_violations(self):
        sdf_result = {
            "all_sdf_values": np.asarray(
                [
                    [0.09, 0.08],
                    [0.02, 0.03],
                    [0.01, 0.04],
                    [0.08, 0.07],
                    [-0.01, 0.01],
                ],
                dtype=np.float32,
            )
        }
        segments = build_risk_segments(sdf_result=sdf_result, d_trigger=0.05)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["timesteps"], [4])
        self.assertEqual(segments[1]["timesteps"], [1, 2])

    def test_build_guidance_target_schedule_matches_default_five_step_profile(self):
        schedule = build_guidance_target_schedule(5)
        np.testing.assert_allclose(
            schedule,
            np.asarray([-0.02, -0.015, -0.01, -0.005, 0.0], dtype=np.float32),
        )

    def test_select_guidance_candidate_indices_skips_safe_and_deep_candidates(self):
        h_min = np.asarray([0.10, 0.04, -0.01, -0.06, 0.02], dtype=np.float32)
        selected = select_guidance_candidate_indices(
            h_min_values=h_min,
            qp_per_step=2,
            d_trigger=0.05,
            eps_deep=0.03,
        )
        np.testing.assert_array_equal(selected, np.asarray([2, 4], dtype=np.int64))

    def test_rank_screened_candidates_prefers_safer_then_shorter_then_smoother(self):
        ranked = rank_screened_candidates(
            [
                {
                    "candidate_index": 0,
                    "coarse_min_margin": 0.02,
                    "coarse_dangerous_timestep_count": 2,
                    "coarse_total_risk": 0.3,
                    "coarse_path_length": 4.0,
                    "coarse_smoothness": 0.5,
                },
                {
                    "candidate_index": 1,
                    "coarse_min_margin": 0.05,
                    "coarse_dangerous_timestep_count": 1,
                    "coarse_total_risk": 0.1,
                    "coarse_path_length": 5.0,
                    "coarse_smoothness": 0.7,
                },
                {
                    "candidate_index": 2,
                    "coarse_min_margin": 0.05,
                    "coarse_dangerous_timestep_count": 1,
                    "coarse_total_risk": 0.1,
                    "coarse_path_length": 4.5,
                    "coarse_smoothness": 0.9,
                },
            ]
        )
        self.assertEqual(ranked, [2, 1, 0])

    def test_summarize_sdf_risk_reports_margin_risk_and_counts(self):
        sdf_result = {
            "all_sdf_values": np.asarray(
                [
                    [0.08, 0.09],
                    [0.02, 0.05],
                    [0.01, -0.01],
                ],
                dtype=np.float32,
            )
        }
        summary = summarize_sdf_risk(
            sdf_result=sdf_result,
            d_safe=0.03,
            d_trigger=0.06,
        )
        self.assertAlmostEqual(summary["min_margin"], -0.04, places=6)
        self.assertAlmostEqual(summary["min_clearance"], -0.01, places=6)
        self.assertEqual(summary["dangerous_timestep_count"], 3)
        self.assertGreater(summary["total_risk"], 0.0)
        self.assertEqual(summary["finite_sdf_count"], 6)
        self.assertEqual(summary["finite_timestep_count"], 3)

    def test_select_segment_window_timesteps_includes_peak_and_worst_window_steps(self):
        sdf_result = {
            "all_sdf_values": np.asarray(
                [
                    [0.08, 0.09],
                    [0.04, 0.03],
                    [0.01, 0.02],
                    [0.05, 0.06],
                    [0.07, 0.08],
                ],
                dtype=np.float32,
            )
        }
        segment = {
            "peak_timestep": 2,
            "start_timestep": 1,
            "end_timestep": 3,
        }
        selected = select_segment_window_timesteps(
            sdf_result=sdf_result,
            segment=segment,
            points_per_segment=2,
            window_radius=1,
        )
        self.assertIn(2, selected)
        self.assertEqual(selected, [1, 2])

    def test_build_segment_window_constraints_respects_active_constraint_cap(self):
        class DummyEnv:
            def __init__(self):
                self.surface_samples = [
                    {"flat_index": 0, "link_index": 0, "point_index": 0, "local_point": np.zeros(3, dtype=np.float32)},
                    {"flat_index": 1, "link_index": 1, "point_index": 0, "local_point": np.zeros(3, dtype=np.float32)},
                ]

            def normalized_to_actual(self, q_norm):
                return np.asarray(q_norm, dtype=np.float32)

            def surface_point_world(self, *, link_index, local_point):
                _ = link_index, local_point
                return np.zeros(3, dtype=np.float32)

            def surface_point_jacobian(self, *, link_index, local_point, q_actual):
                _ = link_index, local_point, q_actual
                return np.vstack([np.ones((3, 2), dtype=np.float32), np.zeros((3, 2), dtype=np.float32)])

            def load_sdf_grid(self):
                class Grid:
                    def query(self, points):
                        return np.zeros((len(points),), dtype=np.float32)

                return Grid()

        sdf_result = {
            "all_sdf_values": np.asarray(
                [
                    [0.08, 0.09],
                    [0.02, 0.03],
                    [0.01, 0.015],
                    [0.06, 0.07],
                ],
                dtype=np.float32,
            )
        }
        segments = build_risk_segments(sdf_result=sdf_result, d_trigger=0.05)
        constraints, selected_segments, selected_timesteps = build_segment_window_cbf_constraints(
            segments=segments,
            sdf_result=sdf_result,
            check_basis=np.ones((4, 3), dtype=np.float32),
            q_check_norm=np.ones((4, 2), dtype=np.float32),
            environment=DummyEnv(),
            d_trigger=0.05,
            points_per_segment=2,
            min_constraints_per_segment=2,
            window_radius=1,
            max_active=3,
        )
        self.assertEqual(len(constraints), 2)
        self.assertEqual(len(selected_segments), 1)
        self.assertTrue(selected_timesteps)

    def test_interpolate_swept_segments_inserts_expected_number_of_points(self):
        trajectory = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=np.float32,
        )
        dense = interpolate_swept_segments(trajectory, num_intermediate=2)
        self.assertEqual(dense.shape, (7, 2))
        np.testing.assert_allclose(dense[0], trajectory[0])
        np.testing.assert_allclose(dense[-1], trajectory[-1])
        np.testing.assert_allclose(dense[1], np.asarray([1.0 / 3.0, 1.0 / 3.0], dtype=np.float32))
        np.testing.assert_allclose(dense[2], np.asarray([2.0 / 3.0, 2.0 / 3.0], dtype=np.float32))

    def test_compute_path_metrics_are_zero_for_constant_trajectory(self):
        trajectory = np.zeros((4, 6), dtype=np.float32)
        self.assertEqual(compute_path_length(trajectory), 0.0)
        self.assertEqual(compute_smoothness(trajectory), 0.0)


if __name__ == "__main__":
    unittest.main()
