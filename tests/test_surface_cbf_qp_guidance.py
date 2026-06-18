import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


from diffusion_policy_3d.common.surface_cbf_qp_guidance import (
    build_guidance_target_schedule,
    compute_path_length,
    compute_smoothness,
    interpolate_swept_segments,
    select_guidance_candidate_indices,
)


class SurfaceCBFQPGuidanceHelperTests(unittest.TestCase):
    def test_build_guidance_target_schedule_matches_default_five_step_profile(self):
        schedule = build_guidance_target_schedule(5)
        np.testing.assert_allclose(
            schedule,
            np.asarray([-0.02, -0.01, -0.005, 0.0, 0.0], dtype=np.float32),
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