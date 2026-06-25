import importlib.util
import pathlib
import random
import sys
import tempfile
import types
import unittest


def _install_import_stubs() -> None:
    class _FakeRNG:
        def __init__(self, seed: int):
            self._rng = random.Random(seed)

        def choice(self, population_size: int, size: int, replace: bool = False):
            if replace:
                return [self._rng.randrange(population_size) for _ in range(size)]
            return self._rng.sample(range(population_size), size)

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.random = types.SimpleNamespace(default_rng=lambda seed: _FakeRNG(seed))
    numpy_stub.float32 = float
    numpy_stub.ndarray = object
    numpy_stub.asarray = lambda value, dtype=None: value
    numpy_stub.sort = sorted
    numpy_stub.load = lambda *args, **kwargs: None
    numpy_stub.save = lambda *args, **kwargs: None
    sys.modules.setdefault("numpy", numpy_stub)

    torch_stub = types.ModuleType("torch")
    torch_stub.device = lambda name: name
    torch_stub.no_grad = lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: False)
    torch_stub.Generator = object
    sys.modules.setdefault("torch", torch_stub)

    train_stub = types.ModuleType("train")
    train_stub.TrainDP3Workspace = object
    sys.modules.setdefault("train", train_stub)

    bspline_stub = types.ModuleType("diffusion_policy_3d.common.bspline")
    bspline_stub._resolve_free_control_point_slice = lambda num_control_points: slice(1, num_control_points - 1)
    bspline_stub.fit_quintic_bspline_to_npz_trajectory = lambda **kwargs: None
    bspline_stub.load_delta_w_stats = lambda path: (None, None)
    bspline_stub.reconstruct_trajectory_from_normalized_free_residual = lambda **kwargs: None
    bspline_stub.unnormalize_joint_trajectory_with_urdf_limits = lambda **kwargs: None
    sys.modules.setdefault("diffusion_policy_3d.common.bspline", bspline_stub)

    input_data_stub = types.ModuleType("diffusion_policy_3d.common.input_data")
    input_data_stub.load_bspline_planning_input_data = lambda **kwargs: None
    sys.modules.setdefault("diffusion_policy_3d.common.input_data", input_data_stub)

    class _FakeOmegaConf:
        @staticmethod
        def load(path):
            _ = path
            return {}

        @staticmethod
        def to_container(value, resolve=True):
            _ = resolve
            return value

    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.OmegaConf = _FakeOmegaConf
    sys.modules.setdefault("omegaconf", omegaconf_stub)

    single_infer_stub = types.ModuleType("infer_bspline_trajectory")
    single_infer_stub.build_obs_dict = lambda **kwargs: ({}, {})
    single_infer_stub.ensure_dir = lambda path: path
    single_infer_stub.save_joint_plot = lambda **kwargs: None
    sys.modules.setdefault("infer_bspline_trajectory", single_infer_stub)


def _load_module():
    _install_import_stubs()
    module_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "infer_bspline_trajectories_batch.py"
    spec = importlib.util.spec_from_file_location("infer_bspline_trajectories_batch", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InferBsplineTrajectoriesBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_parser_supports_random_regular_sampling_and_candidate_compare_flags(self):
        parser = self.module.build_parser()
        args = parser.parse_args(
            [
                "--input-dirs",
                "/tmp/results",
                "--checkpoint-path",
                "/tmp/model.ckpt",
                "--stats-path",
                "/tmp/stats.npz",
                "--output-root",
                "/tmp/out",
                "--sample-source",
                "regular",
                "--sample-count",
                "10",
                "--sample-seed",
                "42",
                "--sampling-mode",
                "compare",
                "--num-candidates",
                "32",
                "--candidate-seed",
                "7",
                "--cspace-feature-dir",
                "/tmp/cspace",
                "--enable-surface-cbf-qp-guidance",
                "--planner-mode",
                "qp_guided_diffusion",
                "--qp-candidates",
                "4",
                "--qp-inner-scp-rounds",
                "2",
                "--coarse-check-steps",
                "32",
                "--guidance-pen-link-points",
                "80",
                "--guidance-wrist3-points",
                "16",
                "--final-post-qp-candidates",
                "1",
                "--final-backup-candidates",
                "1",
                "--final-post-qp-rounds",
                "2",
                "--guidance-trigger-distance",
                "0.06",
                "--guidance-safe-distance",
                "0.05",
                "--trust-region-start",
                "0.015",
                "--trust-region-end",
                "0.05",
                "--blend-weights",
                "0.25",
                "0.5",
                "0.75",
                "--repair-score-weights",
                "1.0",
                "10.0",
                "1.0",
                "--guidance-steps",
                "3",
                "--guidance-max-risk-segments",
                "3",
                "--guidance-window-radius",
                "2",
                "--guidance-points-per-segment",
                "2",
                "--guidance-min-constraints-per-segment",
                "4",
                "--guidance-active-constraints",
                "24",
                "--guidance-scp-iterations",
                "2",
                "--guidance-delta-max-total",
                "0.05",
                "--guidance-delta-max-pass1",
                "0.025",
                "--guidance-delta-max-pass2",
                "0.025",
                "--guidance-d-trigger-pass2-offset",
                "0.005",
                "--guidance-margin-buffer",
                "0.005",
                "--enable-local-waypoint-qp-after-certificate",
                "--local-waypoint-qp-window-radius",
                "2",
                "--local-waypoint-qp-max-collision-segments",
                "2",
                "--local-waypoint-qp-min-clearance-trigger",
                "-0.01",
                "--local-waypoint-qp-target-buffer",
                "0.005",
                "--local-waypoint-qp-lambda-s",
                "0.25",
                "--local-waypoint-qp-delta-max",
                "0.02",
                "--local-waypoint-qp-max-velocity-step",
                "0.2",
                "--local-waypoint-qp-max-acceleration-step",
                "0.4",
                "--local-waypoint-qp-maxiter",
                "100",
            ]
        )

        self.assertEqual(args.sample_source, "regular")
        self.assertEqual(args.sample_count, 10)
        self.assertEqual(args.sample_seed, 42)
        self.assertEqual(args.sampling_mode, "compare")
        self.assertEqual(args.num_candidates, 32)
        self.assertEqual(args.candidate_seed, 7)
        self.assertEqual(args.cspace_feature_dir, "/tmp/cspace")
        self.assertTrue(args.enable_surface_cbf_qp_guidance)
        self.assertEqual(args.planner_mode, "qp_guided_diffusion")
        self.assertEqual(args.qp_candidates, 4)
        self.assertFalse(hasattr(args, "guidance_timesteps"))
        self.assertEqual(args.qp_inner_scp_rounds, 2)
        self.assertEqual(args.coarse_check_steps, 32)
        self.assertEqual(args.guidance_pen_link_points, 80)
        self.assertEqual(args.guidance_wrist3_points, 16)
        self.assertEqual(args.final_post_qp_candidates, 1)
        self.assertEqual(args.final_backup_candidates, 1)
        self.assertEqual(args.final_post_qp_rounds, 2)
        self.assertEqual(args.guidance_trigger_distance, 0.06)
        self.assertEqual(args.guidance_safe_distance, 0.05)
        self.assertEqual(args.trust_region_start, 0.015)
        self.assertEqual(args.trust_region_end, 0.05)
        self.assertEqual(args.blend_weights, [0.25, 0.5, 0.75])
        self.assertEqual(args.repair_score_weights, [1.0, 10.0, 1.0])
        self.assertEqual(args.guidance_steps, 3)
        self.assertEqual(args.guidance_max_risk_segments, 3)
        self.assertEqual(args.guidance_window_radius, 2)
        self.assertEqual(args.guidance_points_per_segment, 2)
        self.assertEqual(args.guidance_min_constraints_per_segment, 4)
        self.assertEqual(args.guidance_active_constraints, 24)
        self.assertEqual(args.guidance_scp_iterations, 2)
        self.assertEqual(args.guidance_delta_max_total, 0.05)
        self.assertEqual(args.guidance_delta_max_pass1, 0.025)
        self.assertEqual(args.guidance_delta_max_pass2, 0.025)
        self.assertEqual(args.guidance_d_trigger_pass2_offset, 0.005)
        self.assertEqual(args.guidance_margin_buffer, 0.005)
        self.assertTrue(args.enable_local_waypoint_qp_after_certificate)
        self.assertEqual(args.local_waypoint_qp_window_radius, 2)
        self.assertEqual(args.local_waypoint_qp_max_collision_segments, 2)
        self.assertEqual(args.local_waypoint_qp_min_clearance_trigger, -0.01)
        self.assertEqual(args.local_waypoint_qp_target_buffer, 0.005)
        self.assertEqual(args.local_waypoint_qp_lambda_s, 0.25)
        self.assertEqual(args.local_waypoint_qp_delta_max, 0.02)
        self.assertEqual(args.local_waypoint_qp_max_velocity_step, 0.2)
        self.assertEqual(args.local_waypoint_qp_max_acceleration_step, 0.4)
        self.assertEqual(args.local_waypoint_qp_maxiter, 100)

    def test_parser_accepts_combined_planner_mode(self):
        parser = self.module.build_parser()
        args = parser.parse_args(
            [
                "--input-dirs",
                "/tmp/results",
                "--checkpoint-path",
                "/tmp/model.ckpt",
                "--stats-path",
                "/tmp/stats.npz",
                "--output-root",
                "/tmp/out",
                "--planner-mode",
                "qp_guided_diffusion_post_qp",
            ]
        )
        self.assertEqual(args.planner_mode, "qp_guided_diffusion_post_qp")

    def test_policy_requires_cspace_feature_detects_cspace_checkpoint(self):
        module = self.module

        plain_policy = types.SimpleNamespace()
        cspace_policy = types.SimpleNamespace(cspace_feature_key="cspace_feature")

        self.assertFalse(module.policy_requires_cspace_feature(plain_policy))
        self.assertTrue(module.policy_requires_cspace_feature(cspace_policy))

    def test_prepare_obs_inputs_injects_cspace_feature_for_cspace_policy(self):
        module = self.module
        captured = {}

        def fake_build_obs_dict(**kwargs):
            captured["build_obs_dict_kwargs"] = kwargs
            return ({"point_cloud": "pc"}, {"point_cloud": "pc"})

        def fake_inject(*, obs_dict, raw_obs, cspace_feature, n_obs_steps, device):
            captured["inject"] = {
                "cspace_feature": cspace_feature,
                "n_obs_steps": n_obs_steps,
                "device": device,
            }
            obs_dict["cspace_feature"] = cspace_feature
            raw_obs["cspace_feature"] = cspace_feature

        class FakeProvider:
            def get_feature(self, workpiece_id: int):
                captured["workpiece_id"] = workpiece_id
                return [[1.0, 2.0], [3.0, 4.0]]

        original_build_obs_dict = module.build_obs_dict
        original_inject = getattr(module, "inject_cspace_feature", None)
        try:
            module.build_obs_dict = fake_build_obs_dict
            module.inject_cspace_feature = fake_inject
            obs_dict, raw_obs, resolved_workpiece_id = module.prepare_obs_inputs(
                npz_path=pathlib.Path("/tmp/results/job_008/transition_0001_0002.npz"),
                stl_path=pathlib.Path("/tmp/jobs/job_008/workpiece.stl"),
                input_dirs=[pathlib.Path("/tmp/results")],
                policy=types.SimpleNamespace(cspace_feature_key="cspace_feature"),
                workspace=types.SimpleNamespace(cfg=types.SimpleNamespace(n_obs_steps=1)),
                device="cuda:0",
                args=types.SimpleNamespace(
                    norm_m=0.1,
                    radius_m=0.1,
                    height_m=0.1,
                    num_output_points=512,
                    num_mesh_sample_points=100000,
                    stl_x_offset_mm=500.0,
                    urdf_path=None,
                    use_poisson_disk=False,
                    simple_workpiece_id_offset=1000,
                ),
                cspace_feature_provider=FakeProvider(),
            )
        finally:
            module.build_obs_dict = original_build_obs_dict
            if original_inject is not None:
                module.inject_cspace_feature = original_inject

        self.assertEqual(resolved_workpiece_id, 8)
        self.assertEqual(captured["workpiece_id"], 8)
        self.assertEqual(captured["inject"]["cspace_feature"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(obs_dict["cspace_feature"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(raw_obs["cspace_feature"], [[1.0, 2.0], [3.0, 4.0]])

    def test_prepare_obs_inputs_rejects_cspace_policy_without_feature_provider(self):
        module = self.module

        original_build_obs_dict = module.build_obs_dict
        try:
            module.build_obs_dict = lambda **kwargs: ({"point_cloud": "pc"}, {"point_cloud": "pc"})
            with self.assertRaisesRegex(ValueError, "cspace-feature-dir"):
                module.prepare_obs_inputs(
                    npz_path=pathlib.Path("/tmp/results/job_008/transition_0001_0002.npz"),
                    stl_path=pathlib.Path("/tmp/jobs/job_008/workpiece.stl"),
                    input_dirs=[pathlib.Path("/tmp/results")],
                    policy=types.SimpleNamespace(cspace_feature_key="cspace_feature"),
                    workspace=types.SimpleNamespace(cfg=types.SimpleNamespace(n_obs_steps=1)),
                    device="cuda:0",
                    args=types.SimpleNamespace(
                        norm_m=0.1,
                        radius_m=0.1,
                        height_m=0.1,
                        num_output_points=512,
                        num_mesh_sample_points=100000,
                        stl_x_offset_mm=500.0,
                        urdf_path=None,
                        use_poisson_disk=False,
                        simple_workpiece_id_offset=1000,
                    ),
                    cspace_feature_provider=None,
                )
        finally:
            module.build_obs_dict = original_build_obs_dict

    def test_filter_npz_files_by_source_keeps_only_regular_jobs(self):
        module = self.module
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            regular_dir = root / "results"
            simple_dir = root / "simple_results"
            regular_dir.mkdir()
            simple_dir.mkdir()
            regular_npz = regular_dir / "job_001" / "transition_a.npz"
            simple_npz = simple_dir / "job_1001" / "transition_b.npz"
            regular_npz.parent.mkdir()
            simple_npz.parent.mkdir()
            regular_npz.touch()
            simple_npz.touch()

            filtered = module.filter_npz_files_by_source(
                [regular_npz.resolve(), simple_npz.resolve()],
                [regular_dir.resolve(), simple_dir.resolve()],
                "regular",
            )

        self.assertEqual(filtered, [regular_npz.resolve()])

    def test_sample_npz_files_returns_deterministic_non_adjacent_subset(self):
        module = self.module
        paths = [pathlib.Path(f"/tmp/job_{idx:03d}/transition_{idx:03d}.npz") for idx in range(20)]

        sampled_once = module.sample_npz_files(paths, sample_count=10, sample_seed=42)
        sampled_twice = module.sample_npz_files(paths, sample_count=10, sample_seed=42)

        self.assertEqual(sampled_once, sampled_twice)
        self.assertEqual(len(sampled_once), 10)
        self.assertEqual(len(set(sampled_once)), 10)
        self.assertNotEqual(sampled_once, paths[:10])

    def test_resolve_sampling_mode_promotes_candidate_flag_and_compare(self):
        module = self.module

        baseline_args = types.SimpleNamespace(sampling_mode="baseline", enable_candidate_pool=False)
        candidate_args = types.SimpleNamespace(sampling_mode="baseline", enable_candidate_pool=True)
        compare_args = types.SimpleNamespace(sampling_mode="compare", enable_candidate_pool=False)

        self.assertEqual(module.resolve_sampling_mode(baseline_args), "baseline")
        self.assertEqual(module.resolve_sampling_mode(candidate_args), "candidate")
        self.assertEqual(module.resolve_sampling_mode(compare_args), "compare")


if __name__ == "__main__":
    unittest.main()
