"""CPU-only orchestration tests; model and geometry backends are mocked."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import plan_dp_dual_from_welds as planner


class Kinematics:
    def forward(self, q):
        return np.eye(4)


class BatchTest(unittest.TestCase):
    def test_shared_geometry_and_partial_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.write_text("{}")
            args = planner.build_parser().parse_args([
                "--workpiece-stl", str(source), "--seam-file", str(source),
                "--workpiece-position", "0", "0", "0", "--urdf-path", str(source),
                "--checkpoint-path", str(source), "--stats-path", str(source),
                "--output-dir", str(root / "out"),
            ])
            requests = [
                {"id": "a", "start": [0, 0, 0], "goal": [1, 0, 0]},
                {"id": "bad", "start": [0, 0, 0], "goal": [0, 0, 0]},
                {"id": "b", "start": [0, 0, 0], "goal": [2, 0, 0]},
            ]
            def matches(a, path):
                return [1], {"world": a.start, "normal_world": np.array([0, 0, 1])}, {"world": a.goal, "normal_world": np.array([0, 0, 1])}
            def shared(a, jobs, out, job, cache):
                cache["geometry"] = (argparse.Namespace(target_frame_from_weld=None), Kinematics(), None, job/"workpiece_sdf.npz", out/"cspace_features")
            def endpoint(**kw):
                tf = np.eye(4)
                tf[:3, 3] = kw["match"]["world"]
                return np.zeros(6), tf, {}
            def infer(command, cwd, label):
                self.assertEqual(command[command.index("--sample-count")+1], "2")
                inputs = Path(command[command.index("--input-dirs")+1])
                output = Path(command[command.index("--output-root")+1])
                processed = []
                for path in sorted(inputs.glob("transition_*.npz")):
                    dest = output/path.stem
                    dest.mkdir(parents=True)
                    np.save(dest/"pred_joint_horizon.npy", np.zeros((64, 6)))
                    processed.append({"npz_path": str(path), "output_dir": str(dest), "planning_success": True})
                planner.write_json(output/"batch_inference_manifest.json", {"processed": processed, "failed": []})
            with patch.object(planner, "load_and_match_endpoints", side_effect=matches), \
                 patch.object(planner, "build_shared_geometry", side_effect=shared) as geometry, \
                 patch.object(planner, "solve_endpoint", side_effect=endpoint), \
                 patch.object(planner, "DP_BATCH_SCRIPT", source), \
                 patch.object(planner, "_run", side_effect=infer) as runner:
                with self.assertRaisesRegex(RuntimeError, "Some batch requests failed"):
                    planner.plan_batch(args, requests)
                self.assertEqual(geometry.call_count, 1)
                self.assertEqual(runner.call_count, 1)
            report = json.loads((args.output_dir/"batch_result.json").read_text())
            self.assertEqual(report["successful_count"], 2)
            self.assertEqual(report["failed_count"], 1)
            self.assertEqual([r["id"] for r in report["results"]], ["a", "bad", "b"])
            self.assertTrue(Path(report["results"][2]["tcp_transforms"]).is_file())
            with self.assertRaisesRegex(ValueError, "empty/new"):
                planner.plan_batch(args, requests)

    def test_manifest_matching_is_by_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = [{"id": str(i), "start": [0]*3, "goal": [1]*3,
                      "prepared": {"outputs": {"transition_npz": root/f"{i}.npz"}, "kinematics": Kinematics()}} for i in range(2)]
            manifest = {"processed": [], "failed": [{"npz_path": str(root/"1.npz"), "error": "IK failed"}]}
            records = planner.collect_batch_results(batch, manifest)["results"]
            self.assertEqual(records[0]["error"], "Missing inference result")
            self.assertEqual(records[1]["error"], "IK failed")

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"pairs.json"
            path.write_text(json.dumps([{"id": "same", "start": [0]*3, "goal": [1]*3}]*2))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                planner.load_requests(argparse.Namespace(requests_file=path, start=None, goal=None))


if __name__ == "__main__":
    unittest.main()
