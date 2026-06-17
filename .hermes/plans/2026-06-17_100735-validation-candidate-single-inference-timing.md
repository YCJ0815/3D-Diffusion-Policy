# Validation Candidate Scoring + Single-Inference Timing Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Update `scripts/validate_all_trajectories.py` so standalone checkpoint validation matches training-time PyBullet validation candidate generation/selection behavior (32 candidates + scoring/selection), and add a new mode that runs inference for exactly one trajectory while reporting the elapsed time from inference start to final selected output.

**Architecture:** Reuse the same candidate-generation and scoring pathway already implemented in `PyBulletValidationRunner.run()` instead of keeping a separate single-candidate inference path in the script. Add a small helper layer in the standalone script to (a) construct candidate batches, (b) score/select with the same PyBullet validator logic, and (c) optionally short-circuit after one episode while measuring end-to-end latency for the inference-and-selection stage.

**Tech Stack:** Python 3.11, PyTorch, NumPy, argparse, existing `TrainDP3CSpaceWorkspace`, `PyBulletValidationRunner`, `PyBulletCollisionValidator`, B-spline reconstruction utilities.

---

## Current context / assumptions

- The standalone validator currently loads a checkpoint from `scripts/validate_all_trajectories.py` and performs exactly one `policy.predict_action(obs_dict)` call per episode, then reconstructs and evaluates that one trajectory.
  - Current single-candidate path: `scripts/validate_all_trajectories.py:435-478`
- Training-time PyBullet validation already implements the desired multi-candidate behavior:
  - Candidate loop, per-candidate seeds, scheduler kwargs, optional action noise: `3D-Diffusion-Policy/diffusion_policy_3d/common/pybullet_validation.py:1550-1586`
  - Candidate reconstruction and scoring: `3D-Diffusion-Policy/diffusion_policy_3d/common/pybullet_validation.py:1596-1679`
  - Candidate score-key ranking/selection continues immediately after the snippet above in the same function.
- Training config already specifies the intended defaults:
  - `num_candidates: 32`
  - `candidate_selection: weighted_sdf`
  - `candidate_scheduler_eta: 1.0`
  - `candidate_action_noise_std: 0.02`
  - `candidate_action_noise_clip: 0.06`
  - `inference_num_steps: ${policy.num_inference_steps}`
  - Source: `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_cspace.yaml:160-210`
- The script already uses the C-space workspace loader, so checkpoint compatibility with `_raw_model` is handled.
- There is no obvious dedicated test suite for this standalone script in the repo; verification will primarily be `py_compile` plus smoke-test CLI runs in the training environment.

## Desired behavior

1. **Default / all-trajectory validation mode**
   - For each validation episode, generate **32 candidates** by default.
   - Use the same seed logic / scheduler kwargs / optional candidate action noise as training-time validation.
   - Reconstruct all candidates, score them, select the best candidate using the same rule as training-time validation, then evaluate that selected trajectory.

2. **New single-trajectory timing mode**
   - Add a CLI mode that processes exactly one selected validation episode.
   - Still use candidate generation + scoring + selection unless explicitly disabled in future.
   - Measure and print the elapsed wall-clock time from **immediately before candidate inference begins** to **the moment the final selected trajectory/result is ready**.
   - Include the measured time in the JSON output for the single-trajectory run.

---

## Proposed approach

- Keep the existing dataset loading, checkpoint loading, and PyBullet validator setup in the script.
- Extract the candidate inference-and-selection workflow into one reusable helper function inside `scripts/validate_all_trajectories.py`.
- Mirror the training-time runner logic as closely as possible, rather than inventing a new scoring implementation.
- Add CLI flags for:
  - `--single-episode-index` or equivalent explicit one-episode selector
  - `--measure-inference-time` (optional if timing should only appear in single mode)
  - optional `--num-candidates` override so the script can default to checkpoint/training config but still be adjusted from CLI
- Preserve current batch-all validation behavior when single-episode mode is not requested.

---

## Files likely to change

- Modify: `scripts/validate_all_trajectories.py`
- Read/reference only (no intended edits unless unavoidable):
  - `3D-Diffusion-Policy/diffusion_policy_3d/common/pybullet_validation.py`
  - `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_cspace.yaml`
  - `3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py`

---

## Step-by-step plan

### Task 1: Document the exact training-time candidate-selection contract to mirror

**Objective:** Freeze the exact behavior that standalone validation must replicate so implementation does not drift from training validation.

**Files:**
- Read: `3D-Diffusion-Policy/diffusion_policy_3d/common/pybullet_validation.py:1550-1680`
- Read: later lines in the same function covering candidate ranking/selection and metric packaging
- Read: `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_cspace.yaml:181-199`

**Step 1: Record the behavior to mirror**

Capture these details in implementation notes / comments while coding:

```python
# Behavior to preserve from training-time validation:
# - num_candidates from cfg (default 32)
# - candidate_seed = diffusion_sampling_seed + candidate_idx * 1_000_003 + batch_start
# - scheduler_step_kwargs includes eta when configured
# - candidate_idx > 0 may receive action-space Gaussian noise
# - each candidate is reconstructed and scored before final selection
```

**Step 2: Verify selection inputs**

Confirm the script can access or already has all required objects:
- `policy`
- `obs_dict`
- `raw_obs`
- `runner.validator`
- `workpiece_id`
- `pyb_cfg`

**Step 3: Commit**

```bash
git add scripts/validate_all_trajectories.py
git commit -m "chore: prepare standalone validator candidate-selection refactor"
```

> Commit is for the future implementer; do not do it during plan-only mode.

---

### Task 2: Add CLI flags for candidate-count override and single-trajectory mode

**Objective:** Extend the CLI so the script can either validate all episodes or run a single-episode timed path.

**Files:**
- Modify: `scripts/validate_all_trajectories.py` in `build_parser()`

**Step 1: Add failing interface expectations**

Expected new flags:

```text
--num-candidates <int>
--single-episode-index <int>
--measure-inference-time
```

Recommended argparse additions:

```python
parser.add_argument(
    "--num-candidates",
    type=int,
    default=None,
    help=(
        "Override the number of candidate trajectories generated per episode. "
        "Default: use checkpoint cfg.training.pybullet_eval.num_candidates (typically 32)."
    ),
)
parser.add_argument(
    "--single-episode-index",
    type=int,
    default=None,
    help=(
        "Validate exactly one episode index from the validation split and skip the full-dataset loop."
    ),
)
parser.add_argument(
    "--measure-inference-time",
    action="store_true",
    help=(
        "Measure elapsed time from candidate inference start to final selected output. "
        "Primarily useful with --single-episode-index."
    ),
)
```

**Step 2: Run parser smoke test**

Run:

```bash
python scripts/validate_all_trajectories.py --help
```

Expected:
- New flags appear in help output.

**Step 3: Commit**

```bash
git add scripts/validate_all_trajectories.py
git commit -m "feat: add standalone validation mode flags"
```

---

### Task 3: Add a helper that generates, reconstructs, scores, and selects candidates

**Objective:** Centralize the multi-candidate logic into a reusable helper instead of duplicating inline code in the main loop.

**Files:**
- Modify: `scripts/validate_all_trajectories.py`

**Step 1: Write the helper signature**

Add a helper near the existing helpers section:

```python
def _predict_select_candidate(
    *,
    policy,
    validator,
    pyb_cfg: PyBulletValidationConfig,
    obs_dict: dict[str, torch.Tensor],
    raw_obs: dict[str, np.ndarray],
    workpiece_id: int,
    device: torch.device,
    batch_start: int = 0,
    num_candidates_override: Optional[int] = None,
    measure_inference_time: bool = False,
) -> dict[str, object]:
    ...
```

**Step 2: Mirror candidate generation**

Implementation should closely follow training-time validation:

```python
num_candidates = (
    int(num_candidates_override)
    if num_candidates_override is not None
    else int(pyb_cfg.num_candidates)
)

scheduler_step_kwargs = {}
if pyb_cfg.candidate_scheduler_eta is not None:
    scheduler_step_kwargs["eta"] = float(pyb_cfg.candidate_scheduler_eta)

candidate_actions = []
candidate_seeds = []
for candidate_idx in range(num_candidates):
    candidate_seed = (
        int(pyb_cfg.diffusion_sampling_seed)
        + candidate_idx * 1_000_003
        + int(batch_start)
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(candidate_seed)
    result = policy.predict_action(
        obs_dict,
        generator=generator,
        num_inference_steps=pyb_cfg.inference_num_steps,
        scheduler_step_kwargs=scheduler_step_kwargs,
    )
    candidate_action = result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
    # apply candidate noise for candidate_idx > 0 exactly as in training-time validation
```

**Step 3: Reconstruct and score all candidates**

```python
candidate_results = []
candidate_score_details = []
for candidate_action in candidate_actions:
    candidate_result = validator.reconstruct_candidate(
        pred_action_horizon=candidate_action,
        start_joint_normalized=raw_obs["first_joint_angles_normalized"][0],
        end_joint_normalized=raw_obs["last_joint_angles_normalized"][0],
    )
    candidate_results.append(candidate_result)
    candidate_score_details.append(
        validator.score_candidate(
            workpiece_id=workpiece_id,
            normalized_control_points=candidate_result["normalized_control_points"],
            joint_trajectory=candidate_result["joint_trajectory"],
        )
    )
```

**Step 4: Select final candidate with the same rule as training validation**

Use the same score-key construction and selection logic as the training runner. Do not invent a new ranking rule.

Expected return payload shape:

```python
return {
    "selected_candidate_idx": selected_candidate_idx,
    "selected_candidate_seed": candidate_seeds[selected_candidate_idx],
    "selected_action_horizon": selected_action_horizon,
    "selected_joint_trajectory": selected_joint_trajectory,
    "selected_score_details": selected_score_details,
    "candidate_score_details": candidate_score_details,
    "candidate_seeds": candidate_seeds,
    "inference_elapsed_sec": inference_elapsed_sec,
}
```

**Step 5: Run syntax verification**

Run:

```bash
python -m py_compile scripts/validate_all_trajectories.py
```

Expected:
- No syntax errors.

**Step 6: Commit**

```bash
git add scripts/validate_all_trajectories.py
git commit -m "feat: add standalone candidate generation and selection helper"
```

---

### Task 4: Replace the single-candidate main-loop path with candidate-selection helper

**Objective:** Make the default full-validation workflow match training-time candidate selection.

**Files:**
- Modify: `scripts/validate_all_trajectories.py:430-500` (current per-trajectory loop region)

**Step 1: Remove direct one-shot inference path**

Replace this pattern:

```python
result = policy.predict_action(obs_dict)
pred_action_horizon = result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
joint_trajectory = validator.reconstruct_joint_trajectory(...)
```

with:

```python
selection = _predict_select_candidate(
    policy=policy,
    validator=validator,
    pyb_cfg=pyb_cfg,
    obs_dict=obs_dict,
    raw_obs=raw_obs,
    workpiece_id=wid,
    device=device,
    batch_start=idx,
    num_candidates_override=args.num_candidates,
    measure_inference_time=False,
)

joint_trajectory = np.asarray(
    selection["selected_joint_trajectory"],
    dtype=np.float32,
)
```

**Step 2: Surface candidate-selection metadata in per-trajectory JSON**

Add fields like:

```python
"selected_candidate_idx": int(selection["selected_candidate_idx"]),
"selected_candidate_seed": int(selection["selected_candidate_seed"]),
"num_candidates": int(args.num_candidates or pyb_cfg.num_candidates),
```

Optional but recommended for debugging:

```python
"selected_score": selection["selected_score_details"],
```

**Step 3: Keep evaluation metrics unchanged**

`validator.evaluate_trajectory(...)` should remain the source of final collision and goal metrics.

**Step 4: Run smoke validation on a tiny sample**

Run in the real training environment:

```bash
python scripts/validate_all_trajectories.py \
  --checkpoint-path '...ckpt' \
  --zarr-path data/realdex_bspline_free10.zarr \
  --stats-path data/realdex_bspline_stats_free10.npz \
  --device cuda:0 \
  --max-episodes 2
```

Expected:
- Script completes.
- Output JSON includes candidate-selection metadata.
- No regression to single-candidate behavior.

**Step 5: Commit**

```bash
git add scripts/validate_all_trajectories.py
git commit -m "feat: make standalone validation use 32-candidate scoring"
```

---

### Task 5: Add single-episode timed mode

**Objective:** Support exactly one validation episode and report end-to-end inference-and-selection latency.

**Files:**
- Modify: `scripts/validate_all_trajectories.py` main control flow

**Step 1: Select exactly one validation episode when requested**

Add logic after `val_episode_indices` is built:

```python
if args.single_episode_index is not None:
    if args.single_episode_index < 0 or args.single_episode_index >= len(val_episode_indices):
        raise IndexError(
            f"--single-episode-index {args.single_episode_index} is out of range "
            f"for {len(val_episode_indices)} validation episodes."
        )
    val_episode_indices = np.asarray(
        [int(val_episode_indices[int(args.single_episode_index)])],
        dtype=np.int64,
    )
    print(f"  (single-episode mode: validation episode offset {args.single_episode_index})")
```

Note: This flag should refer to the **index inside the validation subset**, not raw replay-buffer episode id. Make that explicit in help text and terminal output.

**Step 2: Start timing at the correct point**

Inside `_predict_select_candidate`, wrap only the desired region:

```python
start_time = time.perf_counter() if measure_inference_time else None
# candidate inference + reconstruction + scoring + selection
end_time = time.perf_counter() if measure_inference_time else None
inference_elapsed_sec = (
    float(end_time - start_time)
    if measure_inference_time and start_time is not None and end_time is not None
    else None
)
```

This matches the user request: start from inference start, end when the final selected output is ready.

**Step 3: Print timing in terminal summary**

For single mode, print:

```python
if selection["inference_elapsed_sec"] is not None:
    print(
        f"  Inference-to-selected-output time: "
        f"{selection['inference_elapsed_sec']:.6f} s"
    )
```

**Step 4: Save timing in JSON**

Add fields to the per-trajectory entry and/or summary:

```python
"inference_elapsed_sec": selection["inference_elapsed_sec"],
```

For single-episode mode, also add summary flags:

```python
"single_episode_mode": True,
"single_episode_validation_offset": int(args.single_episode_index),
```

**Step 5: Smoke-test the single-episode path**

Run:

```bash
python scripts/validate_all_trajectories.py \
  --checkpoint-path '...ckpt' \
  --zarr-path data/realdex_bspline_free10.zarr \
  --stats-path data/realdex_bspline_stats_free10.npz \
  --device cuda:0 \
  --single-episode-index 0 \
  --measure-inference-time
```

Expected:
- Only one episode is processed.
- Terminal output prints a timing value in seconds.
- JSON contains `inference_elapsed_sec`.

**Step 6: Commit**

```bash
git add scripts/validate_all_trajectories.py
git commit -m "feat: add single-trajectory timed validation mode"
```

---

### Task 6: Align defaults with training config and make overrides explicit

**Objective:** Ensure the script naturally behaves like training-time validation even when the user passes minimal CLI arguments.

**Files:**
- Modify: `scripts/validate_all_trajectories.py`

**Step 1: Keep checkpoint-derived defaults**

Prefer checkpoint-config defaults when CLI override is absent:

```python
num_candidates = (
    int(args.num_candidates)
    if args.num_candidates is not None
    else int(pyb_cfg.num_candidates)
)
```

**Step 2: Print the effective validation configuration**

Add terminal output before the loop:

```python
print(f"  effective_num_candidates={num_candidates}")
print(f"  candidate_selection={pyb_cfg.candidate_selection}")
print(f"  inference_num_steps={pyb_cfg.inference_num_steps}")
print(f"  candidate_action_noise_std={pyb_cfg.candidate_action_noise_std}")
print(f"  candidate_action_noise_clip={pyb_cfg.candidate_action_noise_clip}")
```

**Step 3: Verify default parity**

Run without `--num-candidates`:

```bash
python scripts/validate_all_trajectories.py \
  --checkpoint-path '...ckpt' \
  --device cuda:0 \
  --max-episodes 1
```

Expected:
- Effective candidate count prints as `32` for current checkpoint/config.

**Step 4: Commit**

```bash
git add scripts/validate_all_trajectories.py
git commit -m "chore: surface effective standalone validation config"
```

---

### Task 7: Validate output schema and backward compatibility expectations

**Objective:** Make sure the script still produces useful outputs for existing workflows while extending the JSON schema safely.

**Files:**
- Modify: `scripts/validate_all_trajectories.py`
- Output artifact to inspect after execution: `analysis_outputs/validation/.../per_trajectory_metrics.json`

**Step 1: Preserve existing summary keys**

Do not rename these existing keys unless absolutely necessary:

```json
{
  "summary": {
    "total_validation_episodes": 0,
    "trajectory_collision_rate": 0.0,
    "overall_segment_collision_rate": 0.0,
    "mean_goal_error_m": 0.0
  }
}
```

**Step 2: Add new keys without breaking old consumers**

Recommended additive keys:

```json
{
  "config": {
    "effective_num_candidates": 32,
    "single_episode_mode": false,
    "measure_inference_time": false
  },
  "per_trajectory": [
    {
      "selected_candidate_idx": 0,
      "selected_candidate_seed": 42,
      "inference_elapsed_sec": null
    }
  ]
}
```

**Step 3: Manual JSON inspection**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('analysis_outputs/validation/.../per_trajectory_metrics.json')
obj = json.loads(p.read_text())
print(obj['config'].keys())
print(obj['per_trajectory'][0].keys())
PY
```

Expected:
- Old keys still exist.
- New additive metadata appears.

**Step 4: Commit**

```bash
git add scripts/validate_all_trajectories.py
git commit -m "chore: extend standalone validation output schema"
```

---

## Tests / validation

### Minimal static validation

```bash
python -m py_compile scripts/validate_all_trajectories.py
```

Expected:
- No syntax errors.

### Full-workflow smoke test

```bash
python scripts/validate_all_trajectories.py \
  --checkpoint-path 'PATH/TO/CHECKPOINT.ckpt' \
  --zarr-path data/realdex_bspline_free10.zarr \
  --stats-path data/realdex_bspline_stats_free10.npz \
  --device cuda:0 \
  --max-episodes 2
```

Expected:
- Two validation episodes processed.
- Each episode uses 32-candidate selection by default.
- JSON output includes selected-candidate metadata.

### Single-trajectory timing validation

```bash
python scripts/validate_all_trajectories.py \
  --checkpoint-path 'PATH/TO/CHECKPOINT.ckpt' \
  --zarr-path data/realdex_bspline_free10.zarr \
  --stats-path data/realdex_bspline_stats_free10.npz \
  --device cuda:0 \
  --single-episode-index 0 \
  --measure-inference-time
```

Expected:
- Exactly one episode processed.
- Timing value printed and saved.
- Selected trajectory still comes from candidate scoring, not one-shot inference.

### Optional parity check against training-time validation

If a convenient checkpoint + sample exists, compare:
- standalone script on one episode
- training-time runner on the same episode / same seeds / same config

Expected:
- selected candidate index and final trajectory metrics should be consistent or near-identical.

---

## Risks / tradeoffs / open questions

### Risk 1: Logic duplication between script and `PyBulletValidationRunner.run()`
- If the script copies candidate logic instead of calling a reusable shared helper, future behavior can drift again.
- Preferred long-term cleanup: move candidate-generation/selection into a reusable method on `PyBulletValidationRunner` or `PyBulletCollisionValidator`, then use it from both training and standalone validation.
- For the first pass, local duplication is acceptable if kept very close to the training implementation.

### Risk 2: Timing definition ambiguity
- User requested: “从开始推理到最后输出结果的时间”.
- This plan interprets that as:
  - start: immediately before first candidate `policy.predict_action(...)`
  - end: after candidate reconstruction + scoring + final selected result is prepared
  - excludes: dataset loading, checkpoint loading, PyBullet runner initialization, JSON writing
- If a stricter interpretation is desired, decide whether `evaluate_trajectory(...)` should also be included in the timed region.

### Risk 3: Single-episode selector semantics
- `--single-episode-index` can refer either to:
  1. index inside the validation subset, or
  2. raw replay-buffer episode id.
- This plan recommends **validation-subset offset** because it is safer and easier for users. Make the printed output explicit.

### Risk 4: Performance overhead of 32-candidate validation
- Standalone validation will become slower than the current one-shot script by design.
- Single-episode timing mode mitigates this by enabling focused profiling.

### Open question 1
- Should the timed single-trajectory mode also write candidate-level details (all seeds / all score details), or only the selected result?
- Recommended default: save selected result only; add full candidate dump later if needed.

### Open question 2
- Should there be a `--single-no-scoring` profiling mode for pure one-shot latency?
- Not required by the current request. YAGNI: skip for now.

---

## Recommended implementation order

1. Add CLI flags.
2. Implement candidate inference-and-selection helper.
3. Switch full validation loop to selected-candidate path.
4. Add single-episode timed mode.
5. Add JSON/terminal metadata.
6. Run `py_compile` and one-episode/two-episode smoke tests.

---

## Example final commands for the implementer

### Full validation with training-parity candidate selection

```bash
python scripts/validate_all_trajectories.py \
  --checkpoint-path '/root/autodl-tmp/dp-train/c_space/.../checkpoints/epoch=0007-val_pybullet_collision_rate=0.270000.ckpt' \
  --zarr-path data/realdex_bspline_free10.zarr \
  --stats-path data/realdex_bspline_stats_free10.npz \
  --device cuda:0 \
  --output-dir analysis_outputs/validation/ckpt_epoch7_full
```

### Single trajectory with timing

```bash
python scripts/validate_all_trajectories.py \
  --checkpoint-path '/root/autodl-tmp/dp-train/c_space/.../checkpoints/epoch=0007-val_pybullet_collision_rate=0.270000.ckpt' \
  --zarr-path data/realdex_bspline_free10.zarr \
  --stats-path data/realdex_bspline_stats_free10.npz \
  --device cuda:0 \
  --single-episode-index 0 \
  --measure-inference-time \
  --output-dir analysis_outputs/validation/ckpt_epoch7_single_timed
```
