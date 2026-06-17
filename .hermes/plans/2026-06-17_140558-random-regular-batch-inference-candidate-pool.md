# Random Regular-Job Batch Inference + Candidate Pool Comparison Implementation Plan
> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 将 `scripts/infer_bspline_trajectories_batch.py` 从“按排序后相邻轨迹顺序取样并单路预测”改为“默认从 regular jobs 中随机抽取 10 条轨迹做预测”，并支持 candidate 轨迹候选池的开关、候选池大小可调（默认 32）、以及对同一批 10 条轨迹同时跑“关闭候选池 / 开启候选池”两套结果后做对比。

**Architecture:** 保持现有 batch inference 脚本为主入口，不改训练逻辑。新增“样本筛选层 + 候选池预测层 + 对比汇总层”：先稳定抽样固定的 regular-job NPZ 子集，再根据模式（baseline / candidate / compare）执行一次或两次预测，最后把每条轨迹与整批对比结果写入结构化输出目录和 manifest。候选池推理尽量复用 `policy.predict_action(..., generator=..., num_inference_steps=..., scheduler_step_kwargs=...)` 与 `pybullet_validation.py` 里已验证过的 multi-candidate 采样思路，避免发明第二套随机采样协议。

**Tech Stack:** Python 3.11, argparse, pathlib, numpy, torch, 项目内 `TrainDP3Workspace` / B-spline 重建工具 / `diffusion_policy_3d.common.pybullet_validation` 既有 candidate-sampling 逻辑。

---

## 1. 当前代码与需求差异审计

### 已确认的现状
- `scripts/infer_bspline_trajectories_batch.py`
  - `collect_npz_files()` 目前是递归收集后 `sorted(...)`，再直接按顺序截断 `max_files`，所以“前 10 条”天然是相邻样本，不是随机样本。
  - `main()` 里只跑一条 `policy.predict_action(obs_dict)` 路径，没有 candidate pool 开关，也没有同一样本双路对比。
  - `run_single_inference()` 只保存单次 `pred_action_horizon` / `pred_joint_horizon` 等产物，summary 里也没有记录采样模式、随机种子、候选池大小、选中 candidate index 等信息。
- `3D-Diffusion-Policy/diffusion_policy_3d/common/pybullet_validation.py`
  - 已经存在成熟的随机 episode 抽样与 deterministic candidate sampling 方案：
    - `prepare_episode_subset()` 支持 `max_episodes + random_sample_episodes + random_seed`
    - `PyBulletValidationRunner.run()` 支持 `num_candidates`、`diffusion_sampling_seed`、`candidate_scheduler_eta`、`candidate_action_noise_std`，并对每个 candidate 调 `policy.predict_action(..., generator=...)`
  - 默认 dataclass `num_candidates=16`，但 `dp3_cspace.yaml` 里实际验证配置已经将 `training.pybullet_eval.num_candidates: 32`。
- `policy.predict_action()`（`dp3.py` / `dp3_cspace.py`）已经支持：
  - `generator`
  - `num_inference_steps`
  - `scheduler_step_kwargs`
  这意味着 inference 脚本可以不改 policy API，直接按 PyBullet validation 的方式做 multi-candidate 采样。

### 需求落点
1. 默认从 **regular jobs** 中随机抽 10 条轨迹，而不是顺序前 10 条。
2. inference 要支持 candidate 候选池开关：
   - 关闭：单轨迹单次预测
   - 开启：同一轨迹生成 `N` 个 candidates，再按规则选一个
3. 支持对 **相同 10 条轨迹** 同时跑：
   - baseline（不开 candidate pool）
   - candidate（开 candidate pool）
   并输出可直接比较的结果。
4. candidate pool 默认大小设为 **32**，但可以通过命令行参数改。

### 需要显式澄清/固定的策略
实现时不要把这些留成隐式行为：
- “regular jobs” 应定义为 `infer_source_kind(...) == "regular"` 的 NPZ；simple jobs 默认不进入随机 10 条样本池。
- “相同 10 条轨迹比较” 必须先抽样一次并冻结样本列表，再分别跑 baseline/candidate，两边复用同一列表，不能各自重新抽样。
- candidate 选优规则应优先复用 `pybullet_validation._select_lowest_candidate_score_index()` + `validator.score_candidate()` 的 SDF 安全排序；若调用方未提供 candidate-scoring 必需资源（例如 jobs root / SDF / URDF / stats），计划里要保留一个显式 fallback 路径，不要默默退化。

---

## 2. 实施范围

### 主要改动文件
- **Modify:** `scripts/infer_bspline_trajectories_batch.py`（主实现，预计主要改动）
- **Possibly Modify:** `scripts/infer_bspline_trajectory.py`（仅当需要抽共享 helper，例如保存对比图/复用 summary 结构时）
- **Reference only:** `3D-Diffusion-Policy/diffusion_policy_3d/common/pybullet_validation.py`（复用 candidate sampling / scoring 设计，不直接重写其训练验证逻辑）

### 不在本次范围
- 不改训练阶段的 `training.pybullet_eval` 执行路径。
- 不改 `policy.predict_action()` 接口。
- 不做大规模重构到新 package/module，除非 batch 脚本局部复杂度已无法维持。

---

## 3. 输出目录与行为设计

### 建议新增 CLI 语义
在 `scripts/infer_bspline_trajectories_batch.py` 新增以下参数（命名可微调，但要保持一眼可懂）：

```python
parser.add_argument(
    "--sample-source",
    type=str,
    choices=["regular", "simple", "all"],
    default="regular",
    help="Which source pool to sample transition NPZs from before inference.",
)
parser.add_argument(
    "--sample-count",
    type=int,
    default=10,
    help="Number of randomly sampled trajectories to infer.",
)
parser.add_argument(
    "--sample-seed",
    type=int,
    default=42,
    help="Seed for deterministic random sampling of NPZ trajectories.",
)
parser.add_argument(
    "--sampling-mode",
    type=str,
    choices=["baseline", "candidate", "compare"],
    default="baseline",
    help="baseline=single prediction; candidate=best of candidate pool; compare=run both on same sampled trajectories.",
)
parser.add_argument(
    "--enable-candidate-pool",
    action="store_true",
    help="Shortcut switch for candidate mode when compare mode is not used.",
)
parser.add_argument(
    "--num-candidates",
    type=int,
    default=32,
    help="Candidate pool size when candidate selection is enabled.",
)
parser.add_argument(
    "--candidate-seed",
    type=int,
    default=42,
    help="Base seed for deterministic candidate sampling.",
)
parser.add_argument(
    "--candidate-inference-steps",
    type=int,
    default=None,
    help="Override diffusion inference steps for candidate sampling.",
)
parser.add_argument(
    "--candidate-scheduler-eta",
    type=float,
    default=1.0,
    help="DDIM eta forwarded via scheduler_step_kwargs when candidate sampling is enabled.",
)
parser.add_argument(
    "--candidate-action-noise-std",
    type=float,
    default=0.0,
    help="Optional extra Gaussian noise added to candidate actions after the first candidate, matching pybullet validation behavior.",
)
```

### 建议新增输出结构
对 `--sampling-mode compare`：

```text
<output_root>/
  sampled_npz_manifest.json
  batch_inference_manifest.json
  compare_summary.json
  <relative_npz_parent>/
    <transition_name>_bspline_inference/
      baseline/
        summary.json
        pred_joint_horizon.npy
        ...
      candidate/
        summary.json
        pred_joint_horizon.npy
        candidate_scores.json
        selected_candidate_index.txt
        ...
      compare/
        summary.json
        pred_joint_horizon_compare.png
```

对 `baseline` 或 `candidate` 单模式，可保持现在目录风格，但建议也在 `summary.json` 增加 `sampling_mode`、`candidate_pool_enabled` 等字段，避免后续文件混淆。

---

## 4. 分任务实施计划

### Task 1: 固化现状与样本选择契约
**Objective:** 在动实现前，先把“随机选样本而不是顺序截断”的行为定义清楚，并把 regular/simple/all 的筛选规则独立出来。

**Files:**
- Modify: `scripts/infer_bspline_trajectories_batch.py:114-123`（`collect_npz_files`）
- Modify: `scripts/infer_bspline_trajectories_batch.py:53-58`（parser 附近，新增 sample 参数）
- Test: `scripts/infer_bspline_trajectories_batch.py`（先用轻量 helper 自检）

**Step 1: 提取样本筛选 helper**
新增 helper，签名建议：

```python
def filter_npz_files_by_source(
    npz_files: list[pathlib.Path],
    input_dirs: list[pathlib.Path],
    sample_source: str,
) -> list[pathlib.Path]:
    if sample_source == "all":
        return list(npz_files)
    return [
        path for path in npz_files
        if infer_source_kind(path, input_dirs) == sample_source
    ]
```

**Step 2: 提取随机采样 helper**

```python
def sample_npz_files(
    npz_files: list[pathlib.Path],
    sample_count: int,
    sample_seed: int,
) -> list[pathlib.Path]:
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if len(npz_files) < sample_count:
        raise ValueError(
            f"Requested {sample_count} samples, but only found {len(npz_files)} eligible NPZ files."
        )
    rng = np.random.default_rng(sample_seed)
    sampled_indices = np.sort(rng.choice(len(npz_files), size=sample_count, replace=False))
    return [npz_files[int(idx)] for idx in sampled_indices]
```

> 这里排序 sampled indices 的目的不是恢复“相邻样本”，而是让 manifest 稳定、输出目录遍历可读；样本本身仍然是随机抽出来的。

**Step 3: 在 `main()` 中用新契约替换旧 `max_files` 截断**
逻辑目标：
1. 收集全部 `transition_*.npz`
2. 按 `sample_source` 过滤（默认 regular）
3. 若用户显式给了 `--max-files`，把它保留为“原始收集上限”或直接标记 deprecated；不要与 `--sample-count` 语义冲突
4. 用 `sample_npz_files(...)` 固定 10 条（默认值）

**Step 4: 输出 sampled manifest**
新增 `sampled_npz_manifest.json`，至少包含：

```json
{
  "sample_source": "regular",
  "sample_count": 10,
  "sample_seed": 42,
  "eligible_count": 317,
  "sampled_npz_paths": ["..."]
}
```

**Step 5: Commit**
```bash
git add scripts/infer_bspline_trajectories_batch.py
git commit -m "feat: randomize regular-job batch inference sampling"
```

---

### Task 2: 给 batch inference 增加明确的模式开关
**Objective:** 把“单路预测 / candidate pool / 双路 compare”做成清晰、互斥、可组合的 CLI/内部配置，而不是散落的 if 分支。

**Files:**
- Modify: `scripts/infer_bspline_trajectories_batch.py:17-58`（parser）
- Modify: `scripts/infer_bspline_trajectories_batch.py:126-251`（推理主流程）

**Step 1: 归一化模式参数**
新增 helper：

```python
def resolve_sampling_mode(args) -> str:
    if args.sampling_mode == "compare":
        return "compare"
    if args.enable_candidate_pool:
        return "candidate"
    return args.sampling_mode
```

**Step 2: 校验参数组合**
至少校验：
- `num_candidates >= 1`
- `sampling_mode == "baseline"` 时忽略 `num_candidates` 但在 summary 记清楚
- `sampling_mode in {"candidate", "compare"}` 时需要可复现的 `candidate_seed`
- 如果后续 candidate scoring 依赖 SDF/URDF/stats，则在启动前一次性检查，不要跑到第 7 个样本才报错

**Step 3: 在 manifest 中记录模式**
`batch_inference_manifest.json` 顶层建议新增：

```json
{
  "sampling_mode": "compare",
  "candidate_pool_enabled": true,
  "num_candidates": 32,
  "candidate_seed": 42
}
```

**Step 4: Commit**
```bash
git add scripts/infer_bspline_trajectories_batch.py
git commit -m "feat: add inference sampling modes and candidate pool flags"
```

---

### Task 3: 抽出单次预测 helper，给 baseline / candidate 共用
**Objective:** 先把“单次预测”从 `run_single_inference()` 内部分层出来，避免后面 candidate 模式复制大段 B-spline 重建与保存逻辑。

**Files:**
- Modify: `scripts/infer_bspline_trajectories_batch.py:138-251`

**Step 1: 提取纯预测 helper**
建议拆成两层：

```python
def predict_single_action_horizon(
    policy,
    obs_dict: dict[str, torch.Tensor],
    *,
    generator=None,
    num_inference_steps: int | None = None,
    scheduler_step_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        result = policy.predict_action(
            obs_dict,
            generator=generator,
            num_inference_steps=num_inference_steps,
            scheduler_step_kwargs=scheduler_step_kwargs,
        )
    return {
        "pred_action_window": result["action"][0].detach().cpu().numpy().astype(np.float32),
        "pred_action_horizon": result["action_pred"][0].detach().cpu().numpy().astype(np.float32),
    }
```

```python
def reconstruct_prediction_artifacts(...):
    # 输入 pred_action_horizon，输出 pred_joint_horizon、delta_w、w_star 等
```

**Step 2: 让 baseline 路径先走新 helper**
保持现有 baseline 行为不变：仍然输出当前 `.npy`、`.png`、`summary.json`。这一步不加 candidate 逻辑，只做安全重构。

**Step 3: 让 `summary.json` 增加新元数据**
建议增加：

```python
summary.update({
    "sampling_mode": mode,
    "candidate_pool_enabled": False,
    "candidate_count": 1,
    "selected_candidate_index": 0,
    "sample_seed": args.sample_seed,
})
```

**Step 4: Commit**
```bash
git add scripts/infer_bspline_trajectories_batch.py
git commit -m "refactor: split batch inference prediction and reconstruction helpers"
```

---

### Task 4: 复用 PyBullet candidate 采样方案实现候选池预测
**Objective:** 在 inference 脚本中加入 deterministic multi-candidate sampling，默认候选池大小 32，并能稳定复现。

**Files:**
- Modify: `scripts/infer_bspline_trajectories_batch.py`
- Reference: `3D-Diffusion-Policy/diffusion_policy_3d/common/pybullet_validation.py:1553-1608`（candidate 采样方式）

**Step 1: 提取 candidate 采样 helper**
参考 `PyBulletValidationRunner.run()` 的做法，新增：

```python
def predict_candidate_action_horizons(
    policy,
    obs_dict: dict[str, torch.Tensor],
    *,
    device: torch.device,
    num_candidates: int,
    candidate_seed: int,
    sample_index: int,
    num_inference_steps: int | None,
    candidate_scheduler_eta: float | None,
    candidate_action_noise_std: float,
    candidate_action_noise_clip: float | None,
) -> list[dict[str, np.ndarray]]:
    scheduler_step_kwargs = {}
    if candidate_scheduler_eta is not None:
        scheduler_step_kwargs["eta"] = float(candidate_scheduler_eta)

    outputs = []
    for candidate_idx in range(num_candidates):
        seed = candidate_seed + candidate_idx * 1_000_003 + sample_index
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        pred = predict_single_action_horizon(
            policy,
            obs_dict,
            generator=generator,
            num_inference_steps=num_inference_steps,
            scheduler_step_kwargs=scheduler_step_kwargs,
        )
        # optional post-noise: match pybullet_validation behavior
        outputs.append({**pred, "candidate_seed": seed, "candidate_index": candidate_idx})
    return outputs
```

**Step 2: 默认值固定为 32**
`--num-candidates` 默认值必须是 32；这要覆盖当前 `PyBulletValidationConfig` dataclass 里的 16，但这里只改 inference CLI 默认，不必强行修改训练 dataclass 默认，除非实现者决定统一两边默认值。

**Step 3: 在 candidate 模式下保存全部候选或至少保存选中项 + 分数表**
最少要求：
- `candidate_scores.json`
- `selected_candidate_index.txt`
- `selected_candidate_seed.txt`
- 选中 candidate 的 `pred_joint_horizon.npy`

如果磁盘可接受，建议再保存：
- `all_candidate_min_sdf.npy`
- `all_candidate_action_horizon.npy`

**Step 4: Commit**
```bash
git add scripts/infer_bspline_trajectories_batch.py
git commit -m "feat: add deterministic multi-candidate batch inference"
```

---

### Task 5: 复用 PyBullet 的 SDF 排序规则给 candidate 选优
**Objective:** 候选池不能只是“生成 32 条然后随便取第一条”；要明确如何选出最终 candidate，并尽量与现有 PyBullet validation 对齐。

**Files:**
- Modify: `scripts/infer_bspline_trajectories_batch.py`
- Reference: `3D-Diffusion-Policy/diffusion_policy_3d/common/pybullet_validation.py:111-188`（`_score_safety_sdf_candidate`）
- Reference: `3D-Diffusion-Policy/diffusion_policy_3d/common/pybullet_validation.py:98-109`（`_select_lowest_candidate_score_index`）
- Reference: `3D-Diffusion-Policy/diffusion_policy_3d/common/pybullet_validation.py:731-780`（`score_candidate`）

**Step 1: 不复制复杂逻辑，优先导入复用**
实现优先级：
1. 尝试直接从 `pybullet_validation` 导入：
   - `PyBulletValidationConfig`
   - `PyBulletCollisionValidator`
   - `_select_lowest_candidate_score_index`
2. 用一个局部 helper 组装 validator 所需 config
3. 用 validator 对每个 candidate 轨迹做 `score_candidate(...)`

**Step 2: 为 inference 脚本增加 candidate scoring 前置校验**
需要在计划中显式规定：candidate 模式若要按 SDF 安全选优，至少要有：
- `jobs_root`
- `simple_jobs_root`（若允许 simple）
- `stats_path`
- `urdf_path` 或默认 URDF 可解析
- 对应 workpiece 的 `workpiece_sdf.npz`

**Step 3: 定义 fallback 行为，不允许静默错误**
推荐顺序：
- 若 `--candidate-selection weighted_sdf` 且所需资源缺失：**直接 fail-fast 报错**
- 若实现者必须保留无 SDF 运行通路，则新增显式选项：

```python
parser.add_argument(
    "--candidate-selection",
    type=str,
    choices=["first", "weighted_sdf"],
    default="weighted_sdf",
)
```

这样资源不足时，用户可手动切回 `first`，而不是脚本默默退化。

**Step 4: `summary.json` 记录选优依据**
至少包含：

```json
{
  "candidate_selection": "weighted_sdf",
  "num_candidates": 32,
  "selected_candidate_index": 7,
  "selected_candidate_score_key": [0, 0, 0, -0.012, -0.011, 0.0]
}
```

**Step 5: Commit**
```bash
git add scripts/infer_bspline_trajectories_batch.py
git commit -m "feat: score and select batch inference candidates by sdf safety"
```

---

### Task 6: 实现同批 10 条轨迹的 baseline vs candidate 对比模式
**Objective:** 支持在一次命令中，对同一批 sampled regular-job 轨迹同时输出 baseline 与 candidate 两套推理结果，并生成可比较汇总。

**Files:**
- Modify: `scripts/infer_bspline_trajectories_batch.py`
- Possibly Modify: `scripts/infer_bspline_trajectory.py`（如果要复用画图工具扩展对比图）

**Step 1: 固定 compare 模式主循环结构**
伪代码应接近：

```python
sampled_npz_files = sample_once(...)
for sample_idx, npz_path in enumerate(sampled_npz_files):
    if mode == "compare":
        baseline_summary = run_single_mode_inference(..., mode="baseline")
        candidate_summary = run_single_mode_inference(..., mode="candidate")
        compare_summary = build_compare_summary(baseline_summary, candidate_summary)
    elif mode == "candidate":
        candidate_summary = run_single_mode_inference(..., mode="candidate")
    else:
        baseline_summary = run_single_mode_inference(..., mode="baseline")
```

**Step 2: 每条轨迹输出 compare summary**
建议结构：

```json
{
  "npz_path": "...",
  "baseline_output_dir": ".../baseline",
  "candidate_output_dir": ".../candidate",
  "baseline_min_sdf_distance_m": -0.004,
  "candidate_min_sdf_distance_m": 0.012,
  "min_sdf_gain_m": 0.016,
  "baseline_selected_candidate_index": 0,
  "candidate_selected_candidate_index": 7,
  "same_sample_seed": 42,
  "same_candidate_seed": 42
}
```

**Step 3: 批级 compare 汇总**
新增 `compare_summary.json`，聚合：
- 10 条样本中 candidate 比 baseline 更优的条数
- mean / median `min_sdf_gain_m`
- baseline 与 candidate 的 collision-free rate（若接入 validator）
- 每条样本目录索引

**Step 4: 如有图像对比，保持最小侵入**
只在真正需要时新增对比图，例如：
- `pred_joint_horizon_compare.png`：同一轨迹上叠加 baseline/candidate/gt

不要为了美观引入额外 plotting framework。

**Step 5: Commit**
```bash
git add scripts/infer_bspline_trajectories_batch.py scripts/infer_bspline_trajectory.py
git commit -m "feat: compare baseline and candidate inference on shared sampled trajectories"
```

---

### Task 7: 让 manifest / summary 足够可复现、可审计
**Objective:** 让后续任何人只看输出文件就知道：抽了哪 10 条、用什么 seed、candidate pool 有没有开、开了多大、比较结果如何。

**Files:**
- Modify: `scripts/infer_bspline_trajectories_batch.py`

**Step 1: 扩展顶层 manifest**
顶层 `batch_inference_manifest.json` 建议新增：

```json
{
  "sample_source": "regular",
  "sample_count": 10,
  "sample_seed": 42,
  "sampling_mode": "compare",
  "candidate_pool_enabled": true,
  "num_candidates": 32,
  "candidate_selection": "weighted_sdf",
  "candidate_seed": 42,
  "processed": [...],
  "failed": [...]
}
```

**Step 2: 每个 summary 都写完整 provenance**
每条 summary 都应带：
- checkpoint / stats / npz / stl
- `sample_index`
- `sample_source_kind`
- `sample_seed`
- `sampling_mode`
- `candidate_pool_enabled`
- `num_candidates`
- `selected_candidate_index`
- `candidate_seed(s)` 或 seed generation 规则

**Step 3: 失败样本也要写清 mode/context**
`failed` 项不要只存 `error`，还要存：
- `sampling_mode`
- `candidate_pool_enabled`
- `npz_path`
- `stl_path`（若已解析）
- `sample_index`

**Step 4: Commit**
```bash
git add scripts/infer_bspline_trajectories_batch.py
git commit -m "chore: enrich batch inference manifests for reproducibility"
```

---

### Task 8: 预留最小测试与验证闭环
**Objective:** 给实现者一条最小但足够可信的验证路径，区分“必须通过的任务正确性门槛”和“已有基线问题”。

**Files:**
- Test target: `scripts/infer_bspline_trajectories_batch.py`
- Test target: 如新增 helper 可考虑抽到可导入函数后补轻量测试文件 `tests/`（若仓库已有测试目录结构适配）

**Step 1: 先做不依赖大模型权重的静态检查**
Run:
```bash
python -m py_compile scripts/infer_bspline_trajectories_batch.py scripts/infer_bspline_trajectory.py
```
Expected:
- PASS，无语法错误

**Step 2: 验证 CLI 解析**
Run:
```bash
python scripts/infer_bspline_trajectories_batch.py --help
```
Expected:
- 帮助文本中出现：`--sample-source`、`--sample-count`、`--sample-seed`、`--sampling-mode`、`--num-candidates`

**Step 3: 轻量 smoke test：只验证抽样与 manifest，不跑重 checkpoint**
如果实现者愿意再拆 helper，可用一个最小 Python 片段或单元测试验证：

```python
all_files = [Path(f"/tmp/job_{i:03d}/transition_{i}.npz") for i in range(20)]
sampled = sample_npz_files(all_files, sample_count=10, sample_seed=42)
assert len(sampled) == 10
assert len(set(sampled)) == 10
assert sampled == sample_npz_files(all_files, sample_count=10, sample_seed=42)
```

**Step 4: 真实 targeted smoke run（任务正确性门槛）**
在有 checkpoint/stats/data 的机器上跑：

```bash
python scripts/infer_bspline_trajectories_batch.py \
  --input-dirs /path/to/results /path/to/simple_results \
  --checkpoint-path /path/to/model.ckpt \
  --stats-path /path/to/realdex_bspline_stats_free10.npz \
  --output-root /tmp/bspline_compare_out \
  --jobs-root /path/to/jobs \
  --simple-jobs-root /path/to/simple_jobs \
  --sample-source regular \
  --sample-count 10 \
  --sample-seed 42 \
  --sampling-mode compare \
  --num-candidates 32 \
  --candidate-seed 42 \
  --urdf-path config/robot-model/ur5e_with_pen.urdf
```

Expected:
- 只抽 regular jobs 的 10 条轨迹
- 输出 `sampled_npz_manifest.json`
- 每条样本都有 `baseline/` 与 `candidate/`
- 生成 `compare_summary.json`

**Step 5: 必须通过的 correctness gates**
- `python -m py_compile ...`
- `python scripts/infer_bspline_trajectories_batch.py --help`
- 真实 compare smoke run 至少 1 次成功完成（最好用 `--sample-count 2` 先快速打通，再上 10）

**Step 6: 允许存在但需记录的 baseline 问题**
若仓库原本存在与本任务无关的问题，应记录但不扩 scope 修复，例如：
- 其他训练配置 YAML 的历史字段不一致
- 与本脚本无关的旧测试失败
- 某些 simple-job 缺失 `workpiece_sdf.npz`

**Step 7: Commit**
```bash
git add scripts/infer_bspline_trajectories_batch.py
# 如果新增测试文件，也一并 add
git commit -m "test: validate random sampling and candidate compare batch inference"
```

---

## 5. 建议的内部函数布局

为了避免 `main()` 继续膨胀，建议将 batch 脚本内部组织成这些 helper：

```python
def collect_npz_files(...): ...
def filter_npz_files_by_source(...): ...
def sample_npz_files(...): ...
def resolve_sampling_mode(args): ...
def build_candidate_validator(args): ...
def predict_single_action_horizon(...): ...
def reconstruct_prediction_artifacts(...): ...
def save_prediction_artifacts(...): ...
def run_baseline_inference(...): ...
def run_candidate_inference(...): ...
def build_compare_summary(...): ...
```

原则：
- `run_*_inference` 负责 orchestration
- `predict_*` 负责模型推理
- `reconstruct_*` 负责 B-spline 后处理
- `save_*` 负责 I/O

不要让一个函数同时做“抽样 + 推理 + 选优 + 写盘 + 汇总”。

---

## 6. 风险、取舍与开放问题

### 风险 1：candidate 选优依赖 SDF 资源
如果候选池开启后要走 `weighted_sdf`，但某些 regular jobs 没有 `workpiece_sdf.npz`，compare 模式会中途失败。

**建议取舍：**
- 默认严格失败，保证“candidate 模式 = 真正有选优能力”
- 若用户确实只想看多次随机采样而不做 SDF 选优，再显式用 `--candidate-selection first`

### 风险 2：compare 模式输出体积明显变大
每条轨迹多一套目录，若还保存全部 candidate 的 `.npy`，磁盘会翻倍以上。

**建议取舍：**
- 默认只保存选中 candidate + 分数表
- 如后续需要调试，再加 `--save-all-candidates`

### 风险 3：`--max-files` 与 `--sample-count` 语义冲突
当前脚本已有 `--max-files`，如果继续保留，很容易让使用者误解“到底先截断还是先随机”。

**建议取舍：**
- 最好将 `--max-files` 降级为内部诊断用途，help 文案明确写成“cap raw discovered NPZ files before random sampling”
- 或直接在本次改造中弃用，统一走 `--sample-count`

### 风险 4：candidate 默认 32 与训练验证默认 16 不一致
`PyBulletValidationConfig` dataclass 默认 `num_candidates=16`，但 `dp3_cspace.yaml` 已设为 32。本任务如果只改脚本默认值，会存在“脚本默认 32、验证 dataclass 默认 16”的分裂。

**建议取舍：**
- 本次至少让 inference 脚本默认 32，满足用户当前需求
- 若实现者想顺手统一，需单独审查训练/验证调用面，避免带出无关副作用

### 开放问题（实现前最好确认，但不阻碍写计划）
1. candidate 模式最终是否必须接入 SDF/pybullet 选优，还是“多采样后只保存全部候选”也可接受？
2. compare 汇总最关心的指标是 `min_sdf_gain_m`、collision-free rate，还是更偏 joint 轨迹可视化？
3. 是否需要把这套“随机 10 条 regular-job compare 推理”再封装成单独 shell 脚本/README 示例？

---

## 7. 最终交付验收标准

实现完成后，以下陈述都应为真：

- 默认执行 batch inference 时，样本池来自 **regular jobs**，不是 simple jobs。
- 默认轨迹数是 **10**，且这 10 条由 `sample_seed` 控制的随机采样得到，不是顺序相邻样本。
- 可通过参数调整采样数和采样 seed。
- 可通过参数开启/关闭 candidate pool。
- candidate pool 默认大小是 **32**，也可通过参数调整。
- compare 模式会对 **同一批 sampled 轨迹** 输出 baseline 与 candidate 两套结果，并生成对比汇总。
- manifest / summary 能追溯：样本列表、随机 seed、候选池大小、是否启用 candidate、最终选中了哪个 candidate。

---

## 8. 执行建议

推荐实现顺序：
1. 先完成 Task 1~3，确保 baseline 路径不坏
2. 再完成 Task 4~5，打通 candidate pool 与选优
3. 最后完成 Task 6~8，补 compare 汇总和验证

Plan complete and saved. Ready to execute using subagent-driven-development — I'll dispatch a fresh subagent per task with two-stage review (spec compliance then code quality). Shall I proceed?
