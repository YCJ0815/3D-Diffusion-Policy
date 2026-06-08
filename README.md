# 3D Diffusion Policy Training And Inference

本 README 记录当前仓库中可直接使用的：

- B-spline zarr 数据集生成命令
- `simple_dp3` 训练命令
- 日志与 checkpoint 保存位置
- 当前已有推理脚本的使用方式

以下命令默认在仓库根目录运行：

```bash
/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy
```

## 1. 生成 B-spline zarr 数据集

当前训练使用的是：

- 每条样本的 point cloud 观测
- `goal_position`
- `goal_direction`，当前是 `6` 维
- `first_joint_angles_normalized`
- `last_joint_angles_normalized`
- `action = 10 x 6`，对应 16 个控制点中间 10 个自由控制点残差

当前推荐命令：

```bash
/Users/ycj/miniconda3/envs/pybullet/bin/python scripts/build_bspline_zarr.py \
  --input-dirs data/raw_data/results data/raw_data/simple_results \
  --jobs-root data/raw_data/jobs \
  --simple-jobs-root data/raw_data/simple_jobs \
  --output-zarr data/realdex_bspline_free10.zarr \
  --stats-path data/raw_data/realdex_bspline_stats.npz \
  --mesh-cache-dir data/cache/mesh_points \
  --bspline-cache-dir data/cache/bspline_artifacts \
  --reuse-stats-if-exists \
  --norm-m 0.1 \
  --radius-m 0.1 \
  --height-m 0.1 \
  --radius-m-min 0.08 \
  --radius-m-max 0.14 \
  --height-m-min 0.1 \
  --height-m-max 0.14 \
  --augment-copies 4 \
  --add-reversed-copy \
  --num-output-points 512 \
  --trajectory-key q_plan \
  --target-steps 64 \
  --spline-degree 5 \
  --num-control-points 16
```

如果只想处理原始 `results`，把上面的：

```bash
--input-dirs data/raw_data/results data/raw_data/simple_results
```

改回：

```bash
--input-dir data/raw_data/results
```

说明：

- `--target-steps 64`：先把原始关节角轨迹重采样到 `64 x 6`，再拟合 B-spline。
- `--num-control-points 16`：拟合 16 个控制点。
- 当前脚本只把中间 `10` 个自由控制点残差写入 zarr。
- `--input-dirs ...`：支持同时扫描多个结果目录；`results -> jobs`、`simple_results -> simple_jobs` 会自动匹配对应 STL。
- `--jobs-root` / `--simple-jobs-root`：可显式指定常规工件和 simple 工件的 STL 根目录；适合结果目录和工件目录不在同一父目录下的情况。
- `--mesh-cache-dir`：缓存每个 STL 采样后的世界系点云，重复构建时不再重新采样网格。
- `--bspline-cache-dir`：缓存每个 transition 的 B-spline 拟合和 planning artifacts，stats 阶段和 zarr 阶段会复用。
- `--reuse-stats-if-exists`：当 `--stats-path` 中的构建元数据和当前输入完全匹配时，直接复用统计文件。
- `--augment-copies 4`：每个 transition 生成 4 份样本，第一份用基准裁剪参数，其余 3 份在设定范围内随机增强。
- `--add-reversed-copy`：为每条轨迹额外生成一份起终点互换的反向样本。
- `--num-output-points 512`：每帧点云固定采样到 512 点。
- 重新生成后的 `meta/workpiece_ids` 会按数据源编码：
  - `results/job_000 ~ job_009` 保持为 `0 ~ 9`
  - `simple_results/job_000 ~ job_064` 编码为 `1000 ~ 1064`
- 脚本已经加入进度条，会显示：
  - `fit bspline stats`
  - `build zarr episodes`

输出：

```text
data/realdex_bspline_free10.zarr
```

统计文件：

```text
data/raw_data/realdex_bspline_stats.npz
```

## 2. 训练脚本

当前推荐训练配置：

- 模型：`simple_dp3`
- 数据集：`realdex_bspline_free10.zarr`
- 验证集比例：`0.04`
- `horizon=10`
- `n_obs_steps=1`
- `n_action_steps=10`
- `batch_size=8`
- `EMA=True`

当前工件划分：

- 训练集：`results/job_000 ~ job_007`，以及 `simple_results/job_000 ~ job_059`
- 验证集：`results/job_008 ~ job_009`，以及 `simple_results/job_060 ~ job_064`

当前 `simple_dp3` 的关键维度：

- Obs encoder total feature dim = `192`
- Global condition dim = `192`
- U-Net internal condition dim = `256`
- down dims = `[64, 128, 128]`

正式训练命令：

```bash
python 3D-Diffusion-Policy/train.py \
  --config-name=simple_dp3.yaml \
  task=realdex_transition \
  task.dataset.zarr_path=data/realdex_bspline_free10.zarr \
  task.dataset.val_ratio=0.04 \
  horizon=10 \
  n_obs_steps=1 \
  n_action_steps=10 \
  training.debug=False \
  training.device=cuda:0 \
  training.use_ema=True \
  logging.mode=offline \
  exp_name=realdex-bspline-free10-simple-dp3 \
  dataloader.batch_size=32 \
  val_dataloader.batch_size=32 \
  checkpoint.save_ckpt=True \
  hydra.run.dir=/root/autodl-tmp/3D-Diffusion-Policy/runs/realdex_bspline_free10_simple_dp3
```

## 3. 日志位置

训练的逐 epoch 日志写在：

```text
/root/autodl-tmp/3D-Diffusion-Policy/runs/realdex_bspline_free10_simple_dp3/logs.json.txt
```

实时查看：

```bash
tail -f /root/autodl-tmp/3D-Diffusion-Policy/runs/realdex_bspline_free10_simple_dp3/logs.json.txt
```

只看关键字段：

```bash
tail -f /root/autodl-tmp/3D-Diffusion-Policy/runs/realdex_bspline_free10_simple_dp3/logs.json.txt | \
grep --line-buffered -E '"epoch"|"train_loss"|"val_loss"|"bc_loss"|"generalization_gap"'
```

说明：

- `train_loss`：当前训练模型的 epoch 平均训练损失
- `val_loss`：当前训练模型的验证损失
- `generalization_gap = (val_loss - train_loss) / train_loss`

## 4. Checkpoint 保存位置

普通 checkpoint：

```text
/root/autodl-tmp/3D-Diffusion-Policy/runs/realdex_bspline_free10_simple_dp3/checkpoints/latest.ckpt
```

这是通过：

```bash
checkpoint.save_ckpt=True
```

启用的。

额外的 generalization-gap checkpoint：

```text
/root/autodl-tmp/3D-Diffusion-Policy/runs/realdex_bspline_free10_simple_dp3/generalization_gap_checkpoints/
```

当前策略：

- 每个 epoch 计算 `generalization_gap`
- 如果连续 `5` 个 epoch 的 gap 都接近 `0.25`
- 则保存这 `5` 个 epoch 中 `val_loss` 最低的 checkpoint
- 该目录与普通 `checkpoints/` 独立，不冲突

## 5. 当前可用推理脚本

当前仓库里现成的推理脚本是：

```text
scripts/infer_transition_trajectory.py
```

入口说明见 [scripts/infer_transition_trajectory.py](/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy/scripts/infer_transition_trajectory.py:1)。

这个脚本的设计目标是：

- 读取一个 STL + transition NPZ
- 加载一个训练好的 checkpoint
- 输出 predicted action / delta trajectory / reconstructed joint trajectory

但它当前适配的是旧的 transition-delta 形式：

- 使用 `load_increment_stats`
- 用 delta trajectory 方式重建关节轨迹

因此：

- 如果 checkpoint 是旧的 transition action 模型，可以直接用。
- 如果 checkpoint 是当前 `bspline_free10` 模型，这个脚本不能直接无修改使用，因为当前 action 已经不是逐步 delta trajectory，而是 `10 x 6` 的 B-spline 自由控制点残差。

### 5.1 旧 transition 模型的推理命令

```bash
python scripts/infer_transition_trajectory.py \
  --stl-path data/raw_data/jobs/job_000/workpiece.stl \
  --npz-path data/raw_data/results/job_000/transition_0001_0008.npz \
  --checkpoint-path /path/to/latest.ckpt \
  --stats-path data/raw_data/results/job_000_increment_stats.npz \
  --output-dir /tmp/transition_inference \
  --device cuda:0
```

输出内容通常包括：

- `pred_action_window_normalized.npy`
- `pred_action_horizon_normalized.npy`
- `pred_delta_horizon.npy`
- `pred_joint_horizon.npy`
- `pred_joint_horizon.png`
- `summary.json`

## 6. 当前 B-spline 模型推理的状态

当前没有单独完成好的 `bspline_free10` 推理脚本。

如果要对当前 `simple_dp3 + bspline_free10` checkpoint 做推理，需要补一版新的推理逻辑：

1. 读取 point cloud 和 planning obs
2. 预测 `10 x 6` 的自由控制点残差
3. 把它填回完整 `16 x 6` 控制点残差
4. 还原控制点
5. 用 B-spline 基函数重建 joint trajectory

如果后续要补这一部分，建议新建一个单独脚本，例如：

```text
scripts/infer_bspline_trajectory.py
```

## 7. 当前配置摘要

当前与训练相关的重要默认值：

- learning rate = `1e-4`
- weight decay = `1e-4`
- dropout = `0.0`，当前模型没有显式 Dropout
- EMA = `True`
- `goal_direction.shape = [6]`

## 8. 相关文件

- 训练入口：[3D-Diffusion-Policy/train.py](/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy/3D-Diffusion-Policy/train.py:1)
- `simple_dp3` 配置：[simple_dp3.yaml](/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/simple_dp3.yaml:1)
- 任务配置：[realdex_transition.yaml](/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task/realdex_transition.yaml:1)
- B-spline 数据集脚本：[build_bspline_zarr.py](/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy/scripts/build_bspline_zarr.py:1)
- 旧 transition 推理脚本：[infer_transition_trajectory.py](/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy/scripts/infer_transition_trajectory.py:1)
