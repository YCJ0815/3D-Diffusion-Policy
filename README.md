# 3D Diffusion Policy Transition Training

## 生成 zarr 数据

在仓库根目录运行：

```bash
python scripts/build_transition_zarr.py \
  --input-dir data/raw_data/results/job_000 \
  --stl-path data/raw_data/jobs/job_000/workpiece.stl \
  --output-zarr 3D-Diffusion-Policy/data/realdex_transition.zarr \
  --stats-path data/raw_data/results/job_000_increment_stats.npz \
  --norm-m 0.1 \
  --trajectory-key q_playback \
  --target-steps 65
```

说明：

- `build_transition_zarr.py` 每次运行都会重新计算并覆盖 `--stats-path` 中的均值、方差和标准差。
- `--output-zarr` 每次运行都会覆盖旧 zarr 内容。
- `--trajectory-key q_playback` 表示使用每个 transition `.npz` 内的 `q_playback` 关节轨迹生成 delta action。
- `--output-zarr` 写到 `3D-Diffusion-Policy/data/realdex_transition.zarr`，这是训练配置 `realdex_transition.yaml` 默认读取的位置。

生成后的 zarr 结构：

```text
3D-Diffusion-Policy/data/realdex_transition.zarr
├── data
│   ├── point_cloud                      # [N_total, 1024, 3]
│   ├── goal_position                    # [N_total, 3]
│   ├── goal_direction                   # [N_total, 3]
│   ├── first_joint_angles_normalized    # [N_total, 6]
│   ├── last_joint_angles_normalized     # [N_total, 6]
│   └── action                           # [N_total, 6]
└── meta
    └── episode_ends                     # [N_episode]
```

## 启动训练

在仓库根目录运行：

```bash
python train.py --config-name=dp3.yaml   task=realdex_transition   task.dataset.zarr_path=/root/autodl-tmp/3D-Diffusion-Policy/data/job_000_transition.zarr   hydra.run.dir=/root/autodl-tmp/3D-Diffusion-Policy/3D-Diffusion-Policy/outputs/debug_realdex_transition_dp3_seed42   training.debug=True   training.seed=42   training.device=cuda:0   logging.mode=offline   exp_name=debug-realdex-transition-dp3   dataloader.num_workers=0   val_dataloader.num_workers=0   dataloader.batch_size=4   val_dataloader.batch_size=4
```

参数含义：

- `dp3`：使用 `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml`。
- `realdex_transition`：使用 `3D-Diffusion-Policy/diffusion_policy_3d/config/task/realdex_transition.yaml`。
- `0527`：实验附加标识，会进入输出目录名。
- `42`：随机种子。
- `0`：GPU id，对应脚本中的 `CUDA_VISIBLE_DEVICES=0`。

训练输出默认保存到：

```text
3D-Diffusion-Policy/data/outputs/realdex_transition-dp3-0527_seed42
```
