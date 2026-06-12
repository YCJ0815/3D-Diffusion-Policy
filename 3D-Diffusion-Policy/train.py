if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import json
import contextlib
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.transition_dataset import TransitionTrajectoryDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.common.pybullet_validation import (
    PyBulletValidationConfig,
    PyBulletValidationRunner,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _is_better_metric(value, best_value, mode, min_delta):
    if best_value is None:
        return True
    if mode == "min":
        return value < (best_value - min_delta)
    if mode == "max":
        return value > (best_value + min_delta)
    raise ValueError(f"Unsupported metric mode: {mode}")


def _override_optimizer_lr(optimizer, lr: float):
    lr = float(lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        # Keep scheduler initialization consistent with the overridden LR.
        param_group["initial_lr"] = lr


def _resolve_amp_dtype(name: str) -> torch.dtype:
    amp_dtype = str(name).lower()
    if amp_dtype in ("fp16", "float16", "half"):
        return torch.float16
    if amp_dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def _build_precision_runtime(cfg, device: torch.device) -> dict[str, object]:
    perf_cfg = OmegaConf.select(cfg, "training.performance", default={}) or {}
    tf32_enabled = bool(perf_cfg.get("tf32", device.type == "cuda"))
    cudnn_benchmark = bool(perf_cfg.get("cudnn_benchmark", device.type == "cuda"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if tf32_enabled:
            torch.set_float32_matmul_precision("high")

    amp_cfg = perf_cfg.get("amp", {}) or {}
    amp_enabled = bool(amp_cfg.get("enabled", device.type == "cuda"))
    amp_dtype_name = str(amp_cfg.get("dtype", "bf16"))
    amp_dtype = _resolve_amp_dtype(amp_dtype_name)
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

    scaler = None
    if amp_enabled and amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    return {
        "tf32_enabled": tf32_enabled,
        "cudnn_benchmark": cudnn_benchmark,
        "amp_enabled": amp_enabled,
        "amp_dtype_name": amp_dtype_name,
        "amp_dtype": amp_dtype,
        "scaler": scaler,
    }


def _autocast_context(precision_runtime: dict[str, object], device: torch.device):
    if not precision_runtime["amp_enabled"]:
        return contextlib.nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=precision_runtime["amp_dtype"],
    )


def _sanitize_dataloader_cfg(dataloader_cfg):
    if int(dataloader_cfg.get("num_workers", 0)) == 0 and bool(
            dataloader_cfg.get("persistent_workers", False)):
        dataloader_cfg.persistent_workers = False
    return dataloader_cfg


def _compile_module_if_requested(
        owner,
        module_name: str,
        performance_cfg,
        compile_log: list[str]) -> None:
    compile_cfg = performance_cfg.get("compile", {}) or {}
    if not bool(compile_cfg.get("enabled", False)):
        return
    if not hasattr(torch, "compile"):
        compile_log.append(f"{module_name}=skip(torch.compile unavailable)")
        return
    if not hasattr(owner, module_name):
        return
    module = getattr(owner, module_name)
    if module is None:
        return
    mode = str(compile_cfg.get("mode", "reduce-overhead"))
    fullgraph = bool(compile_cfg.get("fullgraph", False))
    dynamic = bool(compile_cfg.get("dynamic", False))
    try:
        compiled = torch.compile(
            module,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
    except Exception as exc:
        compile_log.append(f"{module_name}=fallback({type(exc).__name__})")
        return
    setattr(owner, module_name, compiled)
    compile_log.append(f"{module_name}=compiled")


def _optimize_policy_modules(policy, cfg) -> list[str]:
    perf_cfg = OmegaConf.select(cfg, "training.performance", default={}) or {}
    compile_log = []
    for module_name in ("obs_encoder", "model", "cspace_encoder", "fusion_mlp"):
        _compile_module_if_requested(policy, module_name, perf_cfg, compile_log)
    return compile_log

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = bool(getattr(cfg.training, 'run_validation', True))
        _sanitize_dataloader_cfg(cfg.dataloader)
        _sanitize_dataloader_cfg(cfg.val_dataloader)
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                resume_override_lr = OmegaConf.select(
                    cfg, "training.resume_override_lr", default=None)
                if resume_override_lr is not None:
                    _override_optimizer_lr(self.optimizer, resume_override_lr)
                    cprint(
                        f"Overriding resumed optimizer lr to {float(resume_override_lr):.6g}",
                        "yellow",
                    )

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        model_compile_log = _optimize_policy_modules(self.model, cfg)
        ema_compile_log = []
        if self.ema_model is not None:
            ema_compile_log = _optimize_policy_modules(self.ema_model, cfg)
        optimizer_to(self.optimizer, device)
        precision_runtime = _build_precision_runtime(cfg, device)
        cprint(
            "[Precision] "
            f"tf32={precision_runtime['tf32_enabled']} "
            f"cudnn_benchmark={precision_runtime['cudnn_benchmark']} "
            f"amp={precision_runtime['amp_enabled']} "
            f"dtype={precision_runtime['amp_dtype_name']}",
            "yellow",
        )
        if model_compile_log:
            cprint("[Compile][model] " + ", ".join(model_compile_log), "yellow")
        if ema_compile_log:
            cprint("[Compile][ema] " + ", ".join(ema_compile_log), "yellow")

        if hasattr(self.model, "debug_compare_global_condition"):
            debug_batch = next(iter(train_dataloader))
            debug_obs_cpu = debug_batch["obs"]
            print("obs keys:", sorted(debug_obs_cpu.keys()))
            if "point_cloud" in debug_obs_cpu:
                print('batch["obs"]["point_cloud"].shape =', debug_obs_cpu["point_cloud"].shape)
            low_dim_keys = [
                key for key, meta in cfg.task.shape_meta.obs.items()
                if meta.get("type") == "low_dim" and key in debug_obs_cpu
            ]
            for key in low_dim_keys:
                print(f'batch["obs"]["{key}"].shape =', debug_obs_cpu[key].shape)
            if "cspace_feature" in debug_obs_cpu:
                print('batch["obs"]["cspace_feature"].shape =', debug_obs_cpu["cspace_feature"].shape)
                print('batch["obs"]["cspace_feature"].mean() =', debug_obs_cpu["cspace_feature"].mean())
                print('batch["obs"]["cspace_feature"].std() =', debug_obs_cpu["cspace_feature"].std())
                print('batch["obs"]["cspace_feature"].min() =', debug_obs_cpu["cspace_feature"].min())
                print('batch["obs"]["cspace_feature"].max() =', debug_obs_cpu["cspace_feature"].max())
            debug_obs = dict_apply(
                debug_obs_cpu,
                lambda x: x.to(device, non_blocking=True),
            )
            self.model.eval()
            with torch.no_grad():
                self.model.debug_compare_global_condition(debug_obs)
            self.model.train()

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        gap_cfg = OmegaConf.select(cfg, "checkpoint.generalization_gap", default={})
        gap_enabled = bool(gap_cfg.get("enabled", False))
        gap_target = float(gap_cfg.get("target", 0.2))
        gap_tolerance = float(gap_cfg.get("tolerance", 0.02))
        gap_window = int(gap_cfg.get("window", 3))
        gap_dirname = str(gap_cfg.get("dirname", "generalization_gap_checkpoints"))
        gap_history = []
        saved_gap_source_epochs = set()

        early_stop_cfg = OmegaConf.select(cfg, "training.early_stop", default={}) or {}
        early_stop_enabled = bool(early_stop_cfg.get("enabled", False))
        early_stop_monitor_key = str(early_stop_cfg.get("monitor_key", "val_loss"))
        early_stop_mode = str(early_stop_cfg.get("mode", "min"))
        early_stop_patience = int(early_stop_cfg.get("patience", 50))
        early_stop_min_delta = float(early_stop_cfg.get("min_delta", 0.0))
        early_stop_warmup_epochs = int(early_stop_cfg.get("warmup_epochs", 0))
        if early_stop_mode not in ("min", "max"):
            raise ValueError(f"training.early_stop.mode must be `min` or `max`, got {early_stop_mode}")
        if early_stop_patience <= 0:
            raise ValueError(f"training.early_stop.patience must be positive, got {early_stop_patience}")
        best_early_stop_value = None
        early_stop_bad_epochs = 0
        pybullet_eval_cfg = PyBulletValidationConfig.from_omegaconf(
            OmegaConf.select(cfg, "training.pybullet_eval", default={}) or {}
        )
        pybullet_validation_runner = None
        if pybullet_eval_cfg.enabled:
            if not isinstance(dataset, TransitionTrajectoryDataset):
                raise TypeError(
                    "training.pybullet_eval currently supports TransitionTrajectoryDataset only."
                )
            pybullet_validation_runner = PyBulletValidationRunner(pybullet_eval_cfg)

        try:
            for local_epoch_idx in range(cfg.training.num_epochs):
                stop_training = False
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        t1 = time.time()
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                    
                        # compute loss
                        t1_1 = time.time()
                        with _autocast_context(precision_runtime, device):
                            raw_loss, loss_dict = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        scaler = precision_runtime["scaler"]
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        
                        t1_2 = time.time()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            if scaler is not None:
                                scaler.step(self.optimizer)
                                scaler.update()
                            else:
                                self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            lr_scheduler.step()
                        t1_3 = time.time()
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)
                        t1_4 = time.time()
                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        t1_5 = time.time()
                        step_log.update(loss_dict)
                        t2 = time.time()
                        
                        if verbose:
                            print(f"total one step time: {t2-t1:.3f}")
                            print(f" compute loss time: {t1_2-t1_1:.3f}")
                            print(f" step optimizer time: {t1_3-t1_2:.3f}")
                            print(f" update ema time: {t1_4-t1_3:.3f}")
                            print(f" logging time: {t1_5-t1_4:.3f}")

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                    t3 = time.time()
                    # runner_log = env_runner.run(policy, dataset=dataset)
                    runner_log = env_runner.run(policy)
                    t4 = time.time()
                    # print(f"rollout time: {t4-t3:.3f}")
                    # log all
                    step_log.update(runner_log)

            
                
                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                with _autocast_context(precision_runtime, device):
                                    loss, loss_dict = policy.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                    if pybullet_validation_runner is not None:
                        step_log.update(
                            pybullet_validation_runner.run(
                                policy=policy,
                                replay_buffer=val_dataset.replay_buffer,
                                episode_mask=val_dataset.train_mask,
                                obs_keys=val_dataset.obs_keys,
                                n_obs_steps=cfg.n_obs_steps,
                                device=device,
                                dataset=val_dataset,
                            )
                        )

                if gap_enabled and ('val_loss' in step_log) and train_loss != 0:
                    gap_value = (step_log['val_loss'] - train_loss) / train_loss
                    step_log['generalization_gap'] = gap_value

                    gap_root = pathlib.Path(self.output_dir).joinpath(gap_dirname)
                    candidate_dir = gap_root.joinpath(".window_candidates")
                    candidate_dir.mkdir(parents=True, exist_ok=True)
                    candidate_path = candidate_dir.joinpath(
                        f"epoch={self.epoch:04d}-val_loss={step_log['val_loss']:.6f}"
                        f"-gap={gap_value:.6f}.ckpt"
                    )
                    self.save_checkpoint(path=candidate_path)

                    gap_history.append({
                        "epoch": self.epoch,
                        "gap": gap_value,
                        "val_loss": step_log['val_loss'],
                        "path": candidate_path,
                    })
                    if len(gap_history) > gap_window:
                        stale = gap_history.pop(0)
                        stale_path = stale["path"]
                        if stale_path.exists():
                            stale_path.unlink()

                    if len(gap_history) == gap_window:
                        close_to_target = [
                            abs(item["gap"] - gap_target) <= gap_tolerance
                            for item in gap_history
                        ]
                        if all(close_to_target):
                            best_item = min(gap_history, key=lambda item: item["val_loss"])
                            if best_item["epoch"] not in saved_gap_source_epochs:
                                final_path = gap_root.joinpath(
                                    f"gap_window_best-epoch={best_item['epoch']:04d}"
                                    f"-window_end={self.epoch:04d}"
                                    f"-val_loss={best_item['val_loss']:.6f}"
                                    f"-gap={best_item['gap']:.6f}.ckpt"
                                )
                                final_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(best_item["path"], final_path)
                                saved_gap_source_epochs.add(best_item["epoch"])
                                step_log['generalization_gap_ckpt'] = str(final_path)

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']

                        with _autocast_context(precision_runtime, device):
                            result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if env_runner is None:
                    step_log['test_mean_score'] = - train_loss
                
                # checkpoint
                if cfg.checkpoint.save_ckpt:
                    if cfg.checkpoint.save_last_ckpt:
                        if (self.epoch % cfg.training.checkpoint_every) == 0:
                            self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        if (self.epoch % cfg.training.checkpoint_every) == 0:
                            self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    monitor_key = cfg.checkpoint.topk.monitor_key
                    if monitor_key in metric_dict:
                        # Save best validation checkpoints as soon as the monitored metric is logged,
                        # instead of waiting for checkpoint_every and missing the true best epoch.
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                if early_stop_enabled and early_stop_monitor_key in step_log \
                        and self.epoch >= early_stop_warmup_epochs:
                    current_value = float(step_log[early_stop_monitor_key])
                    if _is_better_metric(
                            current_value,
                            best_early_stop_value,
                            early_stop_mode,
                            early_stop_min_delta):
                        best_early_stop_value = current_value
                        early_stop_bad_epochs = 0
                    else:
                        early_stop_bad_epochs += 1
                    step_log["early_stop_best_value"] = best_early_stop_value
                    step_log["early_stop_bad_epochs"] = early_stop_bad_epochs
                    if early_stop_bad_epochs >= early_stop_patience:
                        step_log["early_stop"] = True
                        stop_training = True

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(_json_safe(step_log), ensure_ascii=True) + '\n')
                    f.flush()
                self.global_step += 1
                self.epoch += 1
                del step_log
                if stop_training:
                    cprint(
                        f"Early stopping triggered after {early_stop_bad_epochs} epochs "
                        f"without {early_stop_monitor_key} improvement.",
                        "yellow",
                    )
                    break
        finally:
            if pybullet_validation_runner is not None:
                pybullet_validation_runner.close()

    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
