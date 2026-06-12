from typing import Dict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D as SimpleConditionalUnet1D
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.policy.simple_dp3 import SimpleDP3


class CSpaceEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, out_dim=64):
        super().__init__()

        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.point_mlp(x)
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1).values
        x = torch.cat([x_mean, x_max], dim=-1)
        return self.proj(x)


class DP3CSpace(DP3):
    @staticmethod
    def _validate_temporal_downsampling_compatibility(horizon, down_dims):
        horizon = int(horizon)
        num_downsamples = max(len(tuple(down_dims)) - 1, 0)
        required_multiple = 2 ** num_downsamples
        if required_multiple > 1 and (horizon % required_multiple) != 0:
            raise ValueError(
                "DP3CSpace requires a horizon compatible with the temporal U-Net "
                f"downsampling stack. Got horizon={horizon}, down_dims={tuple(down_dims)}; "
                f"horizon must be divisible by {required_multiple}."
            )

    def __init__(
            self,
            shape_meta: dict,
            noise_scheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            encoder_dropout_prob=0.0,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            action_loss_mask_indices=None,
            cspace_feature_key="cspace_feature",
            cspace_in_dim=2,
            cspace_hidden_dim=64,
            cspace_output_dim=64,
            fusion_output_dim=256,
            **kwargs):
        if not obs_as_global_cond:
            raise ValueError("DP3CSpace requires obs_as_global_cond=true.")
        if int(n_obs_steps) != 1:
            raise ValueError(
                f"DP3CSpace requires n_obs_steps=1, got {n_obs_steps}."
            )
        if int(n_action_steps) > int(horizon):
            raise ValueError(
                f"n_action_steps ({n_action_steps}) cannot exceed horizon ({horizon})."
            )
        self._validate_temporal_downsampling_compatibility(horizon, down_dims)
        if condition_type != "film":
            raise ValueError(
                f"DP3CSpace currently supports condition_type='film', got {condition_type!r}."
            )

        base_shape_meta = copy.deepcopy(shape_meta)
        obs_shape_meta = base_shape_meta.get("obs", {})
        if cspace_feature_key not in obs_shape_meta:
            raise KeyError(
                f"shape_meta.obs is missing C-space key {cspace_feature_key!r}."
            )
        cspace_shape = tuple(obs_shape_meta.pop(cspace_feature_key)["shape"])
        if cspace_shape != (128, int(cspace_in_dim)):
            raise ValueError(
                f"C-space shape must be [128, {cspace_in_dim}], got {cspace_shape}."
            )

        super().__init__(
            shape_meta=base_shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
            encoder_output_dim=encoder_output_dim,
            encoder_dropout_prob=encoder_dropout_prob,
            crop_shape=crop_shape,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            action_loss_mask_indices=action_loss_mask_indices,
            **kwargs,
        )

        self.cspace_feature_key = cspace_feature_key
        self.cspace_in_dim = int(cspace_in_dim)
        self.cspace_output_dim = int(cspace_output_dim)
        self.fusion_output_dim = int(fusion_output_dim)
        self.cspace_encoder = CSpaceEncoder(
            in_dim=self.cspace_in_dim,
            hidden_dim=int(cspace_hidden_dim),
            out_dim=self.cspace_output_dim,
        )
        fusion_input_dim = int(self.obs_feature_dim + self.cspace_output_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, self.fusion_output_dim),
            nn.LayerNorm(self.fusion_output_dim),
            nn.ReLU(),
        )
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=self.fusion_output_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

    def _validate_cspace_feature(
            self,
            cspace_feature: torch.Tensor,
            batch_size: int) -> torch.Tensor:
        expected_shape = (batch_size, 128, self.cspace_in_dim)
        if tuple(cspace_feature.shape) != expected_shape:
            raise ValueError(
                f"{self.cspace_feature_key} must have shape {expected_shape}, "
                f"got {tuple(cspace_feature.shape)}."
            )
        if not torch.is_floating_point(cspace_feature):
            raise TypeError(
                f"{self.cspace_feature_key} must be floating point, "
                f"got {cspace_feature.dtype}."
            )
        if not torch.all(torch.isfinite(cspace_feature)):
            raise ValueError(f"{self.cspace_feature_key} contains non-finite values.")
        return cspace_feature

    def _encode_global_condition(
            self,
            normalized_obs: Dict[str, torch.Tensor],
            cspace_feature: torch.Tensor) -> torch.Tensor:
        first_value = next(iter(normalized_obs.values()))
        batch_size = int(first_value.shape[0])
        one_step_obs = dict_apply(
            normalized_obs,
            lambda value: value[:, :1, ...].reshape(
                batch_size, *value.shape[2:]
            ),
        )
        obs_feature = self.obs_encoder(one_step_obs)
        cspace_feature = self._validate_cspace_feature(
            cspace_feature, batch_size=batch_size
        )
        z_space = self.cspace_encoder(cspace_feature)
        fused_feature = torch.cat([obs_feature, z_space], dim=-1)
        return self.fusion_mlp(fused_feature)

    def _split_observations(self, obs_dict):
        if self.cspace_feature_key not in obs_dict:
            raise KeyError(
                f"Observation dictionary is missing {self.cspace_feature_key!r}."
            )
        cspace_feature = obs_dict[self.cspace_feature_key]
        base_obs = {
            key: value
            for key, value in obs_dict.items()
            if key != self.cspace_feature_key
        }
        normalized_obs = self.normalizer.normalize(base_obs)
        if not self.use_pc_color:
            normalized_obs["point_cloud"] = normalized_obs["point_cloud"][..., :3]
        return normalized_obs, cspace_feature

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized_obs, cspace_feature = self._split_observations(obs_dict)
        first_value = next(iter(normalized_obs.values()))
        batch_size = int(first_value.shape[0])
        global_cond = self._encode_global_condition(
            normalized_obs, cspace_feature
        )

        condition_data = torch.zeros(
            size=(batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        normalized_sample = self.conditional_sample(
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs,
        )

        normalized_action = normalized_sample[..., :self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(normalized_action)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {
            "action": action,
            "action_pred": action_pred,
        }

    def compute_loss(self, batch):
        normalized_obs, cspace_feature = self._split_observations(batch["obs"])
        normalized_actions = self.normalizer["action"].normalize(batch["action"])
        global_cond = self._encode_global_condition(
            normalized_obs, cspace_feature
        )

        trajectory = normalized_actions
        condition_data = trajectory
        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )
        loss_mask = self._apply_action_loss_mask(~condition_mask)
        noisy_trajectory[condition_mask] = condition_data[condition_mask]

        prediction = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond,
        )
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = trajectory
        elif prediction_type == "v_prediction":
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(
                self.device
            )
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(
                self.device
            )
            alpha_t = self.noise_scheduler.alpha_t[timesteps]
            sigma_t = self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            target = alpha_t * noise - sigma_t * trajectory
        else:
            raise ValueError(
                f"Unsupported prediction type {prediction_type}"
            )

        loss = F.mse_loss(prediction, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean").mean()
        return loss, {"bc_loss": loss.item()}


class SimpleDP3CSpace(SimpleDP3):
    def __init__(
            self,
            shape_meta: dict,
            noise_scheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            encoder_dropout_prob=0.0,
            state_mlp_size=(64, 64),
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            action_loss_mask_indices=None,
            cspace_feature_key="cspace_feature",
            cspace_in_dim=2,
            cspace_hidden_dim=64,
            cspace_output_dim=64,
            fusion_output_dim=256,
            **kwargs):
        if not obs_as_global_cond:
            raise ValueError("SimpleDP3CSpace requires obs_as_global_cond=true.")
        if int(n_obs_steps) != 1:
            raise ValueError(
                f"SimpleDP3CSpace requires n_obs_steps=1, got {n_obs_steps}."
            )
        if int(n_action_steps) > int(horizon):
            raise ValueError(
                f"n_action_steps ({n_action_steps}) cannot exceed horizon ({horizon})."
            )
        if condition_type != "film":
            raise ValueError(
                "SimpleDP3CSpace currently supports condition_type='film', "
                f"got {condition_type!r}."
            )

        base_shape_meta = copy.deepcopy(shape_meta)
        obs_shape_meta = base_shape_meta.get("obs", {})
        if cspace_feature_key not in obs_shape_meta:
            raise KeyError(
                f"shape_meta.obs is missing C-space key {cspace_feature_key!r}."
            )
        cspace_shape = tuple(obs_shape_meta.pop(cspace_feature_key)["shape"])
        if cspace_shape != (128, int(cspace_in_dim)):
            raise ValueError(
                f"C-space shape must be [128, {cspace_in_dim}], got {cspace_shape}."
            )

        super().__init__(
            shape_meta=base_shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
            encoder_output_dim=encoder_output_dim,
            encoder_dropout_prob=encoder_dropout_prob,
            state_mlp_size=state_mlp_size,
            crop_shape=crop_shape,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            action_loss_mask_indices=action_loss_mask_indices,
            **kwargs,
        )

        self.cspace_feature_key = cspace_feature_key
        self.cspace_in_dim = int(cspace_in_dim)
        self.cspace_output_dim = int(cspace_output_dim)
        self.fusion_output_dim = int(fusion_output_dim)
        self.cspace_encoder = CSpaceEncoder(
            in_dim=self.cspace_in_dim,
            hidden_dim=int(cspace_hidden_dim),
            out_dim=self.cspace_output_dim,
        )
        fusion_input_dim = int(self.obs_feature_dim + self.cspace_output_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, self.fusion_output_dim),
            nn.LayerNorm(self.fusion_output_dim),
            nn.ReLU(),
        )
        self.model = SimpleConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=self.fusion_output_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

    def _validate_cspace_feature(
            self,
            cspace_feature: torch.Tensor,
            batch_size: int) -> torch.Tensor:
        expected_shape = (batch_size, 128, self.cspace_in_dim)
        if tuple(cspace_feature.shape) != expected_shape:
            raise ValueError(
                f"{self.cspace_feature_key} must have shape {expected_shape}, "
                f"got {tuple(cspace_feature.shape)}."
            )
        if not torch.is_floating_point(cspace_feature):
            raise TypeError(
                f"{self.cspace_feature_key} must be floating point, "
                f"got {cspace_feature.dtype}."
            )
        if not torch.all(torch.isfinite(cspace_feature)):
            raise ValueError(f"{self.cspace_feature_key} contains non-finite values.")
        return cspace_feature

    def _encode_global_condition(
            self,
            normalized_obs: Dict[str, torch.Tensor],
            cspace_feature: torch.Tensor) -> torch.Tensor:
        first_value = next(iter(normalized_obs.values()))
        batch_size = int(first_value.shape[0])
        one_step_obs = dict_apply(
            normalized_obs,
            lambda value: value[:, :1, ...].reshape(
                batch_size, *value.shape[2:]
            ),
        )
        obs_feature = self.obs_encoder(one_step_obs)
        cspace_feature = self._validate_cspace_feature(
            cspace_feature, batch_size=batch_size
        )
        z_space = self.cspace_encoder(cspace_feature)
        fused_feature = torch.cat([obs_feature, z_space], dim=-1)
        return self.fusion_mlp(fused_feature)

    def _split_observations(self, obs_dict):
        if self.cspace_feature_key not in obs_dict:
            raise KeyError(
                f"Observation dictionary is missing {self.cspace_feature_key!r}."
            )
        cspace_feature = obs_dict[self.cspace_feature_key]
        base_obs = {
            key: value
            for key, value in obs_dict.items()
            if key != self.cspace_feature_key
        }
        normalized_obs = self.normalizer.normalize(base_obs)
        if not self.use_pc_color:
            normalized_obs["point_cloud"] = normalized_obs["point_cloud"][..., :3]
        return normalized_obs, cspace_feature

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized_obs, cspace_feature = self._split_observations(obs_dict)
        first_value = next(iter(normalized_obs.values()))
        batch_size = int(first_value.shape[0])
        global_cond = self._encode_global_condition(
            normalized_obs, cspace_feature
        )

        condition_data = torch.zeros(
            size=(batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        normalized_sample = self.conditional_sample(
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs,
        )

        normalized_action = normalized_sample[..., :self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(normalized_action)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {
            "action": action,
            "action_pred": action_pred,
        }

    def compute_loss(self, batch):
        normalized_obs, cspace_feature = self._split_observations(batch["obs"])
        normalized_actions = self.normalizer["action"].normalize(batch["action"])
        global_cond = self._encode_global_condition(
            normalized_obs, cspace_feature
        )

        trajectory = normalized_actions
        condition_data = trajectory
        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )
        loss_mask = self._apply_action_loss_mask(~condition_mask)
        noisy_trajectory[condition_mask] = condition_data[condition_mask]

        prediction = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond,
        )
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = trajectory
        elif prediction_type == "v_prediction":
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(
                self.device
            )
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(
                self.device
            )
            alpha_t = self.noise_scheduler.alpha_t[timesteps]
            sigma_t = self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            target = alpha_t * noise - sigma_t * trajectory
        else:
            raise ValueError(
                f"Unsupported prediction type {prediction_type}"
            )

        loss = F.mse_loss(prediction, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean").mean()
        return loss, {"bc_loss": loss.item()}
