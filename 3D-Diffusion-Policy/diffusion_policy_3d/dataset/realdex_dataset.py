from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class RealDexDataset(BaseDataset):
    DEFAULT_OBS_KEYS = ('point_cloud', 'start_pos', 'goal_pos', 'start_rot', 'goal_rot')

    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            point_cloud_key='point_cloud',
            obs_keys=None,
            fallback_state_key='state',
            state_slices=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.point_cloud_key = point_cloud_key
        self.obs_keys = tuple(obs_keys) if obs_keys is not None else self.DEFAULT_OBS_KEYS
        self.fallback_state_key = fallback_state_key
        self.state_slices = state_slices if state_slices is not None else dict()

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=None)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            self.point_cloud_key: self.replay_buffer[self.point_cloud_key],
        }
        for key in self.obs_keys:
            if key == self.point_cloud_key:
                continue
            data[key] = self._get_obs_array(key)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _get_obs_array(self, key):
        if key in self.replay_buffer:
            return self.replay_buffer[key][...].astype(np.float32)
        if self.fallback_state_key not in self.replay_buffer:
            raise KeyError(
                f"Observation key '{key}' is missing and fallback state key "
                f"'{self.fallback_state_key}' is not available in replay buffer."
            )
        if key not in self.state_slices:
            raise KeyError(
                f"Observation key '{key}' is missing in replay buffer and no slice "
                f"was provided in state_slices."
            )
        start, end = self.state_slices[key]
        return self.replay_buffer[self.fallback_state_key][..., start:end].astype(np.float32)

    def _sample_to_data(self, sample):
        point_cloud = sample[self.point_cloud_key][:,].astype(np.float32)
        obs = {
            self.point_cloud_key: point_cloud,
        }
        for key in self.obs_keys:
            if key == self.point_cloud_key:
                continue
            if key in sample:
                obs[key] = sample[key][:,].astype(np.float32)
            else:
                if key not in self.state_slices:
                    raise KeyError(
                        f"Sample is missing observation key '{key}' and no fallback slice "
                        f"was configured in state_slices."
                    )
                start, end = self.state_slices[key]
                obs[key] = sample[self.fallback_state_key][:, start:end].astype(np.float32)

        data = {
            'obs': obs,
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
