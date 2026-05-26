from typing import Dict
import copy

import numpy as np
import torch

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


def create_identity_normalizer_for_shape(shape, dtype=torch.float32):
    flat_dim = int(np.prod(shape))
    scale = torch.ones(flat_dim, dtype=dtype)
    offset = torch.zeros(flat_dim, dtype=dtype)
    input_stats_dict = {
        'min': torch.full((flat_dim,), -1.0, dtype=dtype),
        'max': torch.full((flat_dim,), 1.0, dtype=dtype),
        'mean': torch.zeros(flat_dim, dtype=dtype),
        'std': torch.ones(flat_dim, dtype=dtype),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=input_stats_dict,
    )


class TransitionTrajectoryDataset(BaseDataset):
    DEFAULT_OBS_KEYS = (
        'point_cloud',
        'goal_position',
        'goal_direction',
        'first_joint_angles_normalized',
        'last_joint_angles_normalized',
    )

    def __init__(self,
            zarr_path,
            horizon=64,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            point_cloud_key='point_cloud',
            obs_keys=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.point_cloud_key = point_cloud_key
        self.obs_keys = tuple(obs_keys) if obs_keys is not None else self.DEFAULT_OBS_KEYS

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

    def get_normalizer(self, **kwargs):
        normalizer = LinearNormalizer()
        normalizer['action'] = create_identity_normalizer_for_shape((6,))
        for key in self.obs_keys:
            sample_array = self.replay_buffer[key]
            normalizer[key] = create_identity_normalizer_for_shape(sample_array.shape[1:])
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = dict()
        for key in self.obs_keys:
            obs[key] = sample[key][:].astype(np.float32)

        data = {
            'obs': obs,
            'action': sample['action'].astype(np.float32)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
