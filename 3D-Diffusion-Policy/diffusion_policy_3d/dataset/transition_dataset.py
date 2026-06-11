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

    @staticmethod
    def _resolve_workpiece_split_masks(
            replay_buffer: ReplayBuffer,
            train_workpiece_ids=None,
            val_workpiece_ids=None):
        if train_workpiece_ids is None and val_workpiece_ids is None:
            return None
        if 'workpiece_ids' not in replay_buffer.meta:
            raise KeyError(
                "Dataset split by workpiece was requested, but `meta/workpiece_ids` "
                "is missing from the zarr dataset. Rebuild the dataset with workpiece metadata."
            )

        episode_workpiece_ids = np.asarray(replay_buffer.meta['workpiece_ids'][:], dtype=np.int64)
        if episode_workpiece_ids.shape != (replay_buffer.n_episodes,):
            raise ValueError(
                f"`meta/workpiece_ids` must have shape ({replay_buffer.n_episodes},), "
                f"got {episode_workpiece_ids.shape}"
            )

        train_mask = np.zeros_like(episode_workpiece_ids, dtype=bool)
        val_mask = np.zeros_like(episode_workpiece_ids, dtype=bool)

        if train_workpiece_ids is not None:
            train_ids = np.asarray(train_workpiece_ids, dtype=np.int64).reshape(-1)
            train_mask = np.isin(episode_workpiece_ids, train_ids)
        if val_workpiece_ids is not None:
            val_ids = np.asarray(val_workpiece_ids, dtype=np.int64).reshape(-1)
            val_mask = np.isin(episode_workpiece_ids, val_ids)

        if train_workpiece_ids is None:
            train_mask = ~val_mask
        if val_workpiece_ids is None:
            val_mask = ~train_mask

        if np.any(train_mask & val_mask):
            overlapping = np.unique(episode_workpiece_ids[train_mask & val_mask])
            raise ValueError(
                f"Train/validation workpiece ids overlap in split configuration: {overlapping.tolist()}"
            )
        if not np.any(train_mask):
            raise ValueError("No training episodes remain after applying workpiece split.")
        if not np.any(val_mask):
            raise ValueError("No validation episodes remain after applying workpiece split.")

        return train_mask.astype(bool), val_mask.astype(bool)

    @staticmethod
    def _resolve_ratio_split_masks(
            replay_buffer: ReplayBuffer,
            val_ratio: float,
            seed: int,
            split_by_workpiece: bool,
            stratify_workpiece_split: bool,
            simple_workpiece_id_offset: int,
            workpiece_split_strategy: str):
        if not split_by_workpiece:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed)
            return ~val_mask, val_mask

        if 'workpiece_ids' not in replay_buffer.meta:
            raise KeyError(
                "Dataset split by workpiece was requested, but `meta/workpiece_ids` "
                "is missing from the zarr dataset. Rebuild the dataset with workpiece metadata."
            )
        episode_workpiece_ids = np.asarray(replay_buffer.meta['workpiece_ids'][:], dtype=np.int64)
        if episode_workpiece_ids.shape != (replay_buffer.n_episodes,):
            raise ValueError(
                f"`meta/workpiece_ids` must have shape ({replay_buffer.n_episodes},), "
                f"got {episode_workpiece_ids.shape}"
            )

        workpiece_ids = np.unique(episode_workpiece_ids)
        if workpiece_split_strategy not in ('random', 'tail'):
            raise ValueError(
                f"workpiece_split_strategy must be `random` or `tail`, got {workpiece_split_strategy}"
            )

        def select_val_workpieces(group):
            n_val = min(max(1, round(len(group) * val_ratio)), len(group) - 1)
            if n_val <= 0:
                return np.asarray([], dtype=group.dtype)
            if workpiece_split_strategy == 'tail':
                return group[-n_val:]
            rng = np.random.default_rng(seed=seed)
            return rng.choice(group, size=n_val, replace=False)

        if stratify_workpiece_split:
            val_workpiece_ids = []
            groups = (
                workpiece_ids[workpiece_ids < simple_workpiece_id_offset],
                workpiece_ids[workpiece_ids >= simple_workpiece_id_offset],
            )
            for group in groups:
                if len(group) == 0:
                    continue
                val_workpiece_ids.append(select_val_workpieces(group))
            if len(val_workpiece_ids) == 0:
                val_workpiece_ids = np.asarray([], dtype=workpiece_ids.dtype)
            else:
                val_workpiece_ids = np.concatenate(val_workpiece_ids)
        else:
            val_workpiece_ids = select_val_workpieces(workpiece_ids)
        val_mask = np.isin(episode_workpiece_ids, val_workpiece_ids)
        train_mask = ~val_mask
        return train_mask.astype(bool), val_mask.astype(bool)

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
            train_workpiece_ids=None,
            val_workpiece_ids=None,
            split_by_workpiece=False,
            stratify_workpiece_split=False,
            simple_workpiece_id_offset=1000,
            workpiece_split_strategy='random',
            simple_as_train_validation_split=False,
            ):
        super().__init__()
        self.task_name = task_name
        self.point_cloud_key = point_cloud_key
        self.obs_keys = tuple(obs_keys) if obs_keys is not None else self.DEFAULT_OBS_KEYS

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=None)
        if simple_as_train_validation_split:
            if 'workpiece_ids' not in self.replay_buffer.meta:
                raise KeyError(
                    'simple_as_train_validation_split=True requires `meta/workpiece_ids` '
                    'in the zarr dataset. Rebuild the dataset with workpiece metadata.'
                )
            episode_workpiece_ids = np.asarray(self.replay_buffer.meta['workpiece_ids'][:], dtype=np.int64)
            if episode_workpiece_ids.shape != (self.replay_buffer.n_episodes,):
                raise ValueError(
                    f'`meta/workpiece_ids` must have shape ({self.replay_buffer.n_episodes},), '
                    f'got {episode_workpiece_ids.shape}'
                )
            unique_ids = np.unique(episode_workpiece_ids)
            train_workpiece_ids = unique_ids[unique_ids >= int(simple_workpiece_id_offset)]
            val_workpiece_ids = unique_ids[unique_ids < int(simple_workpiece_id_offset)]
            if len(train_workpiece_ids) == 0:
                raise ValueError(
                    'simple_as_train_validation_split=True found no simple workpiece ids '
                    f'(>= {simple_workpiece_id_offset}) for training.'
                )
            if len(val_workpiece_ids) == 0:
                raise ValueError(
                    'simple_as_train_validation_split=True found no complex workpiece ids '
                    f'(< {simple_workpiece_id_offset}) for validation.'
                )
        split_masks = self._resolve_workpiece_split_masks(
            replay_buffer=self.replay_buffer,
            train_workpiece_ids=train_workpiece_ids,
            val_workpiece_ids=val_workpiece_ids,
        )
        if split_masks is None:
            train_mask, val_mask = self._resolve_ratio_split_masks(
                replay_buffer=self.replay_buffer,
                val_ratio=val_ratio,
                seed=seed,
                split_by_workpiece=split_by_workpiece,
                stratify_workpiece_split=stratify_workpiece_split,
                simple_workpiece_id_offset=simple_workpiece_id_offset,
                workpiece_split_strategy=workpiece_split_strategy)
        else:
            train_mask, val_mask = split_masks
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
        if len(self.sampler) == 0:
            train_lengths = self.replay_buffer.episode_lengths[train_mask]
            raise ValueError(
                "TransitionTrajectoryDataset produced zero training windows. "
                f"horizon={horizon}, pad_before={pad_before}, pad_after={pad_after}, "
                f"selected_train_episodes={int(np.sum(train_mask))}, "
                f"train_episode_lengths={train_lengths.tolist()}. "
                "This usually means the selected episodes are shorter than the configured horizon, "
                "or the split removed all usable episodes."
            )
        self.train_mask = train_mask
        self.val_mask = val_mask
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
            episode_mask=self.val_mask
        )
        if len(val_set.sampler) == 0:
            val_lengths = self.replay_buffer.episode_lengths[self.val_mask]
            raise ValueError(
                "TransitionTrajectoryDataset produced zero validation windows. "
                f"horizon={self.horizon}, pad_before={self.pad_before}, pad_after={self.pad_after}, "
                f"selected_val_episodes={int(np.sum(self.val_mask))}, "
                f"val_episode_lengths={val_lengths.tolist()}. "
                "Reduce horizon, change the split, or rebuild the dataset with longer episodes."
            )
        val_set.train_mask = self.val_mask
        val_set.val_mask = self.val_mask
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
