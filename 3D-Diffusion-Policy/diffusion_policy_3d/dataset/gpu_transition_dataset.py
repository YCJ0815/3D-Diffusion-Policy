from typing import Dict

import numpy as np
import torch
from termcolor import cprint

from diffusion_policy_3d.dataset.transition_dataset import TransitionTrajectoryDataset
from diffusion_policy_3d.dataset.transition_cspace_dataset import TransitionTrajectoryCSpaceDataset


class GPUTransitionTrajectoryDataset(TransitionTrajectoryDataset):
    """GPU-resident dataset that loads all replay buffer data into GPU tensors
    at initialization, eliminating DataLoader CPU overhead entirely.

    Use num_workers=0 with this dataset for best performance.
    """

    def __init__(self, device="cuda:0", **kwargs):
        super().__init__(**kwargs)
        self._gpu_device = torch.device(device)
        self._gpu_data: Dict[str, torch.Tensor] = {}
        self._gpu_loaded = False
        self._preload_to_gpu()

    def _preload_to_gpu(self):
        if self._gpu_loaded:
            return
        device = self._gpu_device
        total_bytes = 0

        for key in self.sampler.keys:
            arr = self.replay_buffer[key]
            tensor = torch.from_numpy(arr[:].astype(np.float32)).to(
                device, non_blocking=True
            )
            self._gpu_data[key] = tensor
            total_bytes += tensor.numel() * tensor.element_size()
            cprint(
                f"[GPU Dataset] Loaded {key}: shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}",
                "green",
            )

        self._gpu_indices = torch.from_numpy(
            self.sampler.indices.astype(np.int64)
        ).to(device, non_blocking=True)

        cprint(
            f"[GPU Dataset] Total GPU memory: {total_bytes / 1024**3:.2f} GB, "
            f"samples={len(self.sampler)}",
            "green",
        )
        self._gpu_loaded = True

    def get_validation_dataset(self):
        val_set = TransitionTrajectoryDataset.get_validation_dataset(self)
        # Update GPU indices for validation split
        val_set._gpu_indices = torch.from_numpy(
            val_set.sampler.indices.astype(np.int64)
        ).to(self._gpu_device, non_blocking=True)
        return val_set

    def _gpu_sample_to_data(self, idx: int) -> Dict[str, torch.Tensor]:
        indices = self._gpu_indices[idx]
        buffer_start = int(indices[0].item())
        buffer_end = int(indices[1].item())
        sample_start = int(indices[2].item())
        sample_end = int(indices[3].item())

        obs = {}
        for key in self.obs_keys:
            gpu_arr = self._gpu_data[key]
            sample = gpu_arr[buffer_start:buffer_end]
            if sample_start > 0 or sample_end < self.horizon:
                padded = torch.zeros(
                    (self.horizon,) + sample.shape[1:],
                    dtype=sample.dtype,
                    device=sample.device,
                )
                padded[sample_start:sample_end] = sample
                if sample_start > 0:
                    padded[:sample_start] = sample[0]
                if sample_end < self.horizon:
                    padded[sample_end:] = sample[-1]
                obs[key] = padded
            else:
                obs[key] = sample

        action = self._gpu_data["action"][buffer_start:buffer_end]
        if sample_start > 0 or sample_end < self.horizon:
            padded_action = torch.zeros(
                (self.horizon,) + action.shape[1:],
                dtype=action.dtype,
                device=action.device,
            )
            padded_action[sample_start:sample_end] = action
            if sample_start > 0:
                padded_action[:sample_start] = action[0]
            if sample_end < self.horizon:
                padded_action[sample_end:] = action[-1]
            action = padded_action

        return {"obs": obs, "action": action}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self._gpu_sample_to_data(idx)
        if self.episode_workpiece_ids is not None:
            buffer_start_idx = int(self.sampler.indices[idx, 0])
            episode_idx = int(
                np.searchsorted(
                    self.episode_ends,
                    buffer_start_idx,
                    side="right",
                )
            )
            if episode_idx >= self.replay_buffer.n_episodes:
                raise IndexError(
                    f"Unable to resolve episode for replay-buffer index {buffer_start_idx}."
                )
            data["workpiece_id"] = torch.tensor(
                int(self.episode_workpiece_ids[episode_idx]),
                dtype=torch.long,
            )
        return data


class GPUTransitionCSpaceDataset(
    GPUTransitionTrajectoryDataset, TransitionTrajectoryCSpaceDataset
):
    """GPU-resident C-space dataset.

    Init order: TransitionTrajectoryCSpaceDataset first (sets up cspace features,
    episode_workpiece_ids, etc.), then GPU preloading from GPUTransitionTrajectoryDataset.
    """

    def __init__(self, device="cuda:0", **kwargs):
        # Step 1: full cspace dataset init (inherits TransitionTrajectoryDataset init)
        TransitionTrajectoryCSpaceDataset.__init__(self, **kwargs)
        # Step 2: GPU preloading
        self._gpu_device = torch.device(device)
        self._gpu_data: Dict[str, torch.Tensor] = {}
        self._gpu_loaded = False
        self._preload_to_gpu()
        # Step 3: cspace features to GPU
        self._gpu_cspace_features = torch.from_numpy(
            np.ascontiguousarray(self.cspace_features)
        ).to(self._gpu_device, non_blocking=True)
        cprint(
            f"[GPU Dataset] Loaded cspace_features: "
            f"shape={tuple(self._gpu_cspace_features.shape)}",
            "green",
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = GPUTransitionTrajectoryDataset.__getitem__(self, idx)
        if "workpiece_id" not in data:
            raise KeyError(
                "TransitionTrajectoryCSpaceDataset requires the base dataset to "
                "provide `workpiece_id`."
            )
        workpiece_id = int(data["workpiece_id"].item())
        feature_row = self.cspace_row_by_workpiece_id[workpiece_id]
        data["obs"]["cspace_feature"] = self._gpu_cspace_features[feature_row]
        return data
