from typing import Dict
from pathlib import Path

import numpy as np
import torch

from diffusion_policy_3d.dataset.transition_dataset import TransitionTrajectoryDataset


class TransitionTrajectoryCSpaceDataset(TransitionTrajectoryDataset):
    def __init__(
            self,
            cspace_feature_dir,
            cspace_feature_filename="workpiece_key_config_features.npy",
            cspace_workpiece_ids_filename="workpiece_ids.npy",
            **kwargs):
        super().__init__(**kwargs)

        feature_dir = self._resolve_feature_dir(cspace_feature_dir)
        feature_path = feature_dir / cspace_feature_filename
        workpiece_ids_path = feature_dir / cspace_workpiece_ids_filename
        missing_paths = [
            str(path)
            for path in (feature_path, workpiece_ids_path)
            if not path.is_file()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"Missing C-space feature artifacts: {missing_paths}"
            )

        features = np.asarray(np.load(feature_path), dtype=np.float32)
        workpiece_ids = np.asarray(
            np.load(workpiece_ids_path), dtype=np.int64
        ).reshape(-1)

        if features.ndim != 3 or features.shape[1:] != (128, 2):
            raise ValueError(
                "C-space features must have shape [N, 128, 2], "
                f"got {features.shape} from {feature_path}"
            )
        if workpiece_ids.shape != (features.shape[0],):
            raise ValueError(
                "C-space workpiece IDs must have shape "
                f"({features.shape[0]},), got {workpiece_ids.shape}"
            )
        if not np.all(np.isfinite(features)):
            invalid_count = int(np.size(features) - np.count_nonzero(np.isfinite(features)))
            raise ValueError(
                f"C-space features contain {invalid_count} non-finite values."
            )

        unique_ids, counts = np.unique(workpiece_ids, return_counts=True)
        duplicate_ids = unique_ids[counts > 1]
        if duplicate_ids.size > 0:
            raise ValueError(
                "C-space workpiece IDs must be unique; duplicates: "
                f"{duplicate_ids.tolist()}"
            )
        if "workpiece_ids" not in self.replay_buffer.meta:
            raise KeyError(
                "C-space dataset requires `meta/workpiece_ids` in the zarr dataset."
            )

        episode_workpiece_ids = np.asarray(
            self.replay_buffer.meta["workpiece_ids"][:], dtype=np.int64
        )
        if episode_workpiece_ids.shape != (self.replay_buffer.n_episodes,):
            raise ValueError(
                "`meta/workpiece_ids` must have shape "
                f"({self.replay_buffer.n_episodes},), got {episode_workpiece_ids.shape}"
            )
        missing_ids = np.setdiff1d(np.unique(episode_workpiece_ids), unique_ids)
        if missing_ids.size > 0:
            raise KeyError(
                "C-space features are missing zarr workpiece IDs: "
                f"{missing_ids.tolist()}"
            )

        self.cspace_features = np.ascontiguousarray(features)
        self.cspace_workpiece_ids = workpiece_ids
        self.cspace_row_by_workpiece_id = {
            int(workpiece_id): int(row_index)
            for row_index, workpiece_id in enumerate(workpiece_ids)
        }
        self.episode_workpiece_ids = episode_workpiece_ids
        self.episode_ends = np.asarray(
            self.replay_buffer.episode_ends[:], dtype=np.int64
        )
        self.cspace_feature_dir = str(feature_dir)

    @staticmethod
    def _resolve_feature_dir(path):
        return Path(path).expanduser().resolve()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = super().__getitem__(idx)
        buffer_start_idx = int(self.sampler.indices[idx, 0])
        episode_idx = int(
            np.searchsorted(self.episode_ends, buffer_start_idx, side="right")
        )
        if episode_idx >= self.replay_buffer.n_episodes:
            raise IndexError(
                f"Unable to resolve episode for replay-buffer index {buffer_start_idx}."
            )

        workpiece_id = int(self.episode_workpiece_ids[episode_idx])
        feature_row = self.cspace_row_by_workpiece_id[workpiece_id]
        data["obs"]["cspace_feature"] = torch.from_numpy(
            self.cspace_features[feature_row]
        )
        return data
