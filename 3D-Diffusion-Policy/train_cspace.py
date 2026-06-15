if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pathlib

import hydra
from omegaconf import open_dict

from train import TrainDP3Workspace


class TrainDP3CSpaceWorkspace(TrainDP3Workspace):
    exclude_keys = tuple(TrainDP3Workspace.exclude_keys) + ("_raw_model",)

    def load_payload(
        self,
        payload,
        exclude_keys=None,
        include_keys=None,
        **kwargs,
    ):
        effective_exclude_keys = set(exclude_keys or ())
        for key in payload.get("state_dicts", {}):
            if key not in self.__dict__:
                effective_exclude_keys.add(key)
                print(
                    f"Skipping checkpoint state `{key}` because it is not "
                    "present in the current C-space workspace."
                )
        return super().load_payload(
            payload=payload,
            exclude_keys=tuple(effective_exclude_keys),
            include_keys=include_keys,
            **kwargs,
        )


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.joinpath("diffusion_policy_3d", "config")
    ),
    config_name="dp3_cspace",
)
def main(cfg):
    with open_dict(cfg):
        cfg.checkpoint.topk.monitor_key = "val_pybullet_collision_rate"
        cfg.checkpoint.topk.mode = "min"
        cfg.checkpoint.topk.k = 3
        cfg.checkpoint.topk.format_str = (
            "epoch={epoch:04d}-val_pybullet_collision_rate="
            "{val_pybullet_collision_rate:.6f}.ckpt"
        )
    workspace = TrainDP3CSpaceWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
