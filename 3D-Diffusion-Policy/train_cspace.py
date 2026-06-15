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
    workspace = TrainDP3Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
