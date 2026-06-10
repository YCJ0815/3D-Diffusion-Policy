import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
from pathlib import Path

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class PretrainedPointNetEncoderXYZ(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 64,
            pretrained_checkpoint_path: Optional[str] = None,
            freeze_pretrained: bool = True,
            unfreeze_last_layer: bool = False,
            **kwargs
            ):
        super().__init__()
        if in_channels != 3:
            raise ValueError(
                f"PretrainedPointNetEncoderXYZ only supports 3-channel XYZ point clouds, got {in_channels}"
            )
        if out_channels != 64:
            raise ValueError(
                f"PretrainedPointNetEncoderXYZ requires out_channels=64 to match the pretrained projection head, got {out_channels}"
            )
        if not pretrained_checkpoint_path:
            raise ValueError("pretrained_checkpoint_path must be provided for pretrained pointnet")

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.projection = nn.Linear(1024, 64)
        self.projection_norm = nn.LayerNorm(64)

        self.pretrained_checkpoint_path = str(pretrained_checkpoint_path)
        self.freeze_pretrained = bool(freeze_pretrained)
        self.unfreeze_last_layer = bool(unfreeze_last_layer)

        self._load_pretrained_weights()
        self._configure_trainable_parameters()

        cprint(
            f"[PretrainedPointNetEncoderXYZ] checkpoint: {self.pretrained_checkpoint_path}",
            "cyan",
        )
        cprint(
            f"[PretrainedPointNetEncoderXYZ] frozen: {self.freeze_pretrained}",
            "cyan",
        )
        cprint(
            f"[PretrainedPointNetEncoderXYZ] unfreeze_last_layer: {self.unfreeze_last_layer}",
            "cyan",
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_pretrained:
            if self.unfreeze_last_layer:
                self.conv1.train(False)
                self.conv2.train(False)
                self.conv3.train(False)
                self.bn1.train(False)
                self.bn2.train(False)
                self.bn3.train(False)
                self.projection.train(mode)
                self.projection_norm.train(mode)
            else:
                super().train(False)
        return self

    def _configure_trainable_parameters(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

        if self.freeze_pretrained:
            if self.unfreeze_last_layer:
                for module in (self.projection, self.projection_norm):
                    for parameter in module.parameters():
                        parameter.requires_grad = True
                self.train(False)
                self.projection.train(True)
                self.projection_norm.train(True)
            else:
                # Keep BatchNorm running stats frozen during outer policy training.
                super().train(False)
        else:
            for parameter in self.parameters():
                parameter.requires_grad = True

    @staticmethod
    def _to_channel_first(point_cloud: torch.Tensor) -> torch.Tensor:
        if point_cloud.ndim != 3:
            raise ValueError(
                "point_cloud must have shape [B, N, 3] or [B, 3, N], "
                f"got {tuple(point_cloud.shape)}"
            )
        if point_cloud.shape[-1] == 3:
            point_cloud = point_cloud.transpose(1, 2)
        elif point_cloud.shape[1] != 3:
            raise ValueError(
                "point_cloud must have coordinate dimension 3, "
                f"got {tuple(point_cloud.shape)}"
            )
        if point_cloud.shape[2] == 0:
            raise ValueError("point_cloud must contain at least one point")
        return point_cloud.contiguous()

    def _load_pretrained_weights(self):
        checkpoint_path = Path(self.pretrained_checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Pretrained pointnet checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain `model_state_dict`."
            )

        source_state_dict = checkpoint["model_state_dict"]
        point_encoder_prefix = "point_encoder."
        point_encoder_state_dict = {
            key[len(point_encoder_prefix):]: value
            for key, value in source_state_dict.items()
            if key.startswith(point_encoder_prefix)
        }
        if not point_encoder_state_dict:
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain any `point_encoder.*` weights."
            )

        incompatible_keys = self.load_state_dict(point_encoder_state_dict, strict=True)
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            raise RuntimeError(
                "Failed to strictly load pretrained pointnet weights. "
                f"missing={incompatible_keys.missing_keys}, "
                f"unexpected={incompatible_keys.unexpected_keys}"
            )

        cprint(
            f"[PretrainedPointNetEncoderXYZ] loaded {len(point_encoder_state_dict)} tensors",
            "cyan",
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        x = self._to_channel_first(point_cloud)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, dim=2).values
        return F.relu(self.projection_norm(self.projection(x)))

    


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 encoder_dropout_prob: float = 0.0,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        self.excluded_obs_keys = {
            self.point_cloud_key,
            self.imagination_key,
            self.rgb_image_key,
        }
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.lowdim_keys = [
            key for key in observation_space.keys()
            if key not in self.excluded_obs_keys
        ]
        if len(self.lowdim_keys) == 0:
            raise RuntimeError("DP3Encoder requires at least one low-dimensional observation key.")
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] lowdim keys: {self.lowdim_keys}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.encoder_dropout = nn.Dropout(p=float(encoder_dropout_prob))
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        elif pointnet_type == "pointnet_pretrained_joint_collision_distance":
            if use_pc_color:
                raise ValueError(
                    "pointnet_pretrained_joint_collision_distance does not support point cloud color input."
                )
            pointcloud_encoder_cfg.in_channels = 3
            self.extractor = PretrainedPointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.lowdim_mlps = nn.ModuleDict()
        for key in self.lowdim_keys:
            key_shape = observation_space[key]
            self.lowdim_mlps[key] = nn.Sequential(
                *create_mlp(key_shape[0], output_dim, net_arch, state_mlp_activation_fn)
            )
            self.n_output_channels += output_dim

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")
        cprint(f"[DP3Encoder] encoder dropout: {encoder_dropout_prob}", "yellow")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel
            
        lowdim_feats = []
        for key in self.lowdim_keys:
            lowdim_value = observations[key]
            lowdim_feats.append(self.lowdim_mlps[key](lowdim_value))
        final_feat = torch.cat([pn_feat] + lowdim_feats, dim=-1)
        final_feat = self.encoder_dropout(final_feat)
        return final_feat


    def output_shape(self):
        return self.n_output_channels
