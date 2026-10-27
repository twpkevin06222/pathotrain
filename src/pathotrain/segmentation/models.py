"""Segmentation networks from segmentation_models_pytorch (SMP) and nnU-Net.

Every builder returns a plain ``nn.Module`` mapping ``(B, C, H, W)`` images to
``(B, num_classes, H, W)`` logits, so the engine can train any of them::

    build_model("smp:unet", num_classes=6, encoder_name="resnet34", encoder_weights="imagenet")
    build_model("nnunet:resenc", num_classes=6)
"""

import segmentation_models_pytorch as smp
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn


def build_model(name: str, num_classes: int, in_channels: int = 3, **kwargs) -> nn.Module:
    """Build ``"smp:<arch>"`` or ``"nnunet:plain"`` / ``"nnunet:resenc"``.

    Args:
        name: ``<library>:<architecture>``. SMP architectures are those of
            ``smp.create_model`` (``unet``, ``unetplusplus``, ``deeplabv3plus``,
            ``fpn``, ``segformer``, ...).
        num_classes: Output channels, including background.
        in_channels: Image channels.
        **kwargs: Passed to ``smp.create_model`` (e.g. ``encoder_name``,
            ``encoder_weights``) or to :func:`nnunet`.
    """
    library, _, arch = name.partition(":")
    if library == "smp":
        return smp.create_model(arch, in_channels=in_channels, classes=num_classes, **kwargs)
    if library == "nnunet":
        return nnunet(arch, num_classes, in_channels, **kwargs)
    raise ValueError(f"unknown model {name!r}, expected 'smp:<arch>' or 'nnunet:<arch>'")


def nnunet(
    arch: str,
    num_classes: int,
    in_channels: int = 3,
    n_stages: int = 7,
    base_features: int = 32,
    max_features: int = 512,
) -> nn.Module:
    """2D nnU-Net network (architecture only, no nnU-Net pipeline).

    Mirrors nnU-Net's 2D defaults: features double per stage from
    ``base_features`` up to ``max_features``, 3x3 convs, stride-2 downsampling,
    instance norm, leaky ReLU and He initialisation. ``"plain"`` is the original
    U-Net, ``"resenc"`` the residual encoder (ResEnc) variant. Deep supervision
    is off, so the output is a single logit map. The input side must be
    divisible by ``2 ** (n_stages - 1)`` (64 by default). No pretrained weights.
    """
    common = {
        "input_channels": in_channels,
        "n_stages": n_stages,
        "features_per_stage": [min(base_features * 2**i, max_features) for i in range(n_stages)],
        "conv_op": nn.Conv2d,
        "kernel_sizes": 3,
        "strides": [1] + [2] * (n_stages - 1),
        "num_classes": num_classes,
        "conv_bias": True,
        "norm_op": nn.InstanceNorm2d,
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "nonlin": nn.LeakyReLU,
        "nonlin_kwargs": {"inplace": True},
    }
    if arch == "plain":
        model = PlainConvUNet(n_conv_per_stage=2, n_conv_per_stage_decoder=2, **common)
    elif arch == "resenc":
        blocks = [1, 3, 4] + [6] * (n_stages - 3)
        model = ResidualEncoderUNet(
            n_blocks_per_stage=blocks[:n_stages], n_conv_per_stage_decoder=1, **common
        )
    else:
        raise ValueError(f"unknown nnU-Net architecture {arch!r}, expected 'plain' or 'resenc'")
    model.apply(InitWeights_He(1e-2))
    return model
