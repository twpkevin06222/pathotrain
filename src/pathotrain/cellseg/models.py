"""Cell (nucleus) segmentation networks, written from their papers. No authors' code or weights.

Each maps ``(B, 3, H, W)`` images to ``(B, num_classes, H, W)`` semantic logits, the
nuclear-type (NT) branch; train these with :mod:`pathotrain.segmentation.train`, naming
the model ``cellseg:<arch>``. With ``instance=True`` each returns HoVer-Net's three
branches as a dict (see :data:`BRANCHES`) for :mod:`pathotrain.cellseg.train`.

- ``hovernet``: HoVer-Net (Graham et al. 2019). Pre-activation ResNet-50 encoder whose
  stem keeps full resolution, and HoVer-Net's dense decoder; its "fast" variant, with
  padded convolutions, so the output has the input's size.
- ``hovernext``: HoVer-NeXt (Baumann et al. 2024). ConvNeXt-V2 encoder and U-Net decoder.
- ``cellvit``: CellViT (Hörst et al. 2024). A ``pathotrain.pathofm`` ViT encoder and the
  CellViT decoder. CellViT++'s cell classifier on the encoder's tokens is in
  :mod:`pathotrain.cellseg.classify`.

Encoder weights are the only pretrained weights: timm ImageNet weights, or a
``pathofm`` foundation model.
"""

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pathotrain.pathofm import load_backbone
from pathotrain.segmentation.encoders import token_grids

# The instance branches, as HoVer-Net: nucleus vs background (NP, 2 logits), horizontal and
# vertical distance maps (HV, 2 values) and nucleus type (NT, num_classes logits).
BRANCHES = ("np", "hv", "nt")


def _channels(num_classes: int, instance: bool) -> dict[str, int]:
    return {"np": 2, "hv": 2, "nt": num_classes} if instance else {"nt": num_classes}


def _outputs(out: dict[str, Tensor], instance: bool) -> Tensor | dict[str, Tensor]:
    return out if instance else out["nt"]


def build_model(arch: str, num_classes: int, in_channels: int = 3, **kwargs) -> nn.Module:
    """Build ``"hovernet"``, ``"hovernext"`` or ``"cellvit"``; ``kwargs`` go to its class."""
    if arch == "hovernet":
        return HoVerNet(num_classes, in_channels, **kwargs)
    if arch == "hovernext":
        return hovernext(num_classes, in_channels, **kwargs)
    if arch == "cellvit":
        if in_channels != 3:
            raise ValueError("CellViT's foundation-model encoders take RGB input only")
        kwargs.pop("encoder_weights", None)  # the foundation model loads its own weights
        return CellViT(num_classes=num_classes, **kwargs)
    raise ValueError(
        f"unknown cell segmentation model {arch!r}, expected hovernet, hovernext or cellvit"
    )


def _bn_relu(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(inplace=True))


class _DenseBlock(nn.Module):
    """HoVer-Net's dense block: ``units`` times (BN-ReLU, 1x1 conv to 128, BN-ReLU,
    ``ksize`` grouped conv to 32), each unit's output concatenated to its input."""

    def __init__(self, channels: int, units: int, ksize: int = 5) -> None:
        super().__init__()
        self.units = nn.ModuleList(
            nn.Sequential(
                _bn_relu(channels + i * 32),
                nn.Conv2d(channels + i * 32, 128, 1, bias=False),
                _bn_relu(128),
                nn.Conv2d(128, 32, ksize, padding=ksize // 2, groups=4, bias=False),
            )
            for i in range(units)
        )
        self.out = _bn_relu(channels + units * 32)

    def forward(self, x: Tensor) -> Tensor:
        for unit in self.units:
            x = torch.cat([x, unit(x)], 1)
        return self.out(x)


class _HoVerBranch(nn.Module):
    """One HoVer-Net decoder branch: stride 8 to 1, adding the encoder skips."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.u3 = nn.Sequential(
            nn.Conv2d(1024, 256, 5, padding=2, bias=False),
            _DenseBlock(256, 8),
            nn.Conv2d(512, 512, 1, bias=False),
        )
        self.u2 = nn.Sequential(
            nn.Conv2d(512, 128, 5, padding=2, bias=False),
            _DenseBlock(128, 4),
            nn.Conv2d(256, 256, 1, bias=False),
        )
        self.u1 = nn.Conv2d(256, 64, 5, padding=2, bias=False)
        self.head = nn.Sequential(_bn_relu(64), nn.Conv2d(64, channels, 1))

    def forward(self, d0: Tensor, d1: Tensor, d2: Tensor, d3: Tensor) -> Tensor:
        u = self.u3(F.interpolate(d3, scale_factor=2) + d2)
        u = self.u2(F.interpolate(u, scale_factor=2) + d1)
        u = self.u1(F.interpolate(u, scale_factor=2) + d0)
        return self.head(u)


class HoVerNet(nn.Module):
    """HoVer-Net: one decoder branch per output (NT; with ``instance``, also NP and HV).

    The encoder is timm's pre-activation ResNet-50 (``resnetv2_50``) with HoVer-Net's
    changes: stride-1 stem, no max pool, so its four stages sit at strides 1, 2, 4 and 8.

    Args:
        num_classes: Output channels, including background.
        in_channels: Image channels.
        encoder_weights: ``"imagenet"`` loads timm's ImageNet weights (Apache-2.0),
            ``None`` trains from scratch.
        instance: Return the NP, HV and NT branches as a dict.
    """

    output_stride = 8  # input sides must be divisible by this

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        encoder_weights: str | None = "imagenet",
        instance: bool = False,
    ) -> None:
        super().__init__()
        if encoder_weights not in ("imagenet", None):
            raise ValueError(f"encoder_weights must be 'imagenet' or None, got {encoder_weights!r}")
        self.encoder = timm.create_model(
            "resnetv2_50", pretrained=encoder_weights == "imagenet", in_chans=in_channels
        )
        self.encoder.stem.conv.stride = (1, 1)
        self.encoder.stem.pool = nn.Identity()
        self.encoder.head = nn.Identity()
        self.conv_bot = nn.Conv2d(2048, 1024, 1, bias=False)
        self.instance = instance
        self.branches = nn.ModuleDict(
            {k: _HoVerBranch(c) for k, c in _channels(num_classes, instance).items()}
        )

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        e = self.encoder
        d0 = e.stages[0](e.stem(x))
        d1 = e.stages[1](d0)
        d2 = e.stages[2](d1)
        d3 = self.conv_bot(e.norm(e.stages[3](d2)))
        out = {k: branch(d0, d1, d2, d3) for k, branch in self.branches.items()}
        return _outputs(out, self.instance)


def hovernext(
    num_classes: int,
    in_channels: int = 3,
    encoder: str = "convnextv2_base",
    encoder_weights: str | None = None,
    instance: bool = False,
    **kwargs,
) -> nn.Module:
    """HoVer-NeXt's network: a timm ``encoder`` (ConvNeXt-V2 in the paper) under SMP's
    U-Net decoder; ``kwargs`` go to ``smp.Unet``. With ``instance``, the NP, HV and NT
    branches are channel groups of one shared decoder's output (a dict).

    ``encoder_weights`` defaults to ``None`` because timm's ConvNeXt-V2 ImageNet weights are
    CC-BY-NC-4.0. ``encoder="convnext_base"`` with ``encoder_weights="imagenet"``
    gives ConvNeXt (V1) ImageNet weights under Apache-2.0.
    """
    channels = _channels(num_classes, instance)
    model = smp.create_model(
        "unet",
        encoder_name=f"tu-{encoder}",
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=sum(channels.values()),
        **kwargs,
    )
    return _SplitChannels(model, channels) if instance else model


class _SplitChannels(nn.Module):
    """``model``'s output channels as a dict of consecutive groups of ``channels``."""

    def __init__(self, model: nn.Module, channels: dict[str, int]) -> None:
        super().__init__()
        self.model, self.channels = model, channels
        self.output_stride = model.encoder.output_stride

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        parts = self.model(x).split(list(self.channels.values()), 1)
        return dict(zip(self.channels, parts, strict=True))


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))


def _deconv(cin: int, cout: int) -> nn.Sequential:
    """2x upsampling: transposed conv, then a conv block (CellViT's Deconv2DBlock)."""
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 2, stride=2), _conv(cout, cout))


class _CellViTBranch(nn.Module):
    """One CellViT decoder branch: upsamples the deepest tokens, merging the shared skips."""

    def __init__(self, dim: int, bottleneck: int, channels: int) -> None:
        super().__init__()
        b = bottleneck
        self.bottleneck_upsampler = nn.ConvTranspose2d(dim, b, 2, stride=2)
        self.decoder3_upsampler = nn.Sequential(
            _conv(2 * b, b), _conv(b, b), _conv(b, b), nn.ConvTranspose2d(b, 256, 2, stride=2)
        )
        self.decoder2_upsampler = nn.Sequential(
            _conv(512, 256), _conv(256, 256), nn.ConvTranspose2d(256, 128, 2, stride=2)
        )
        self.decoder1_upsampler = nn.Sequential(
            _conv(256, 128), _conv(128, 128), nn.ConvTranspose2d(128, 64, 2, stride=2)
        )
        self.head = nn.Sequential(_conv(128, 64), _conv(64, 64), nn.Conv2d(64, channels, 1))

    def forward(self, z4: Tensor, s3: Tensor, s2: Tensor, s1: Tensor, s0: Tensor) -> Tensor:
        u = self.bottleneck_upsampler(z4)  # stride 8
        u = self.decoder3_upsampler(torch.cat([s3, u], 1))  # 4
        u = self.decoder2_upsampler(torch.cat([s2, u], 1))  # 2
        u = self.decoder1_upsampler(torch.cat([s1, u], 1))  # 1
        return self.head(torch.cat([s0, u], 1))


class CellViT(nn.Module):
    """CellViT on a ViT encoder: shared skips, one decoder branch per output (NT; with
    ``instance``, also NP and HV).

    Tokens of four layers (as ``(B, D, H/16, W/16)`` grids, see
    :func:`~pathotrain.segmentation.encoders.token_grids`) and the image itself are the
    skips of a U-Net-like decoder that upsamples by transposed convolutions.

    Args:
        backbone: A foundation model name from ``pathotrain.pathofm`` or a ViT with the
            ``transformers`` interface ``ViTPyramidEncoder`` takes.
        num_classes: Output channels, including background.
        layers: Four 1-based transformer layers, shallow to deep. Default: evenly
            spaced, ending with the last layer.
        freeze: Keep the backbone fixed (only the decoder trains). ``False`` fine-tunes it.
        skip_dims: Channels of the stride-8 and stride-4 steps of the shallowest skip.
        bottleneck_dim: Channels of the deepest skip.
        instance: Return the NP, HV and NT branches as a dict.
    """

    output_stride = 16

    def __init__(
        self,
        backbone: str | nn.Module,
        num_classes: int,
        layers: tuple[int, int, int, int] | None = None,
        freeze: bool = True,
        skip_dims: tuple[int, int] = (512, 256),
        bottleneck_dim: int = 512,
        instance: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = load_backbone(backbone) if isinstance(backbone, str) else backbone
        n, dim = self.backbone.config.num_hidden_layers, self.backbone.config.hidden_size
        self.layers = tuple(layers or (round(n * i / 4) for i in range(1, 5)))
        if len(self.layers) != 4 or not all(1 <= i <= n for i in self.layers):
            raise ValueError(f"layers must be four indices in [1, {n}], got {self.layers}")
        self.freeze = freeze
        self.backbone.requires_grad_(not freeze)
        s1, s2, b = *skip_dims, bottleneck_dim
        self.decoder0 = nn.Sequential(_conv(3, 32), _conv(32, 64))  # stride 1, the image
        self.decoder1 = nn.Sequential(_deconv(dim, s1), _deconv(s1, s2), _deconv(s2, 128))
        self.decoder2 = nn.Sequential(_deconv(dim, s1), _deconv(s1, 256))
        self.decoder3 = _deconv(dim, b)
        self.instance = instance
        self.branches = nn.ModuleDict(
            {k: _CellViTBranch(dim, b, c) for k, c in _channels(num_classes, instance).items()}
        )

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        z1, z2, z3, z4 = token_grids(self.backbone, x, self.layers, self.freeze)
        skips = (self.decoder3(z3), self.decoder2(z2), self.decoder1(z1), self.decoder0(x))
        out = {k: branch(z4, *skips) for k, branch in self.branches.items()}
        return _outputs(out, self.instance)
