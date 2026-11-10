"""Foundation-model (ViT) backbones as SMP encoders.

SMP decoders expect a CNN-like pyramid: features at strides 1, 2, 4, 8, 16 and
32. A ViT has one token grid, so :class:`ViTPyramidEncoder` takes the patch
tokens (the model card's segmentation embedding, not the CLS-based
classification embedding) of four layers, each after the backbone's final
LayerNorm, and resamples them DPT-style: layer 1 up to stride 4, layer 2 up to
stride 8, layer 3 kept at stride 16, layer 4 down to stride 32. With
``layers=(40, 40, 40, 40)`` every scale comes from the card's last-layer
embedding only.
The stride-2 feature is empty (0 channels), as in SMP's own transformer encoders.

The tile is resized so the token grid is exactly ``H/16 x W/16`` (512 px ->
448 px for patch 14), which lines the grid up with stride 16.
"""

import functools

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torch import Tensor, nn

from pathotrain.pathofm import FOUNDATION_MODELS, load_backbone


def token_grids(
    backbone: nn.Module, x: Tensor, layers: tuple[int, ...], freeze: bool = True
) -> list[Tensor]:
    """Patch tokens of the 1-based transformer ``layers`` as ``(B, D, H/16, W/16)`` grids.

    ``x`` is resized so the token grid is exactly ``H/16 x W/16`` (512 px -> 448 px for
    patch 14). With ``freeze`` the backbone runs without gradients.
    """
    b, _, h, w = x.shape
    gh, gw, patch = h // 16, w // 16, backbone.config.patch_size
    pixels = F.interpolate(x, size=(gh * patch, gw * patch), mode="bilinear", align_corners=False)
    with torch.set_grad_enabled(torch.is_grad_enabled() and not freeze):
        hidden = backbone(pixel_values=pixels, output_hidden_states=True).hidden_states
    norm = getattr(backbone, "layernorm", nn.Identity())
    # Final LayerNorm on every layer, so the last one is exactly the model card's
    # segmentation embedding (last_hidden_state); drop CLS/register tokens.
    return [norm(hidden[i][:, -gh * gw :]).transpose(1, 2).reshape(b, -1, gh, gw) for i in layers]


def adapt_backbone(
    backbone: nn.Module, freeze: bool, adaptation: dict | None
) -> tuple[nn.Module, bool]:
    """Set which weights of ``backbone`` train and return the backbone to use (a peft
    model for LoRA) and whether it runs without gradients.

    Without ``adaptation``, ``freeze`` keeps it fixed or trains it whole. ``adaptation``
    is histossl's (:func:`~pathotrain.histossl.adaptation.apply_adaptation`), e.g.
    ``{type: lora, rank: 8}`` (LoRA on every linear layer of the blocks; needs peft),
    ``{type: lora, rank: 16, targets: [q_proj, v_proj]}``, ``{type: full}`` or
    ``{type: frozen}``; it overrides ``freeze``.
    """
    if adaptation is None:
        backbone.requires_grad_(not freeze)
        return backbone, freeze
    from pathotrain.histossl.adaptation import apply_adaptation  # peft only when asked

    kwargs = dict(adaptation)
    kind = kwargs.pop("type")
    return apply_adaptation(backbone, kind, **kwargs), kind == "frozen"


def _resample(dim: int, channels: int, stride: int) -> nn.Sequential:
    """1x1 projection, then resample the stride-16 grid to ``stride``."""
    layers = [nn.Conv2d(dim, channels, 1)]
    if stride < 16:
        k = 16 // stride
        layers.append(nn.ConvTranspose2d(channels, channels, k, stride=k))
    elif stride == 32:
        layers.append(nn.Conv2d(channels, channels, 3, stride=2, padding=1))
    return nn.Sequential(*layers)


class ViTPyramidEncoder(nn.Module, EncoderMixin):
    """SMP encoder around a ViT with a ``transformers``-style interface
    (``config.patch_size``, ``config.hidden_size``, ``config.num_hidden_layers``,
    ``forward(pixel_values, output_hidden_states=True)``), e.g. ``Dinov2Model``.

    Args:
        backbone: A foundation model name from ``pathotrain.pathofm`` or a model.
        layers: Four 1-based transformer layers for strides 4, 8, 16 and 32.
            Default: evenly spaced, ending with the last layer.
        channels: Output channels for strides 4, 8, 16 and 32.
        freeze: Keep the backbone fixed and run it without gradients (only the
            resampling and the decoder train). ``False`` fine-tunes it; then
            consider ``backbone.gradient_checkpointing_enable()``.
        adaptation: Train the backbone partly instead, e.g. ``{type: lora, rank: 8}``;
            see :func:`adapt_backbone`. Overrides ``freeze``.
    """

    def __init__(
        self,
        backbone: str | nn.Module,
        layers: tuple[int, int, int, int] | None = None,
        channels: tuple[int, int, int, int] = (256, 512, 1024, 1024),
        freeze: bool = True,
        depth: int = 5,
        output_stride: int = 32,
        adaptation: dict | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= depth <= 5 or output_stride not in (8, 16, 32):
            raise ValueError(
                f"need encoder_depth in [1, 5] and output_stride 8, 16 or 32, "
                f"got {depth} and {output_stride}"
            )
        self.name = backbone if isinstance(backbone, str) else type(backbone).__name__
        self.backbone = load_backbone(backbone) if isinstance(backbone, str) else backbone
        cfg = self.backbone.config
        n = cfg.num_hidden_layers
        self.layers = layers or tuple(round(n * i / 4) for i in range(1, 5))
        if len(self.layers) != 4 or not all(1 <= i <= n for i in self.layers):
            raise ValueError(f"layers must be four indices in [1, {n}], got {self.layers}")
        if len(channels) != 4:
            raise ValueError(f"channels must have four values, got {channels}")
        self.patch_size = cfg.patch_size
        # The backbone's LayerNorm under a second name: unused here (token_grids applies
        # it), kept so existing checkpoints, which hold it as encoder.norm.*, still load.
        self.norm = getattr(self.backbone, "layernorm", nn.Identity())
        # A dilated encoder (output_stride 8 or 16, for DeepLab and PAN) keeps the
        # deepest features at output_stride instead of downsampling further.
        self.resample = nn.ModuleList(
            _resample(cfg.hidden_size, c, min(s, output_stride))
            for c, s in zip(channels, (4, 8, 16, 32), strict=True)
        )
        self.backbone, self.freeze = adapt_backbone(self.backbone, freeze, adaptation)
        self._depth = depth
        self._in_channels = 3
        self._output_stride = output_stride
        self._out_channels = [3, 0, *channels]

    def forward(self, x: Tensor) -> list[Tensor]:
        b, _, h, w = x.shape
        grids = token_grids(self.backbone, x, self.layers, self.freeze)
        feats = [resample(g) for g, resample in zip(grids, self.resample, strict=True)]
        return [x, x.new_empty(b, 0, h // 2, w // 2), *feats][: self._depth + 1]

    def make_dilated(self, output_stride: int) -> None:
        pass  # handled in __init__: the resampling targets output_stride


def build_fm_model(
    arch: str,
    backbone: str | nn.Module,
    num_classes: int,
    in_channels: int = 3,
    encoder_kwargs: dict | None = None,
    **kwargs,
) -> nn.Module:
    """SMP ``arch`` (e.g. ``"unet"``, ``"fpn"``, ``"upernet"``) on a
    :class:`ViTPyramidEncoder`. ``encoder_kwargs`` go to the encoder, ``kwargs``
    to the SMP model (e.g. ``decoder_channels``)."""
    if in_channels != 3:
        raise ValueError("foundation-model encoders take RGB input only")
    name = f"_pathotrain_vit_{id(backbone)}"
    # SMP builds encoders from its registry; register this one just for the call.
    # The partial carries the backbone, so SMP does not deep-copy a large model.
    smp.encoders.encoders[name] = {
        "encoder": functools.partial(ViTPyramidEncoder, backbone, **(encoder_kwargs or {})),
        "pretrained_settings": {},
        "params": {},
    }
    try:
        return smp.create_model(
            arch, encoder_name=name, encoder_weights=None, classes=num_classes, **kwargs
        )
    finally:
        del smp.encoders.encoders[name]


__all__ = [
    "FOUNDATION_MODELS",
    "ViTPyramidEncoder",
    "adapt_backbone",
    "build_fm_model",
    "token_grids",
]
