"""SAM 3's image encoder as a dense backbone (``"sam3"``), for segmentation and cellseg.

SAM 3 (``facebook/sam3``, Meta) is gated on Hugging Face: request access and log in
first. Its weights are under the SAM License: commercial use is allowed, but anything
built on them that you distribute (e.g. a fine-tuned model) must go out under that
licence with a copy of it, and publications must acknowledge SAM.

Only the image encoder's ViT is loaded (ViT-L/14: 32 layers, 1024-dim tokens, windowed
attention with four global layers, 2D RoPE), from the ``detector_model.vision_encoder.
backbone.`` weights of the checkpoint. It has no CLS token, so it serves dense tasks
only. Normalisation is mean = std = 0.5 (``MIDNIGHT_MEAN`` / ``MIDNIGHT_STD``).

SAM 3 runs at 1008 px (a 72 x 72 token grid). Its global layers precompute RoPE
positions for that grid; :class:`Sam3Backbone` recomputes them for the grid of each
input, scaled to the window as SAM 3 does, so tiles of any size multiple of 14 work.
"""

import json
from types import SimpleNamespace

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor, nn
from transformers import Sam3ViTConfig, Sam3ViTModel
from transformers.modeling_outputs import BaseModelOutput

SAM3_REPO = "facebook/sam3"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
_PREFIX = "detector_model.vision_encoder.backbone."


class _CachedRope(nn.Module):
    """A layer's RoPE ``(cos, sin)``, which depend on the token grid only, computed once
    per grid (as SAM 3's ``Sam3ViTRotaryEmbedding`` does). Computed in numpy, so an ONNX
    trace sees constants: SAM 3's own RoPE module does not export."""

    def __init__(self, rope: nn.Module, positions: Tensor) -> None:
        super().__init__()
        self.inv_freq = rope.inv_freq.cpu().numpy()
        self.scaling = rope.attention_scaling
        self.update(positions)

    def update(self, positions: Tensor) -> None:
        freqs = positions.cpu().numpy()[..., None] * self.inv_freq  # (N, 2, F): h, w
        freqs = np.repeat(np.concatenate([freqs[:, 0], freqs[:, 1]], -1), 2, -1)[None]
        device = getattr(self, "cos", positions).device
        for name, table in (("cos", np.cos(freqs)), ("sin", np.sin(freqs))):
            tensor = torch.from_numpy((table * self.scaling).astype(np.float32)).to(device)
            self.register_buffer(name, tensor, persistent=False)

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.cos, self.sin


def cache_rope(vit: Sam3ViTModel) -> Sam3ViTModel:
    """Replace each layer's RoPE module of ``vit`` (in place) with tables computed once for
    its current grid, so the ViT traces to ONNX; returns ``vit``."""
    for layer in vit.layers:
        if not isinstance(layer.rotary_emb, _CachedRope):
            layer.rotary_emb = _CachedRope(layer.rotary_emb, layer.position_ids)
    return vit


class Sam3Backbone(nn.Module):
    """A ``Sam3ViTModel`` with the interface pathotrain's ViT encoders take: ``config``
    (``patch_size``, ``hidden_size``, ``num_hidden_layers``) and ``forward(pixel_values,
    output_hidden_states)`` returning tokens as ``(B, N, D)``, ``hidden_states[i]`` after
    layer ``i`` (0: the embeddings). Any input whose sides are multiples of 14 works."""

    def __init__(self, vit: Sam3ViTModel) -> None:
        super().__init__()
        self.vit = cache_rope(vit)
        self.config = vit.config
        grid = vit.config.image_size // vit.config.patch_size
        self._grid = (grid, grid)

    def _set_grid(self, height: int, width: int) -> None:
        """RoPE positions of the global-attention layers for a ``height x width`` grid."""
        if (height, width) == self._grid:
            return
        cfg, patch = self.config, self.config.patch_size
        # precompute_positions reads image_size as (width, height) in pixels.
        size = SimpleNamespace(
            image_size=(width * patch, height * patch),
            patch_size=patch,
            window_size=cfg.window_size,
        )
        for layer in self.vit.layers:
            if layer.window_size == 0:
                positions = layer.precompute_positions(size, 0).to(layer.position_ids)
                layer.position_ids = positions
                layer.rotary_emb.update(positions)
        self._grid = (height, width)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> BaseModelOutput:
        patch = self.config.patch_size
        self._set_grid(pixel_values.shape[-2] // patch, pixel_values.shape[-1] // patch)
        out = self.vit(pixel_values=pixel_values, output_hidden_states=output_hidden_states)
        hidden = None
        if output_hidden_states:
            hidden = tuple(h.flatten(1, 2) for h in out.hidden_states)  # (B, H, W, D) -> (B, N, D)
        return BaseModelOutput(last_hidden_state=out.last_hidden_state, hidden_states=hidden)


def sam3_config(path: str) -> Sam3ViTConfig:
    """The image encoder ViT's config from SAM 3's ``config.json``."""
    with open(path) as f:
        cfg = json.load(f)
    vision = cfg.get("detector_config", cfg)["vision_config"]["backbone_config"]
    return Sam3ViTConfig(**{k: v for k, v in vision.items() if k != "model_type"})


def load_sam3(dtype: torch.dtype = torch.float32, revision: str = SAM3_REVISION) -> Sam3Backbone:
    """Download (once, into the Hugging Face cache; needs access to the gated repo) and
    return SAM 3's image encoder ViT, weights in ``dtype``."""
    config = sam3_config(hf_hub_download(SAM3_REPO, "config.json", revision=revision))
    path = hf_hub_download(SAM3_REPO, "model.safetensors", revision=revision)
    with safe_open(path, "pt") as f:
        names = [k for k in f.keys() if k.startswith(_PREFIX)]  # noqa: SIM118 (not a dict)
        state = {k[len(_PREFIX) :]: f.get_tensor(k).to(dtype) for k in names}
    vit = Sam3ViTModel(config).to(dtype)  # its RoPE buffers are not in the checkpoint
    vit.load_state_dict(state, strict=True)
    return Sam3Backbone(vit.eval())
