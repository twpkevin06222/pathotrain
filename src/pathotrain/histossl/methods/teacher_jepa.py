"""teacher_jepa: a frozen teacher's patch tokens as masked-prediction targets.

The teacher (frozen Midnight) sees the whole view; the student sees the same
view with block-masked patches replaced by its mask token, and a small
transformer predictor maps the student's tokens to the teacher's. The loss is
smooth L1 at the masked patches against the teacher's final patch tokens,
layer-normalised per token (as data2vec normalises its targets, Baevski et al.
2022). This follows the stage-2 recipe described for GenBio-PathFM (continued
pretraining against a frozen teacher); no code was ported from it, nor from
I-JEPA (Assran et al. 2023, CC-BY-NC), whose predictor idea this borrows.

With LoRA, the teacher is the student's own backbone with its adapters off
(``disable_adapter``), so the 1.1B weights are held once; otherwise it is a
frozen copy.
"""

import copy
import functools

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pathotrain.histossl.adaptation import apply_adaptation
from pathotrain.histossl.backbones import build_backbone, tokens
from pathotrain.histossl.masking import block_mask, random_mask
from pathotrain.histossl.methods.base import SSLTask


class Predictor(nn.Module):
    """Student tokens ``(B, N, dim)`` -> predicted teacher tokens ``(B, N, dim)``:
    a narrow pre-norm transformer between two projections. The tokens already
    carry their positions from the backbone, so it adds no position embedding."""

    def __init__(self, dim: int, width: int = 384, depth: int = 6, heads: int = 6) -> None:
        super().__init__()
        self.inp = nn.Linear(dim, width)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.norm(self.blocks(self.inp(x))))


class TeacherJEPAModel(nn.Module):
    """``forward(images, mask)`` -> ``(predicted, target)`` patch tokens, both
    ``(B, N, dim)``; the target without gradients. ``teacher=None`` uses the
    (peft) backbone with its adapters off."""

    def __init__(
        self, backbone: nn.Module, predictor: Predictor, teacher: nn.Module | None
    ) -> None:
        super().__init__()
        self.backbone, self.predictor, self.teacher = backbone, predictor, teacher
        if teacher is not None:
            teacher.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()  # frozen
        return self

    def forward(self, images: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            if self.teacher is None:
                with self.backbone.disable_adapter():
                    target = tokens(self.backbone, images)[1]
            else:
                target = tokens(self.teacher, images)[1]
        return self.predictor(tokens(self.backbone, images, mask)[1]), target


class TeacherJEPATask(SSLTask):
    """Train a :class:`TeacherJEPAModel` (see the module docstring); ``masking``
    is ``block`` or ``random``, ``ratio`` the masked fraction ``(low, high)``,
    ``p`` the share of views block-masked at all. Other arguments as for :class:`SSLTask`."""

    def __init__(
        self,
        model: TeacherJEPAModel,
        masking: str = "block",
        ratio: tuple[float, float] = (0.4, 0.6),
        min_aspect: float = 0.3,
        p: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        if masking == "block":
            self.mask_fn = functools.partial(block_mask, ratio=ratio, min_aspect=min_aspect, p=p)
        else:
            self.mask_fn = functools.partial(random_mask, ratio=ratio)
        self.patch = model.backbone.config.patch_size

    def _losses(self, batch, generator: torch.Generator | None) -> dict[str, Tensor]:
        images = self._images(batch)
        grid = images.shape[-1] // self.patch
        mask = torch.stack([self.mask_fn(grid, generator=generator) for _ in images])
        mask = mask.to(images.device)
        predicted, target = self.model(images, mask)
        target = F.layer_norm(target.float(), target.shape[-1:])
        return {"loss": F.smooth_l1_loss(predicted.float()[mask], target[mask])}


def build(cfg, ssl, probe=None) -> TeacherJEPATask:
    """The task of a validated config: ``cfg`` the experiment dict, ``ssl`` its
    :class:`~pathotrain.histossl.config.SSLConfig`. Method params: the predictor's
    ``width``, ``depth`` and ``heads``."""
    bb, ad = ssl.backbone, ssl.adaptation
    backbone = build_backbone(bb.init, bb.id, bb.arch, bb.grad_checkpointing, bb.dtype)
    teacher = None
    if ad.type != "lora":
        teacher_id = ssl.teacher.id or bb.id
        # shortcut: a full fp32 copy (4.4 GB for ViT-g); with last_n only the last blocks
        # differ, upgrade by sharing the frozen trunk if memory gets tight.
        if teacher_id == bb.id:
            teacher = copy.deepcopy(backbone)
        else:
            teacher = build_backbone("pretrained", teacher_id, False, dtype=bb.dtype)
    student = apply_adaptation(backbone, ad.type, ad.n, ad.rank, ad.alpha, ad.dropout, ad.targets)
    predictor = Predictor(backbone.config.hidden_size, **ssl.method.params)
    m = ssl.masking
    return TeacherJEPATask(
        TeacherJEPAModel(student, predictor, teacher),
        masking=m.type,
        ratio=m.ratio,
        min_aspect=m.min_aspect,
        p=m.p,
        seed=cfg.get("seed", 0),
        probe=probe,
        probe_every=ssl.eval.every if ssl.eval is not None else 1,
        **cfg.get("task", {}),
    )
