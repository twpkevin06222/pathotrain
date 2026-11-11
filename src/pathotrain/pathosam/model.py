"""SAM 3's point and box model (``Sam3TrackerModel``) with its image encoder frozen."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pathotrain.pathofm.sam3 import SAM3_REPO, SAM3_REVISION

PAD = -10  # the label of a padding point, which SAM's prompt encoder ignores


class InteractiveSam(nn.Module):
    """``Sam3TrackerModel`` for one object per prompt, its image encoder frozen.

    ``forward`` does both halves, so the training engine's wrapper (precision, device)
    applies to each: given ``images``, it returns the image embeddings; given
    ``embeddings`` and prompts, the mask logits and predicted IoUs.

    Images are normalised tiles ``(B, 3, H, W)`` (mean = std = 0.5, as SAM 3's processor);
    they are resized to the model's input size here. Points and boxes are in tile pixels.
    """

    def __init__(self, revision: str = SAM3_REVISION) -> None:
        super().__init__()
        from transformers import Sam3TrackerModel

        self.sam = Sam3TrackerModel.from_pretrained(SAM3_REPO, revision=revision)
        self.sam.vision_encoder.requires_grad_(False)
        self.side = self.sam.config.vision_config.backbone_config.image_size  # 1008

    def train(self, mode: bool = True) -> "InteractiveSam":
        super().train(mode)
        self.sam.vision_encoder.eval()  # frozen: no dropout
        return self

    def forward(
        self,
        images: Tensor | None = None,
        embeddings: list[Tensor] | None = None,
        size: tuple[int, int] | None = None,
        points: Tensor | None = None,
        labels: Tensor | None = None,
        boxes: Tensor | None = None,
        masks: Tensor | None = None,
        multimask: bool = False,
    ) -> list[Tensor] | tuple[Tensor, Tensor]:
        """Embeddings of ``images``; or, for N objects with ``embeddings`` (one entry per
        object, ``(N, ...)``) of a tile of ``size``: mask logits ``(N, M, h, w)`` and
        predicted IoUs ``(N, M)``, M = 3 with ``multimask``, else 1.

        ``points`` ``(N, K, 2)`` as ``(x, y)`` with ``labels`` ``(N, K)`` (1 positive,
        0 negative, :data:`PAD` ignored), ``boxes`` ``(N, 4)`` as ``(x0, y0, x1, y1)``,
        ``masks`` ``(N, 1, h, w)``: the previous round's logits.
        """
        if images is not None:
            with torch.no_grad():
                pixels = F.interpolate(images, (self.side, self.side), mode="bilinear")
                return self.sam.get_image_embeddings(pixels)
        scale = torch.tensor(
            [self.side / size[1], self.side / size[0]], device=embeddings[0].device
        )
        kwargs = {}
        if points is not None:
            kwargs["input_points"] = (points.float() * scale)[:, None]
            kwargs["input_labels"] = labels[:, None]
        if boxes is not None:
            kwargs["input_boxes"] = (boxes.float() * scale.repeat(2))[:, None]
        out = self.sam(
            image_embeddings=embeddings, input_masks=masks, multimask_output=multimask, **kwargs
        )
        return out.pred_masks[:, 0], out.iou_scores[:, 0]


def to_tile(logits: Tensor, size: tuple[int, int]) -> Tensor:
    """Low-resolution logits ``(N, M, h, w)`` at tile ``size``, as SAM 3's processor
    resizes them: up to the model's input size (1008), then down to the tile's."""
    side = 1008
    full = F.interpolate(logits.float(), (side, side), mode="bilinear")
    return F.interpolate(full, size, mode="bilinear")
