"""Contrastive image-text fine-tuning as a pathotrain :class:`~pathotrain.engine.Task`.

Training is SigLIP's sigmoid loss over the batch (every image against every
caption), with the model's learnable temperature and bias (Zhai et al. 2023);
pairs that share a caption are positives, not only the diagonal. Validation embeds held-out pairs and reports, under ``val/``:

- ``loss``: the mean batch loss.
- ``i2t_r1`` / ``r5`` / ``r10``: images whose caption ranks in the top k of all
  validation captions; ``t2i_r*`` the other way round; ``recall_mean`` their mean.
  Pairs that share a caption (sub-figures of one figure) all count as matches.
- what the ``evaluators`` add, e.g. zero-shot classification (:mod:`.zero_shot`).
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pathotrain.engine import Task, warmup_cosine

KS = (1, 5, 10)


def embed_images(model: nn.Module, pixel_values: Tensor) -> Tensor:
    return F.normalize(model.get_image_features(pixel_values=pixel_values).pooler_output, dim=-1)


def embed_texts(model: nn.Module, input_ids: Tensor) -> Tensor:
    return F.normalize(model.get_text_features(input_ids=input_ids).pooler_output, dim=-1)


def sigmoid_loss(logits_per_text: Tensor, groups: Tensor) -> Tensor:
    """SigLIP's loss: -log sigmoid(+-logit) summed over each caption's row, averaged;
    + for the images of the same caption group."""
    sign = 2 * (groups[:, None] == groups[None, :]).to(logits_per_text.dtype) - 1
    return -F.logsigmoid(sign * logits_per_text).sum(-1).mean()


def retrieval(images: Tensor, texts: Tensor, groups: Tensor) -> dict[str, float]:
    """Recall@k both ways for L2-normalised embeddings ``(N, D)``; ``groups[i]``
    names pair ``i``'s caption, so pairs of one caption match each other."""
    sims = images @ texts.T
    match = groups[:, None] == groups[None, :]
    out = {}
    for name, s, m in (("i2t", sims, match), ("t2i", sims.T, match.T)):
        # The rank of the best match: how many non-matches score above it.
        best = s.masked_fill(~m, -torch.inf).max(1, keepdim=True).values
        rank = ((s > best) & ~m).sum(1)
        out |= {f"{name}_r{k}": (rank < k).float().mean().item() for k in KS}
    out["recall_mean"] = sum(out.values()) / len(out)
    return out


class ContrastiveTask(Task):
    """``model`` (a SigLIP model, optionally LoRA-wrapped, or a
    :class:`~.models.LockedImageCLIP`) trained on batches of ``pixel_values`` (or
    ``image_features``), ``input_ids`` and ``groups`` (:class:`~.data.CaptionCollator`).
    Each of ``evaluators`` maps the model to extra metrics, run every validation."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_epochs: float = 0.0,
        evaluators: list[Callable[[nn.Module], dict[str, float]]] | None = None,
        trainable_only: bool = False,
    ) -> None:
        super().__init__()
        self.model, self.lr, self.weight_decay = model, lr, weight_decay
        self.warmup_epochs, self.evaluators = warmup_epochs, evaluators or []
        self.checkpoint_trainable_only = trainable_only
        self._val: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []

    def _forward(self, batch):
        out = self.model(**{k: v for k, v in batch.items() if k != "groups"})
        return out, sigmoid_loss(out.logits_per_text.float(), batch["groups"])

    def training_step(self, batch) -> tuple[Tensor, dict[str, Tensor]]:
        # shortcut: in-batch negatives only; under DDP the other ranks' pairs are not used.
        _, loss = self._forward(batch)
        scale, bias = self._logit_params()
        return loss, {"loss": loss, "temperature": scale.exp(), "bias": bias}

    def validation_step(self, batch) -> None:
        out, loss = self._forward(batch)
        # shortcut: embeddings gathered on one device; under DDP each rank scores its shard only.
        images = F.normalize(out.image_embeds.float(), dim=-1)
        texts = F.normalize(out.text_embeds.float(), dim=-1)
        self._val.append((images, texts, batch["groups"], loss.detach()))

    def compute_metrics(self) -> dict[str, float]:
        images, texts, groups, losses = (list(x) for x in zip(*self._val, strict=True))
        self._val = []
        metrics = {"loss": torch.stack(losses).mean().item()}
        metrics |= retrieval(torch.cat(images), torch.cat(texts), torch.cat(groups))
        metrics = {f"val/{k}": v for k, v in metrics.items()}
        for evaluate in self.evaluators:
            metrics |= evaluate(self.model)
        return metrics

    def _logit_params(self) -> tuple[Tensor, Tensor]:
        params = dict(self.model.named_parameters())
        scale = next(p for n, p in params.items() if n.endswith("logit_scale"))
        bias = next(p for n, p in params.items() if n.endswith("logit_bias"))
        return scale.detach().squeeze(), bias.detach().squeeze()

    def configure_optimizers(
        self, total_steps: int, steps_per_epoch: int
    ) -> tuple[Optimizer, LRScheduler | None]:
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                # No weight decay on biases, norms, the logit scale / bias and position embeddings.
                (decay if p.ndim > 1 and "position_embedding" not in name else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = AdamW([g for g in groups if g["params"]], lr=self.lr)
        return optimizer, warmup_cosine(optimizer, total_steps, steps_per_epoch, self.warmup_epochs)
