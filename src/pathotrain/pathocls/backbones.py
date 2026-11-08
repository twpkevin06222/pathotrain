"""The backbones a classifier can sit on: a pathofm model, a histossl checkpoint or export
(Hugging Face DINOv2 models), or a timm model (``timm:<name>``). Everything that differs
between them (loading, the tile embedding, its width, preprocessing, gradient
checkpointing) is here; the rest of :mod:`pathotrain.pathocls` does not care which."""

import timm
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from pathotrain.histossl import backbones as hf
from pathotrain.histossl.eval.linear_probe import load_encoder
from pathotrain.pathofm import classification_embedding

TIMM = "timm:"


def load(name: str, pretrained: bool = True) -> nn.Module:
    """The backbone ``name`` names, on CUDA if available. ``timm:<name>`` is a timm model
    with its classifier removed (``num_classes=0``), with ImageNet weights unless
    ``pretrained`` is false; any other name goes to histossl's ``load_encoder`` (a pathofm
    model, a histossl ``.ckpt`` or export folder), always pretrained."""
    if not name.startswith(TIMM):
        if not pretrained:
            raise ValueError(f"pretrained: false is for timm backbones, not {name!r}")
        return load_encoder(name)
    model = timm.create_model(name.removeprefix(TIMM), pretrained=pretrained, num_classes=0)
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def is_timm(backbone: nn.Module) -> bool:
    return hasattr(backbone, "pretrained_cfg")


def embedding(backbone: nn.Module, image: Tensor) -> Tensor:
    """Tile embedding: timm's pooled features, or for DINOv2 models the CLS token and mean
    patch token side by side (:func:`~pathotrain.pathofm.classification_embedding`)."""
    return backbone(image) if is_timm(backbone) else classification_embedding(backbone, image)


def embedding_dim(backbone: nn.Module) -> int:
    """Width of :func:`embedding`."""
    return backbone.num_features if is_timm(backbone) else 2 * backbone.config.hidden_size


def preprocessing(backbone: nn.Module) -> dict:
    """``{"input_size", "mean", "std"}`` the backbone expects; for timm, its pretrained
    config's (``crop_pct`` ignored: tiles are resized and centre cropped to the input)."""
    if not is_timm(backbone):
        return hf.preprocessing(backbone)
    data = timm.data.resolve_data_config({}, model=backbone)
    return {"input_size": data["input_size"][-1], "mean": data["mean"], "std": data["std"]}


def enable_grad_checkpointing(backbone: nn.Module) -> None:
    if is_timm(backbone):
        backbone.set_grad_checkpointing(True)
    else:
        backbone.gradient_checkpointing_enable({"use_reentrant": False})


@torch.no_grad()
def extract(
    backbone: nn.Module, dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> tuple[Tensor, Tensor]:
    """:func:`embedding` ``(N, D)`` float32 and labels ``(N,)`` of ``(image, label)``
    samples, on the backbone's device (bf16 autocast on CUDA), in order."""
    device = next(backbone.parameters()).device
    was_training = backbone.training
    backbone.eval()
    feats, labels = [], []
    # Fresh workers: forking a threaded (training) process can deadlock them.
    context = "forkserver" if num_workers else None
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, multiprocessing_context=context
    )
    for images, y in loader:
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            feats.append(embedding(backbone, images.to(device)).float())
        labels.append(y)
    backbone.train(was_training)
    return torch.cat(feats), torch.cat(labels)
