import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from pathotrain.cellseg import CellViT
from pathotrain.segmentation import SegmentationTask, build_model
from pathotrain.segmentation.task import input_multiple

TINY_VIT = Dinov2Config(
    hidden_size=32, num_hidden_layers=4, num_attention_heads=4, patch_size=14, image_size=224
)


def _cellvit(**kwargs):
    torch.manual_seed(0)
    return CellViT(
        Dinov2Model(TINY_VIT), num_classes=6, skip_dims=(16, 8), bottleneck_dim=16, **kwargs
    )


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("cellseg:hovernet", {"encoder_weights": None}),
        ("cellseg:hovernext", {"encoder": "convnextv2_atto"}),
    ],
)
def test_cnn_cell_models_give_full_resolution_logits(name, kwargs):
    model = build_model(name, num_classes=6, **kwargs).eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 64, 64)).shape == (2, 6, 64, 64)
    assert input_multiple(model) in (8, 32)


def test_cellvit_gives_full_resolution_logits_and_trains_only_the_decoder():
    model = _cellvit().eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 64, 96)).shape == (2, 6, 64, 96)
    assert input_multiple(model) == 16
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for n, p in model.named_parameters() if not n.startswith("backbone"))


def test_cellvit_layers_are_checked():
    with pytest.raises(ValueError, match="four indices"):
        _cellvit(layers=(1, 2, 3, 5))


def test_cell_model_trains_and_validates_in_the_segmentation_task():
    task = SegmentationTask(_cellvit(freeze=False), num_classes=6)
    image, mask = torch.rand(2, 3, 32, 32), torch.randint(0, 6, (2, 32, 32))
    loss, _ = task.training_step((image, mask))
    loss.backward()
    assert next(task.model.backbone.parameters()).grad is not None
    with torch.no_grad():
        task.validation_step((torch.rand(1, 3, 40, 40), torch.randint(0, 6, (1, 40, 40))))
    assert "val/miou" in task.compute_metrics()


def test_unknown_cell_model_is_rejected():
    with pytest.raises(ValueError, match="cell segmentation model"):
        build_model("cellseg:stardist", num_classes=2)
