import pytest
import torch

from pathotrain.segmentation import CEDiceLoss, FocalDiceLoss, FocalLoss, build_model


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("smp:unet", {"encoder_name": "resnet18", "encoder_weights": None}),
        ("nnunet:plain", {"n_stages": 4}),
        ("nnunet:resenc", {"n_stages": 4}),
    ],
)
def test_build_model_outputs_logits_per_class(name, kwargs):
    model = build_model(name, num_classes=5, **kwargs).eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 64, 64)).shape == (2, 5, 64, 64)


def test_build_model_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("timm:unet", num_classes=2)
    with pytest.raises(ValueError, match="nnU-Net"):
        build_model("nnunet:transformer", num_classes=2)


def test_loss_ignores_255_and_survives_all_ignored_batch():
    loss = CEDiceLoss()
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8))
    ignored = target.clone()
    ignored[:, :4] = 255
    # Changing the logits under ignored pixels must not change the loss.
    other = logits.detach().clone()
    other[:, :, :4] += 10
    assert torch.isclose(loss(logits, ignored), loss(other, ignored))

    zero = loss(logits, torch.full_like(target, 255))
    zero.backward()
    assert zero.item() == 0 and torch.isfinite(logits.grad).all()


def test_focal_loss_matches_ce_at_gamma_0_and_ignores_255():
    logits = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 3, (2, 8, 8))
    target[:, :4] = 255
    weights = [0.5, 1.0, 2.0]
    ce = torch.nn.functional.cross_entropy(logits, target, ignore_index=255, reduction="sum")
    assert torch.isclose(FocalLoss(gamma=0)(logits, target), ce / (target != 255).sum())
    # Per-class alpha scales each pixel's term by its class weight.
    w = torch.tensor(weights)[target.clamp(max=2)][target != 255]
    per_px = torch.nn.functional.cross_entropy(logits, target, ignore_index=255, reduction="none")
    expected = (per_px[target != 255] * w).mean()
    assert torch.isclose(FocalLoss(gamma=0, class_weights=weights)(logits, target), expected)
    assert FocalLoss()(logits, target) < FocalLoss(gamma=0)(logits, target)
    assert FocalLoss()(logits, torch.full_like(target, 255)).item() == 0


def test_focal_dice_loss_adds_weighted_dice():
    import segmentation_models_pytorch as smp

    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8))
    target[:, :4] = 255
    dice = smp.losses.DiceLoss("multiclass", ignore_index=255)(logits, target)
    expected = FocalLoss()(logits, target) + 0.5 * dice
    assert torch.isclose(FocalDiceLoss(dice_weight=0.5)(logits, target), expected)
    zero = FocalDiceLoss()(logits, torch.full_like(target, 255))
    zero.backward()
    assert zero.item() == 0 and torch.isfinite(logits.grad).all()
