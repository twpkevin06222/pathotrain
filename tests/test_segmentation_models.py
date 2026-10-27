import pytest
import torch

from pathotrain.segmentation import CEDiceLoss, build_model


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
