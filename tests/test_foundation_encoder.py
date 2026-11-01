import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Dinov2Config, Dinov2Model

from pathotrain.pathofm import (
    classification_embedding,
    from_timm_dinov2,
    load_backbone,
    segmentation_embedding,
)
from pathotrain.segmentation import CEDiceLoss, ViTPyramidEncoder, build_model

# Same architecture family as Midnight-12k (DINOv2, patch 14, SwiGLU, no registers), tiny.
CONFIG = Dinov2Config(
    hidden_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    patch_size=14,
    image_size=224,
    use_swiglu_ffn=True,
)
CHANNELS = {"channels": (8, 16, 32, 32)}


def _backbone():
    torch.manual_seed(0)
    return Dinov2Model(CONFIG).eval()


def test_card_embeddings_have_card_shapes():
    model, x = _backbone(), torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        assert classification_embedding(model, x).shape == (2, 64)  # CLS + mean patch
        assert segmentation_embedding(model, x).shape == (2, 32, 16, 16)  # 224 / 14


def test_last_layer_feature_is_the_card_segmentation_embedding():
    encoder = ViTPyramidEncoder(_backbone(), layers=(4, 4, 4, 4), **CHANNELS)
    encoder.resample[2] = nn.Identity()  # the stride-16 feature without its projection
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        feature = encoder(x)[4]
        card = segmentation_embedding(
            encoder.backbone, F.interpolate(x, size=(224, 224), mode="bilinear")
        )
    torch.testing.assert_close(feature, card)


def test_timm_dinov2_with_registers_converts_exactly():
    timm = pytest.importorskip("timm")
    torch.manual_seed(0)
    # OpenMidnight's architecture, tiny: 2 heads of 64, SwiGLU 344 wide (HF's for 128).
    ref = timm.create_model(
        "vit_giant_patch14_reg4_dinov2",
        embed_dim=128,
        depth=2,
        num_heads=2,
        mlp_ratio=688 / 128,
        img_size=224,
        pretrained=False,
    ).eval()
    for p in ref.parameters():  # LayerScale starts at 1e-5: make every weight count
        p.data.normal_(0, 0.1)
    model = from_timm_dinov2({k: v.clone() for k, v in ref.state_dict().items()})
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        want = ref.forward_features(x)
        torch.testing.assert_close(model(pixel_values=x).last_hidden_state, want)
        emb = classification_embedding(model, x)  # registers (tokens 1-4) left out
        torch.testing.assert_close(emb, torch.cat([want[:, 0], want[:, 5:].mean(1)], -1))
        assert segmentation_embedding(model, x).shape == (2, 128, 16, 16)


@pytest.mark.parametrize(
    "arch",
    [
        "unet",
        "unetplusplus",
        "manet",
        "linknet",
        "fpn",
        "pspnet",
        "pan",
        "deeplabv3",
        "deeplabv3plus",
        "upernet",
        "segformer",
    ],
)
def test_every_smp_decoder_accepts_the_foundation_encoder(arch):
    model = build_model(f"smp:{arch}", 5, encoder_name=_backbone(), encoder_kwargs=CHANNELS)
    with torch.no_grad():
        assert model.eval()(torch.rand(2, 3, 256, 256)).shape == (2, 5, 256, 256)


@pytest.mark.parametrize("freeze", [True, False])
def test_freeze_trains_only_the_pyramid_and_decoder(freeze):
    encoder_kwargs = {**CHANNELS, "freeze": freeze}
    model = build_model("smp:unet", 3, encoder_name=_backbone(), encoder_kwargs=encoder_kwargs)
    target = torch.randint(0, 3, (2, 64, 64))
    CEDiceLoss()(model(torch.rand(2, 3, 64, 64)), target).backward()
    backbone_grads = [p.grad for p in model.encoder.backbone.parameters()]
    assert all(g is None for g in backbone_grads) is freeze
    assert all(p.grad is not None for p in model.encoder.resample.parameters())


def test_bad_layers_and_names_are_rejected():
    with pytest.raises(ValueError, match="layers"):
        ViTPyramidEncoder(_backbone(), layers=(1, 2, 3, 5), **CHANNELS)
    with pytest.raises(ValueError, match="unknown foundation model"):
        load_backbone("uni2-h")
