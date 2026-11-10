import pytest
import torch
from transformers import Sam3ViTConfig, Sam3ViTModel

from pathotrain.cellseg import CellInstanceTask, CellViT
from pathotrain.pathofm import classification_embedding
from pathotrain.pathofm.sam3 import Sam3Backbone
from pathotrain.segmentation import build_model
from pathotrain.segmentation.encoders import token_grids

# SAM 3's image encoder ViT, tiny: windowed attention with global layers 2 and 4 (1-based).
CONFIG = dict(  # noqa: C408 (Sam3ViTConfig keywords)
    hidden_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=64,
    global_attn_indexes=[1, 3],
    window_size=4,
    image_size=112,  # an 8 x 8 grid, as SAM 3's 1008 px is 72 x 72
    pretrain_image_size=56,
)


def _sam3():
    torch.manual_seed(0)
    return Sam3Backbone(Sam3ViTModel(Sam3ViTConfig(**CONFIG)).eval())


def test_native_grid_matches_the_plain_vit_and_other_grids_run():
    backbone = _sam3()
    torch.manual_seed(0)
    reference = Sam3ViTModel(Sam3ViTConfig(**CONFIG)).eval()  # the same weights, unwrapped
    x = torch.rand(2, 3, 112, 112)
    with torch.no_grad():
        ours = backbone(pixel_values=x, output_hidden_states=True)
        plain = reference(pixel_values=x, output_hidden_states=True)
        assert len(ours.hidden_states) == 5 and ours.hidden_states[1].shape == (2, 64, 32)
        torch.testing.assert_close(ours.last_hidden_state, plain.last_hidden_state)
        torch.testing.assert_close(ours.hidden_states[2], plain.hidden_states[2].flatten(1, 2))
        # A 16 x 12 grid: the global layers' RoPE positions follow the input.
        assert backbone(pixel_values=torch.rand(1, 3, 224, 168)).last_hidden_state.shape == (
            1,
            16 * 12,
            32,
        )
        torch.testing.assert_close(
            backbone(pixel_values=x).last_hidden_state, plain.last_hidden_state
        )


def test_token_grids_take_sam3_tokens_without_a_prefix():
    backbone, x = _sam3(), torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        (grid,) = token_grids(backbone, x, (4,))
    assert grid.shape == (1, 32, 4, 4)


def test_sam3_refuses_classification_embedding():
    with pytest.raises(ValueError, match="no CLS token"):
        classification_embedding(_sam3(), torch.rand(1, 3, 56, 56))


def _trainable(model, prefix):
    return {n for n, p in model.named_parameters() if p.requires_grad and n.startswith(prefix)}


def test_lora_trains_adapters_and_decoder_only_in_a_segmentation_encoder():
    pytest.importorskip("peft")
    lora = {"type": "lora", "rank": 2, "targets": ["q_proj", "v_proj"]}
    model = build_model(
        "smp:unet",
        3,
        encoder_name=_sam3(),
        encoder_kwargs={"channels": (8, 8, 8, 8), "adaptation": lora},
        decoder_channels=(16, 8, 8, 8, 8),
    )
    backbone = _trainable(model, "encoder.backbone")
    assert backbone and all(".lora_" in n for n in backbone)
    assert _trainable(model, "decoder")
    out = model(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 3, 64, 64)
    out.mean().backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if ".lora_B" in n)


def test_cellvit_on_sam3_with_lora_trains_the_instance_task():
    pytest.importorskip("peft")
    model = CellViT(
        _sam3(),
        num_classes=3,
        skip_dims=(16, 8),
        bottleneck_dim=16,
        instance=True,
        adaptation={"type": "lora", "rank": 2},
    )
    assert all(".lora_" in n for n in _trainable(model, "backbone"))
    task = CellInstanceTask(model, 3)
    inst = torch.zeros(1, 64, 64, dtype=torch.long)
    inst[0, 10:20, 10:20] = 1
    target = {"inst": inst, "type": inst * 2, "hv": torch.zeros(1, 2, 64, 64)}
    loss, _ = task.training_step((torch.rand(1, 3, 64, 64), target))
    loss.backward()
    assert any(p.grad is not None for n, p in model.named_parameters() if ".lora_B" in n)


def test_full_and_frozen_adaptations():
    full = CellViT(_sam3(), 3, skip_dims=(16, 8), bottleneck_dim=16, adaptation={"type": "full"})
    assert not full.freeze and all(p.requires_grad for p in full.backbone.parameters())
    frozen = CellViT(
        _sam3(), 3, skip_dims=(16, 8), bottleneck_dim=16, adaptation={"type": "frozen"}
    )
    assert frozen.freeze and not any(p.requires_grad for p in frozen.backbone.parameters())


def test_onnx_export_of_a_lora_sam3_segmentation_model(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("peft")
    from pathotrain.experiment import export_onnx

    model = build_model(
        "smp:unet",
        3,
        encoder_name=_sam3(),
        encoder_kwargs={"channels": (8, 8, 8, 8), "adaptation": {"type": "lora", "rank": 2}},
        decoder_channels=(16, 8, 8, 8, 8),
    )
    export_onnx(model, tmp_path / "m.onnx", 64, ["logits"], {})  # checks ORT against PyTorch
