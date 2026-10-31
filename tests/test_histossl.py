import functools
import pickle

import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from pathotrain.histossl import (
    apply_adaptation,
    block_mask,
    build_backbone,
    normalisation,
    param_groups,
    random_mask,
    tokens,
)


def _tiny(depth=3):
    config = Dinov2Config(
        hidden_size=32,
        num_hidden_layers=depth,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=14,
        image_size=28,
    )
    return Dinov2Model(config)


def _trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_scratch_backbone_masks_tokens():
    torch.manual_seed(0)
    model = build_backbone("scratch", arch="vit_s14").eval()
    assert model.config.hidden_size == 384 and model.config.patch_size == 14
    assert model.is_gradient_checkpointing
    x = torch.rand(2, 3, 28, 28)  # a 2 x 2 grid
    cls, patches = tokens(model, x)
    assert cls.shape == (2, 384) and patches.shape == (2, 4, 384)
    mask = torch.tensor([[True, False, False, False], [False] * 4])
    masked_cls, _ = tokens(model, x, mask)
    assert not torch.allclose(masked_cls[0], cls[0]) and torch.allclose(masked_cls[1], cls[1])
    assert normalisation("scratch") == ((0.5,) * 3, (0.5,) * 3)
    assert normalisation("pretrained", "midnight-12k") == ((0.5,) * 3, (0.5,) * 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"init": "pretrained", "id": "unknown"},
        {"init": "pretrained", "id": "midnight-12k", "arch": "vit_b14"},
        {"init": "scratch", "arch": "vit_x14"},
        {"init": "scratch", "arch": "vit_b14", "id": "midnight-12k"},
        {"init": "random"},
    ],
)
def test_backbone_rejects_bad_names(kwargs):
    with pytest.raises(ValueError):
        build_backbone(**kwargs)


def test_adaptations_set_the_trainable_weights():
    model = _tiny()
    assert _trainable(apply_adaptation(model, "frozen")) == set()
    assert len(_trainable(apply_adaptation(model, "full"))) == len(list(model.parameters()))
    last = _trainable(apply_adaptation(model, "last_n", n=1))
    assert last and all(n.startswith(("encoder.layer.2.", "layernorm.")) for n in last)
    for n in (0, 4, None):
        with pytest.raises(ValueError, match="last_n"):
            apply_adaptation(model, "last_n", n=n)
    with pytest.raises(ValueError, match="adaptation"):
        apply_adaptation(model, "bitfit")


def test_lora_trains_adapters_and_disables_to_the_base():
    pytest.importorskip("peft")
    torch.manual_seed(0)
    base = _tiny()
    x = torch.rand(2, 3, 28, 28)
    with torch.no_grad():
        expected = tokens(base.eval(), x)[1]
    base.gradient_checkpointing_enable({"use_reentrant": False})
    model = apply_adaptation(base, "lora", rank=4, targets=["q_proj", "v_proj", "fc1"])
    trainable = _trainable(model)
    assert trainable and all("lora_" in n for n in trainable)
    assert any("fc1.lora_A" in n for n in trainable)
    for name, p in model.named_parameters():
        if "lora_B" in name:  # start from zero; move off it so the adapters change outputs
            p.data.normal_()
    model.train()
    tokens(model, x)[1].sum().backward()  # gradients reach the adapters through checkpointing
    assert all(p.grad is not None for n, p in model.named_parameters() if "lora_" in n)
    model.eval()
    with torch.no_grad():
        assert not torch.allclose(tokens(model, x)[1], expected, atol=1e-4)
        with model.disable_adapter():  # the frozen teacher of teacher_jepa
            assert torch.allclose(tokens(model, x)[1], expected, atol=1e-5)
    with pytest.raises(ValueError, match="targets"):
        apply_adaptation(_tiny(), "lora", targets=["query"])
    with pytest.raises(ValueError, match="rank"):
        apply_adaptation(_tiny(), "lora", rank=0)


def test_param_groups_skip_decay_on_norms_and_tokens_and_decay_lr_by_layer():
    model = apply_adaptation(_tiny(depth=2), "full")
    groups = param_groups(model, lr=1.0, weight_decay=0.05, layer_decay=0.5)
    names = {id(p): n for n, p in model.named_parameters()}
    seen = {names[id(p)]: g for g in groups for p in g["params"]}
    assert len(seen) == len(names)  # every trainable weight, once
    assert seen["embeddings.cls_token"]["weight_decay"] == 0
    assert seen["embeddings.mask_token"]["weight_decay"] == 0
    assert seen["encoder.layer.0.norm1.weight"]["weight_decay"] == 0
    assert seen["encoder.layer.0.attention.q_proj.weight"]["weight_decay"] == 0.05
    assert seen["encoder.layer.0.attention.q_proj.weight"]["lr"] == 0.5**2
    assert seen["encoder.layer.1.mlp.fc1.weight"]["lr"] == 0.5
    assert seen["embeddings.patch_embeddings.projection.weight"]["lr"] == 0.5**3
    assert seen["layernorm.weight"]["lr"] == 1.0
    last = apply_adaptation(_tiny(), "last_n", n=1)
    groups = param_groups(last, lr=1.0, weight_decay=0.1)  # frozen weights left out
    assert all(g["lr"] == 1.0 for g in groups)
    assert sum(len(g["params"]) for g in groups) == len(_trainable(last))


def test_masks_hit_the_ratio_and_are_deterministic():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        mask = block_mask(16, (0.1, 0.5), generator=g)
        assert mask.shape == (256,) and mask.dtype == torch.bool
        assert 26 <= int(mask.sum()) <= 128
    assert int(block_mask(16, 0.4).sum()) == round(0.4 * 256)
    assert int(random_mask(16, 0.25).sum()) == 64
    assert not block_mask(16, 0.4, p=0.0).any()
    first = block_mask(16, 0.3, generator=torch.Generator().manual_seed(1))
    assert torch.equal(first, block_mask(16, 0.3, generator=torch.Generator().manual_seed(1)))
    # A partial pickles, so DataLoader workers can call it.
    pickle.loads(pickle.dumps(functools.partial(block_mask, ratio=(0.1, 0.5))))()
    with pytest.raises(ValueError, match="ratio"):
        random_mask(16, 1.5)
