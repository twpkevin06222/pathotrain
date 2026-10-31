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


def test_lora_targets_must_exist_in_the_model():
    pytest.importorskip("peft")
    from pathotrain.histossl.adaptation import block_linears

    # Midnight (DINOv2 ViT-g) has a SwiGLU MLP: no fc1 / fc2.
    swiglu = Dinov2Model(
        Dinov2Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            patch_size=14,
            image_size=28,
            use_swiglu_ffn=True,
        )
    )
    assert block_linears(swiglu) == [
        "down_proj",
        "gate_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ]
    with pytest.raises(ValueError, match="\\['fc1'\\] are not linear layers"):
        apply_adaptation(swiglu, "lora", targets=["q_proj", "fc1"])  # peft would skip fc1
    model = apply_adaptation(swiglu, "lora", rank=2, targets=["q_proj", "gate_proj", "down_proj"])
    adapted = {n.split(".lora_")[0].rsplit(".", 1)[-1] for n in _trainable(model)}
    assert adapted == {"q_proj", "gate_proj", "down_proj"}
    assert "fc1" in block_linears(_tiny())  # a plain MLP has them


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


def test_linear_probe_and_knn_separate_easy_features():
    from pathotrain.histossl.eval import knn, linear_probe

    g = torch.Generator().manual_seed(0)
    centres = torch.randn(3, 16, generator=g) * 3

    def sample(n):
        y = torch.arange(n) % 3
        return centres[y] + torch.randn(n, 16, generator=g), y

    (train_x, train_y), (test_x, test_y) = sample(90), sample(30)
    probe = linear_probe(train_x, train_y, test_x, test_y, 3)
    assert probe["probe/balanced_acc"] > 0.95 and probe["probe/auroc"] > 0.99
    assert probe["probe/weight_decay"] in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    assert linear_probe(train_x, train_y, test_x, test_y, 3) == probe  # deterministic
    near = knn(train_x, train_y, test_x, test_y, 3, k=5)
    assert near["knn/acc"] > 0.95 and set(near) == {
        "knn/acc",
        "knn/balanced_acc",
        "knn/macro_f1",
        "knn/auroc",
    }
    noise = linear_probe(torch.randn(90, 16), train_y, torch.randn(30, 16), test_y, 3)
    assert noise["probe/balanced_acc"] < 0.7  # nothing to learn


def test_evaluate_features_runs_on_class_tiles(tmp_path):
    import numpy as np
    from PIL import Image

    from pathotrain.dataloader import ClassTileDataset
    from pathotrain.histossl.eval import evaluate_features
    from pathotrain.histossl.eval.linear_probe import probe_transform

    rng = np.random.default_rng(0)
    for split, n in (("train", 6), ("test", 4)):
        for label, colour in (("a", 40), ("b", 220)):
            (tmp_path / split / label).mkdir(parents=True)
            for i in range(n):
                tile = np.clip(rng.normal(colour, 10, (32, 32, 3)), 0, 255).astype(np.uint8)
                Image.fromarray(tile).save(tmp_path / split / label / f"{i}.png")
    transform = probe_transform((0.5,) * 3, (0.5,) * 3, size=28)
    train = ClassTileDataset(tmp_path / "train", transform)
    test = ClassTileDataset(tmp_path / "test", transform, class_names=train.class_names)
    torch.manual_seed(0)
    metrics = evaluate_features(_tiny(), train, test, 2, k=3, batch_size=4, num_workers=0)
    assert metrics["probe/acc"] == 1.0 and metrics["knn/acc"] == 1.0  # dark vs bright
