# pathotrain - pathofm

> [!WARNING]
> For research use only. Not a medical device.

Pathology foundation models (FMs) as backbones for downstream tasks:
segmentation, linear probing, fine-tuning and multiple-instance learning (MIL)
on whole slides.

Only open models are included. Both are MIT licensed and not gated on Hugging
Face, so they need no login and can be used commercially, subject to the MIT
terms. Models with non-commercial licences, or that require access approval,
are deliberately left out.

| Name | Weights | What it is |
|---|---|---|
| `midnight-12k` | [kaiko-ai/midnight](https://huggingface.co/kaiko-ai/midnight) | Midnight trained on 12k public slides |
| `spectra-midnight-12k` | [medarc/Midnight-12k-SPECTRA](https://huggingface.co/medarc/Midnight-12k-SPECTRA) | Midnight-12k adapted with SPECTRA for robustness to scanner, stain and centre |

Both share one architecture:
- DINOv2 ViT-g/14, loaded as `transformers.Dinov2Model`;
- 1.1B parameters in 40 layers;
- 1536-dim tokens;
- patch size 14, one CLS token, no registers.

The revisions are pinned in `FOUNDATION_MODELS`, so the weights never change
under a run.

## Loading

```python
from pathotrain.pathofm import build_encoder, load_backbone

backbone = load_backbone("midnight-12k")                # Dinov2Model, fp32
backbone, transform = build_encoder("spectra-midnight-12k")  # + model-card preprocessing
```

- **Download:** the first call downloads the weights (about 4.5 GB per model)
  into the Hugging Face cache, `~/.cache/huggingface` by default. Set `HF_HOME`
  to put the cache elsewhere.
- **Normalisation:** both models use mean = std = 0.5 (`MIDNIGHT_MEAN`,
  `MIDNIGHT_STD`), not ImageNet statistics. Using the wrong normalisation
  silently degrades every downstream result.
- **`transform`:** the model card's preprocessing for a PIL tile. It does a
  bilinear resize to 224, a centre crop to 224, `ToTensor` and the 0.5
  normalisation.
- **Inference:** run the frozen backbone under
  `torch.inference_mode()` and `torch.autocast("cuda", torch.bfloat16)`.
  bf16 halves the memory and is how the model cards run it.

## Which output to use

The model card defines two embeddings, and they are not interchangeable.

| Function | Shape | Built from | Use for |
|---|---|---|---|
| `classification_embedding(model, x)` | `(B, 3072)` | CLS token concatenated with the mean patch token | tile classification, linear probing, MIL, retrieval |
| `segmentation_embedding(model, x)` | `(B, 1536, H/14, W/14)` | last-layer patch tokens, after the final LayerNorm | dense tasks: segmentation, detection heads |

```python
import torch
from pathotrain.pathofm import classification_embedding, segmentation_embedding

with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
    vec = classification_embedding(backbone, batch)  # (B, 3072), one vector per tile
    grid = segmentation_embedding(backbone, batch)   # (B, 1536, 16, 16) for 224 px tiles
```

- **Intermediate layers:** for other layers, call
  `backbone(pixel_values=x, output_hidden_states=True).hidden_states`. Index
  `i` is the output of layer `i` before the final LayerNorm; index 0 is the
  patch embedding.
- **Other tile sizes:** the position embeddings are interpolated, so other
  sizes work, but the side should be a multiple of 14. Check the model card for
  the resolution the model was trained at, and tile at that resolution.

## Downstream tasks

### Segmentation backbone

This is built in: `pathotrain.segmentation` wraps the FM as an encoder for SMP
decoders (U-Net, FPN, UPerNet, DeepLabV3+, ...).

```python
from pathotrain.segmentation import build_model

model = build_model("smp:unet", num_classes=6, encoder_name="midnight-12k")
```

See [segmentation/README.md](../segmentation/README.md) for how the encoder is
built and for its options (`layers`, `channels`, `freeze`).

### Feature extraction (shared by linear probing and MIL)

The backbone is the expensive part. When it stays frozen, embed every tile
once and train later models on the saved features.

```python
import torch
from pathotrain.pathofm import classification_embedding, load_backbone

backbone = load_backbone("midnight-12k").cuda().eval()


@torch.inference_mode()
def embed_tiles(loader):
    """loader yields (tiles, coords): tiles normalised with 0.5 mean/std."""
    feats, xy = [], []
    for tiles, coords in loader:
        with torch.autocast("cuda", torch.bfloat16):
            feats.append(classification_embedding(backbone, tiles.cuda()).float().cpu())
        xy.append(coords)
    return torch.cat(feats), torch.cat(xy)


# One file per slide: a "bag" of tile embeddings and their level-0 coordinates.
features, coords = embed_tiles(slide_loader)
torch.save({"features": features, "coords": coords}, "features/slide_name-xx.pt")
```

Tile each slide with patchology (e.g. its tile generator) at the resolution
the model expects.

### Linear probing

Linear probing measures how good the frozen features are: train a single
linear layer on the saved embeddings, and leave the backbone untouched. It's
the cheapest way to compare `midnight-12k` with `spectra-midnight-12k` on your
data.

```python
import torch
from torch import nn

x_train, y_train = ...  # (N, 3072) float embeddings, (N,) int64 labels
probe = nn.Linear(3072, num_classes)
optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
for epoch in range(100):
    loss = nn.functional.cross_entropy(probe(x_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

- **Standardise the features first:** use the mean and std of the training set.
- **Choose the regularisation on a validation split.**
- **Logistic regression works too:** `sklearn.linear_model.LogisticRegression`
  is an equivalent choice on these embeddings.

### Fine-tuning

Fine-tune when frozen features are not enough, e.g. under a large domain
shift. In increasing order of cost:

1. **Partial:** unfreeze only the last few transformer blocks.

   ```python
   backbone.requires_grad_(False)
   for block in backbone.encoder.layer[-4:]:
       block.requires_grad_(True)
   backbone.layernorm.requires_grad_(True)
   ```

2. **LoRA:** train low-rank adapters on the attention and MLP layers (with
   [peft](https://github.com/huggingface/peft), installed separately). This is
   how SPECTRA adapted Midnight-12k. It costs little memory, and the adapters
   can be merged back into the weights.
3. **Full:** unfreeze all 1.1B parameters.
   - Use `backbone.gradient_checkpointing_enable()`, bf16 and a small learning
     rate (around 1e-5).
   - Use a lower rate for the backbone than for the head.
   - Expect several times the memory of frozen training.

For segmentation, the encoder exposes this as `encoder_kwargs={"freeze": False}`.

### Self-supervised (SSL) continued pre-training

Continued pre-training adapts the FM to your own unlabelled slides before any
supervised task. The usual recipe is to continue DINOv2 training from these
weights on tiles from your domain, at a low learning rate.

pathotrain does not provide an SSL trainer yet; `histossl/` is planned. Until
then, use the [DINOv2 code](https://github.com/facebookresearch/dinov2) or
another DINOv2 trainer. Checkpoints saved in the transformers format then load
with `Dinov2Model.from_pretrained(path)`.

### Multiple-instance learning (slide-level labels)

When labels exist per slide rather than per tile, treat each slide as a bag of
tile embeddings (from the feature extraction above) and learn to pool them.
Attention-based MIL (ABMIL; Ilse et al., ICML 2018) is a strong, simple
baseline:

```python
import torch
from torch import nn


class ABMIL(nn.Module):
    """Gated attention pooling over the tiles of one slide."""

    def __init__(self, dim=3072, hidden=256, num_classes=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.attn_v = nn.Linear(hidden, hidden)
        self.attn_u = nn.Linear(hidden, hidden)
        self.attn_w = nn.Linear(hidden, 1)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, bag):  # bag: (N_tiles, dim)
        h = self.proj(bag)
        a = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h)))
        a = torch.softmax(a, dim=0)  # (N_tiles, 1); also a heatmap over the slide
        return self.head((a * h).sum(0)), a


model = ABMIL(num_classes=2)
bag = torch.load("features/slide_name-xx.pt")["features"]
logits, attention = model(bag)
```

- **One bag per step:** use batch size 1 (bags differ in size), or pad and mask
  the bags.
- **Heatmaps:** the attention weights, placed at the saved `coords`, give a
  heatmap of the tiles that drove the prediction.
- **Splits:** split by patient, not by slide or tile. Tiles from one patient
  on both sides of a split inflate every metric.

### With the training engine

Any of these models can be trained with `pathotrain.engine.Trainer`, which
adds DDP, bf16, checkpoints, MLflow and CSV logs, and Optuna sweeps. Wrap the
model in a `Task` with these four methods:

- `training_step(batch)`;
- `validation_step(batch)`;
- `compute_metrics()`;
- `configure_optimizers(total_steps)`.

`SegmentationTask` in `pathotrain.segmentation` is a complete example of the
contract.

## License

The code is Apache License 2.0. Both sets of model weights are MIT; see each
model card.
