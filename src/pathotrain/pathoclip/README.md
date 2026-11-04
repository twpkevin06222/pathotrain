# pathotrain - pathoclip

Contrastive image-text fine-tuning of Google's SigLIP 2 on pathology captions,
for zero-shot tile classification and image-text retrieval. Trained on
pathotrain's engine (a `Task`, `experiment.fit`, `ModelCheckpoint`, MLflow / CSV),
like segmentation or histossl, not TRL. LoRA needs the `histossl` extra
(`pip install 'pathotrain[histossl]'`), whose linear probe pathoclip reuses.

## Models

| Name | Checkpoint | Licence |
|---|---|---|
| `siglip2-base-224` | `google/siglip2-base-patch16-224` (~375M parameters, both towers) | Apache-2.0 |
| `siglip2-so400m-224` | `google/siglip2-so400m-patch14-224` (~1.1B) | Apache-2.0 |

Revisions are pinned (`pathoclip/models.py`). Fixed 224 px input, the tile size
of the rest of pathotrain (the NaFlex variants need another processor). Images
are resized to 224 x 224 and normalised to `[-1, 1]` by the model's processor;
captions are tokenised (Gemma tokenizer) and padded to 64 tokens, longer ones
truncated: SigLIP's text tower pools the last position, so it must see them as
it was trained.

## Data

Caption JSONL (`pathotrain.dataloader.caption.CaptionDataset`):

```json
{"image": "images/3f2a.jpg", "caption": "Lung adenocarcinoma, acinar pattern.", "group": "fig12"}
```

`group` (optional, default the caption) names the pairs that share one caption,
such as the panels of one figure: they are positives for each other in the loss
and matches in retrieval.

ARCH (Gamper and Rajpoot, CVPR 2021; **CC BY-NC-SA 4.0, research use only**;
download `books_set.zip` and `pubmed_set.zip` from the
[Warwick TIA page](https://warwick.ac.uk/fac/cross_fac/tia/data/arch)) converts with:

```bash
python -m pathotrain.dataloader.caption arch ~/data/arch ~/data/arch
```

## Training

```bash
python -m pathotrain.pathoclip.train configs/siglip2-base.example.yaml
python -m pathotrain.pathoclip.train config.yaml --resume runs/<name>/checkpoints/last.ckpt
```

- **Loss:** SigLIP's sigmoid loss over the batch (every image against every
  caption) with the model's learnable temperature and bias (Zhai et al. 2023).
  Negatives are the other pairs of the batch only, so the batch size matters
  and gradient accumulation adds none; under DDP, other ranks' pairs are not
  used (`shortcut:` in `task.py`).
- **Adaptation:** LoRA (default rank 16 on the attention projections of both
  towers' blocks; `fc1` / `fc2` optional) with the logit scale and bias
  trained, or `full` fine-tuning. LoRA checkpoints keep only the trained
  weights.
- **Frozen pathology image tower (LiT, Zhai et al. 2022):** `model.image:
  midnight-12k` (any `pathofm` model) with `adaptation.towers: [text]`. The image
  embedding is the backbone's CLS + mean patch token through a trained linear
  projection to SigLIP's text width; SigLIP's own image tower is unused. Images
  are squashed to 224 x 224 and normalised as the backbone expects (Midnight:
  mean = std = 0.5), for captions and zero-shot tiles alike. The backbone embeds
  every train and val image once into `<output_dir>/image_features_{train,val}.pt`
  (reused on resume; delete them if the images change), so training runs the text
  tower only and batches of 512+ fit; no image augmentation. Checkpoints keep
  the trained weights only. `export` refuses these checkpoints: score the `.ckpt`
  with `eval`.
- **Validation** every epoch, on `data.val`: `val/loss`, `val/i2t_r1/5/10`,
  `val/t2i_r1/5/10` and `val/recall_mean`; with `eval`, zero-shot
  classification of labelled tiles (`zs/acc`, `zs/balanced_acc`, `zs/macro_f1`),
  each class's prompt from `eval.template` or `eval.prompts`, and with
  `eval.train` histossl's linear probe and kNN on the image embeddings
  (`probe/*`, `knn/*`).
- **Checkpoints:** `best.ckpt` by `zs/balanced_acc` with `eval`, else
  `val/recall_mean`; `last.ckpt` to resume. Both store the config and the
  provenance (base model and revision, adaptation, input size, normalisation,
  text length, git commit).

## Evaluation and export

```bash
python -m pathotrain.pathoclip.eval siglip2-base-224 --captions arch/test.jsonl --test crc/CRC-VAL-HE-7K
python -m pathotrain.pathoclip.eval runs/<name>/checkpoints/best.ckpt --captions arch/test.jsonl
python -m pathotrain.pathoclip.export runs/<name>/checkpoints/best.ckpt exported/<name>
```

`eval` scores a registry model (the zero-shot baseline), a checkpoint or an
export; it uses `eval.template` only (no per-class prompts). `export` writes the
model with LoRA merged, its processor and `pathotrain.json`; it loads with
transformers alone (`AutoModel`, `AutoProcessor`).

## Results (2026-11-04)

Full ARCH (`books_set` + `pubmed_set`, `export_arch`'s split): 6,087 train, 746
val and 746 test images (657 test captions). Zero-shot on CRC-VAL-HE-7K (7,180
tiles, 9 classes) with the example configs' prompts. Seed 0, no hyperparameter
search; `best.ckpt` chosen by `val/recall_mean` on ARCH val, so CRC stays held
out. One RTX 5090.

| Model | ARCH test i2t R@1 / R@10 | t2i R@1 / R@10 | recall_mean | CRC zs acc | CRC zs balanced acc |
|---|---|---|---|---|---|
| `siglip2-base-224` zero-shot | 0.021 / 0.090 | 0.025 / 0.091 | 0.058 | 0.451 | 0.376 |
| A: LoRA, both towers (epoch 9 of 10) | 0.055 / 0.283 | 0.068 / 0.327 | 0.189 | 0.307 | 0.299 |
| B: LoRA, text tower only (epoch 8 of 10) | 0.038 / 0.189 | 0.040 / 0.235 | 0.128 | 0.275 | 0.283 |
| C: frozen Midnight-12k + projection, LoRA text tower (epoch 30 of 60) | **0.119 / 0.371** | **0.147 / 0.430** | **0.279** | 0.305 | 0.371 |
| C, 15 epochs (epoch 13 of 15) | 0.068 / 0.314 | 0.084 / 0.412 | 0.229 | 0.408 | **0.459** |

| Arm | Config | lr / batch / epochs | Time | Peak GPU memory |
|---|---|---|---|---|
| A | `configs/siglip2-base.example.yaml` | 1e-4 / 64 / 10 | 9.5 min | 8.4 GB |
| B | the same with `adaptation.towers: [text]` | 1e-4 / 64 / 10 | 10.5 min | 4.0 GB |
| C | `configs/midnight-lit.example.yaml` with 60 epochs, `val_every: 5` | 5e-4 / 512 / 60 | 17.4 min (incl. ~1 min caching) | 18.0 GB |
| C, 15 epochs | `configs/midnight-lit.example.yaml` | 5e-4 / 512 / 15 | 18.1 min (CRC zero-shot every epoch) | 18.0 GB |

Times are from before validation workers persisted across epochs (2026-11-04):
each epoch restarted the validation and zero-shot loaders' workers, ~20 s each
with forkserver. An arm A epoch now takes ~14 s instead of ~53 s (training MFU
0.31 instead of 0.07 on an RTX 5090, `peak_tflops` 209.5); its first epoch still
pays the one-time worker start.

- Training the SigLIP 2 towers on ARCH triples retrieval (A) but costs CRC
  zero-shot accuracy, even with the image tower frozen (B): the text tower drifts
  toward ARCH's caption style.
- C, a frozen Midnight-12k image tower (LiT), retrieves best (recall_mean x1.5 A)
  and keeps CRC zero-shot at the base model's level. Alignment from a random
  projection on ~6k pairs was not the problem it could have been.
- C overfits after ~10-15 epochs: val loss rises 7.3 -> 14.8 while val recall
  stays at 0.25-0.26, and CRC zero-shot balanced accuracy peaked at 0.468 at
  epoch 10 before falling to ~0.35.
- Stopped at 15 epochs, C is the only arm that beats the base model on CRC
  zero-shot (balanced accuracy 0.459 vs 0.376, +8 points), and still retrieves
  better than A (recall_mean 0.229 vs 0.189). The 60-epoch run retrieves best
  (0.279) but loses the zero-shot gain: the epoch count trades one for the
  other. The example config uses 15.
- C's settings differ from A / B (its projection starts from random weights,
  and cached features make large batches cheap), so the arms are not a
  controlled comparison.

## Tests

```bash
uv run pytest packages/pathotrain/tests/test_pathoclip.py
```

On a tiny random SigLIP with the real SigLIP 2 processor (from the Hugging Face
cache, else skipped): retrieval with shared captions, the loss against
transformers', a LoRA run end to end (metrics, checkpoint contents, MLflow),
reload, export equal to the checkpoint, and the eval CLI on the export; a frozen
image tower run (a tiny DINOv2 in place of Midnight) end to end with its cached
features; both example configs valid.
