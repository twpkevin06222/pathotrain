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

## First numbers (2026-11-03)

ARCH `pubmed_set` only (the `books_set` download was not finished): 2,663
train, 332 val, 314 test pairs from `export_arch`'s split. Zero-shot on
CRC-VAL-HE-7K (9 classes) with the example config's prompts.

| Model | ARCH test i2t R@1 / R@10 | t2i R@1 / R@10 | CRC zs acc | CRC zs balanced acc |
|---|---|---|---|---|
| `siglip2-base-224` zero-shot | 0.025 / 0.140 | 0.032 / 0.153 | 0.451 | 0.376 |

Smoke run (example config, LoRA rank 16, batch 64, 2 epochs, RTX 5090): 141 s,
8.3 GB peak, a 9.5 MB `best.ckpt`. On ARCH val, `val/recall_mean` went from 0.090
to 0.100 and `val/loss` from 5.20 to 5.04. CRC zero-shot balanced accuracy stayed
flat (0.374). Two epochs on 2.7k pairs is a check that the run works, not a
result: no full run, no test score of a fine-tuned model yet.

## Tests

```bash
uv run pytest packages/pathotrain/tests/test_pathoclip.py
```

On a tiny random SigLIP with the real SigLIP 2 processor (from the Hugging Face
cache, else skipped): retrieval with shared captions, the loss against
transformers', a LoRA run end to end (metrics, checkpoint contents, MLflow),
reload, export equal to the checkpoint, and the eval CLI on the export.
