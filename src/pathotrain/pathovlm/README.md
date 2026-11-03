# pathotrain - pathovlm

> [!WARNING]
> For research use only. Not a medical device.

Fine-tuning vision-language models (VLMs) for pathology visual question
answering (VQA): an image, a question about it, a short answer. Training will
use [TRL](https://github.com/huggingface/trl)'s `SFTTrainer` with LoRA adapters
through peft, configured from one YAML file and logged to MLflow and CSV like
the other modules.

> [!NOTE]
> In development. Done: the VQA dataset and PathVQA converter, the model
> registry, the training / generation collator, zero-shot and adapter
> evaluation, LoRA fine-tuning with TRL, and full runs of Qwen3-VL 2B and 4B on
> PathVQA ([results](#lora-fine-tuning-on-pathvqa)), and export of the merged
> model. Next: quantized exports.

```bash
pip install -e ".[pathovlm]"   # trl (pinned), peft, num2words (SmolVLM's processor needs it)
```

## Models

| Name | Weights | Size (bf16) | Image input |
|---|---|---|---|
| `qwen3-vl-2b` | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | ~4.5 GB | native resolution, at most 1024 visual tokens (one per 32 x 32 px) |
| `qwen3-vl-4b` | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | ~9 GB | as above |
| `smolvlm2-2.2b` | [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | ~4.5 GB | one 384 px image (~100 tokens) |
| `smolvlm2-500m` | [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | ~1 GB | one 512 px image (~80 tokens); for smoke tests |

All are Apache-2.0 and not gated. Revisions are pinned in `VLMS`, so weights
and chat templates never change under a run.

- **Qwen3-VL** keeps an image at its own resolution up to the token cap: a
  792 x 528 PathVQA figure is ~410 tokens.
- **SmolVLM** would otherwise split an image into 12 sub-images plus the whole
  (~880-1100 tokens); splitting is off by default. Turn it on with
  `load_processor(name, do_image_splitting=True)` when fine detail matters more
  than speed.

```python
from pathotrain.pathovlm import load_processor, load_vlm

model, processor = load_vlm("qwen3-vl-2b")  # bf16, eval mode
processor = load_processor("smolvlm2-500m")  # tokenizer, chat template, image settings only
```

## Data

One JSONL file per split, rows relative to the file's folder (or `root`):

```json
{"image": "images/3f2a....jpg", "question": "is this a carcinoma?", "answer": "yes", "type": "closed"}
```

`type` is `closed` (yes / no) or `open`; when absent it is `closed` for a yes
/ no answer. `VQADataset` returns each row as a dict with the image as an RGB
PIL image.

[PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) (MIT)
converts to this layout (pinned revision, ~785 MB download, 752 MB written):

```bash
python -m pathotrain.dataloader.vqa ~/data/pathvqa
```

Each image is stored once, named by the hash of its bytes, and images shared
between splits are reported.

| Split | Questions | Images | Closed |
|---|---|---|---|
| train | 19,654 | 2,599 | 50% |
| validation | 6,259 | 832 | 50% |
| test | 6,719 | 858 | 50% |

- One image is in both train and validation; none is shared with test
  (identical files only; a re-encoded copy of a figure would not be caught).
- 55% of train's and 54% of test's closed answers are "yes": always answering
  "yes" scores 0.540 closed accuracy on test, the floor any model must beat.
- 3,223 distinct open answers in train; the most common are organ systems
  ("oral", "cardiovascular", ...).

## Batches

`VQACollator` renders each sample with the model's own chat template: an
optional system message, the user turn (image + question) and, for training,
the assistant's answer.

```python
from torch.utils.data import DataLoader

from pathotrain.dataloader.vqa import VQADataset
from pathotrain.pathovlm import VQACollator, load_processor

processor = load_processor("qwen3-vl-2b")
train = DataLoader(
    VQADataset("pathvqa/train.jsonl"),
    batch_size=8,
    collate_fn=VQACollator(processor, system="You are a pathologist."),
)
batch = next(iter(train))  # input_ids, attention_mask, pixel_values, ..., labels
loss = model(**batch).loss  # on the answer tokens only

generate = VQACollator(processor, generation=True)  # prompts only, left-padded
```

- **Training:** only the answer and the end-of-turn tokens after it carry
  labels; the system message, question, image tokens and padding are -100.
  Each batch checks that the prompt is a prefix of the full chat and that the
  answer tokenises the same inside the chat as on its own, so a template or
  tokenizer change raises instead of training on the wrong tokens.
- **Generation:** left-padded prompts, plus the gold `answers` and `types` to
  score against.
- The collator pickles, so it works in `DataLoader` workers.
- The same collator is TRL's `data_collator` (with `remove_unused_columns=False`
  and dataset preparation skipped), so TRL does not re-tokenise the chats.

## Evaluation

```bash
python -m pathotrain.pathovlm.eval qwen3-vl-2b pathvqa/test.jsonl --out zero_shot/qwen3-vl-2b
python -m pathotrain.pathovlm.eval qwen3-vl-2b pathvqa/test.jsonl --out lora/qwen3-vl-2b \
    --adapter runs/<name>/checkpoint-7371
```

Greedy answers of the pinned base model, zero-shot or with a LoRA adapter from
[training](#training) (merged into the base; the image settings and prompt
come from the checkpoint's `provenance.json`, and an adapter trained on another
model is refused), written to `predictions.csv` (question, answer, type,
prediction) and `metrics.json`. `--max-samples N` scores N evenly spaced
questions for a quick check.

Every question is followed by one answer-format instruction, the same for all
questions and models, LLaVA's: *"Answer the question using a single word or
phrase."* Without one, instruction-tuned models answer in sentences and exact
match scores nearly zero. It is one instruction for every question because the
question type comes from the gold answer: a per-type instruction would give it
away. An instruction that also asked for yes / no on yes/no questions raised
closed accuracy by ~5 points on validation but made Qwen3-VL answer yes / no
to 98% of open questions; this one keeps both types sane (1% of open answers
are yes / no, >= 99% of closed answers contain one).

| Metric | Questions | Meaning |
|---|---|---|
| `closed_acc` | closed | the prediction's first "yes" / "no" equals the gold answer (none: wrong) |
| `closed_parsed` | closed | share of predictions with a "yes" or "no" at all |
| `open_em` | open | prediction equals gold after lowercasing and removing punctuation and articles |
| `open_f1` | open | token F1 between prediction and gold (SQuAD style) |
| `open_recall` | open | share of gold tokens in the prediction (LLaVA-Med style); rewards long answers, read it with EM and F1 |

BLEU is not reported: PathVQA answers are one to three words, too short for
its n-gram statistics.

### Zero-shot baseline on PathVQA test

6,719 questions (3,362 closed, 3,357 open), greedy, the instruction above.

| Model | closed_acc | closed_parsed | open_em | open_f1 | open_recall |
|---|---|---|---|---|---|
| always "yes" | 0.540 | 1.000 | - | - | - |
| `smolvlm2-500m` | 0.516 | 0.981 | 0.010 | 0.028 | 0.033 |
| `qwen3-vl-2b` | 0.632 | 0.998 | 0.039 | 0.062 | 0.062 |
| `smolvlm2-2.2b` | 0.584 | 0.983 | 0.018 | 0.034 | 0.035 |
| `qwen3-vl-4b` | 0.684 | 0.998 | 0.030 | 0.058 | 0.059 |

- Zero-shot, closed accuracy is near chance: SmolVLM2-500M is below the
  always-"yes" floor (0.540 on test), SmolVLM2-2.2B 4 points above it, Qwen3-VL
  9 (2B) and 14 (4B) points above it.
- Open answers are almost never right word for word: PathVQA's open answers
  are textbook phrases ("cortical bone forming the outer shell") a model cannot
  guess without training on the dataset's style. Fine-tuning is expected to
  move these most; the zero-shot open numbers mainly show the floor.

## Training

```bash
cp src/pathotrain/pathovlm/configs/qwen3-vl-2b.example.yaml my_run.yaml   # set data paths
python -m pathotrain.pathovlm.train my_run.yaml
python -m pathotrain.pathovlm.train my_run.yaml --resume runs/<name>/checkpoint-1000
```

LoRA fine-tuning with TRL's `SFTTrainer` (`trl==1.17.1`, pinned): the
answer-only loss of the collator above, adapters on the language model's linear
layers (`q/k/v/o_proj`, `gate/up/down_proj`, chosen by full module name, so the
vision tower stays frozen even where it has layers of the same names), bf16 on
GPU with gradient checkpointing. Example configs: `configs/qwen3-vl-2b.example.yaml`,
`configs/qwen3-vl-4b.example.yaml`.

- **Like every pathotrain run:** the YAML is checked first (unknown keys and
  missing files are errors, all listed at once) and translated to the stage's
  TRL config; the config is copied to `<output_dir>/config.yaml` with a
  `train.log` beside it, and a changed config for an existing run is refused
  unless resuming.
- **Stages:** `stage.name` picks the TRL trainer from
  `pathovlm/stages.py` (`STAGES`; only `sft` so far, DPO / KTO / GRPO / RLOO /
  distillation planned). `stage.args` passes any other value of that trainer's
  config class (`SFTConfig` for `sft`), e.g. `{max_grad_norm: 0.5}`. Keys the
  config class does not take, keys pathotrain's sections already set
  (`learning_rate` comes from `trainer.lr`), and values the stage needs as they
  are (`remove_unused_columns` for `sft`) are errors before anything loads. The
  stage is recorded in every checkpoint's `provenance.json`.
- **Logging:** a callback forwards TRL's logs to pathotrain's MLflow and CSV
  loggers under the usual names: `train/loss`, `train/lr`, `val/loss`,
  `val/closed_acc`, ..., end-of-run summary under `run/`, and the best
  checkpoint as `best/<monitor>` plus the `best_checkpoint` parameter. TRL's own
  `report_to` is off.
- **Evaluation** every `trainer.eval_steps` optimizer steps: `val/loss` on
  `data.val_max_samples` evenly spaced validation rows, and answers generated
  for `eval.max_samples` rows with the same instruction as training, scored as
  in [Evaluation](#evaluation). `checkpoint.monitor` (default `val/closed_acc`)
  picks the best checkpoint, which is loaded at the end.
- **Checkpoints:** `<output_dir>/checkpoint-N/` every `eval_steps`, the
  `save_total_limit` latest plus the best: the LoRA adapter
  (`adapter_model.safetensors`, 67 MB for Qwen3-VL-2B at rank 16), optimizer and
  scheduler state for `--resume`, and `provenance.json` (base model and
  revision, image settings, LoRA settings, prompt, git commit, pathotrain and
  TRL versions).

```
runs/<name>/
  config.yaml  train.log
  checkpoint-500/  adapter_model.safetensors  adapter_config.json  provenance.json  optimizer.pt ...
  logs/version_0/metrics.csv
```

Measured on one RTX 5090:

| Run | Steps | Time | Peak GPU | Result |
|---|---|---|---|---|
| smolvlm2-500m, 8 PathVQA rows, train = val (smoke) | 80 | 1 min | - | memorises all 8: closed acc, open EM and F1 = 1.0 by step 20 |
| qwen3-vl-2b, batch 8, 2% of an epoch, 2 evaluations | 50 | 72 s | 8.9 GB | runs end to end; open EM 0.08-0.09 on 128 val rows after 400 examples (zero-shot: 0.04 on test) |
| qwen3-vl-2b example config, 3 epochs, 15 evaluations | 7,371 | 1 h 18 min | ~11 GB (one reading, not tracked) | below; adapter 67 MB |
| qwen3-vl-4b, batch 4 x 2, 3 epochs, 15 evaluations on 512 rows (seed 0) | 7,371 | 2 h 11 min | 17.3 GB | below; adapter 127 MB |
| qwen3-vl-4b example config (batch 4 x 2), 15 evaluations on 2,048 rows (seed 1) | 7,371 | 2 h 35 min | 21.8 GB | below |
| qwen3-vl-4b, batch 8 x 1, eval batch 32 (smoke, 2,048-row evaluations) | 51 | same as 4 x 2 | 27.9 GB | no faster: ~1.0 vs ~1.1 steps/s, evaluation equally long |
| qwen3-vl-4b, batch 4 x 2 without gradient checkpointing (smoke) | - | - | > 32 GB | out of memory at step 12 |

The 4B example already keeps the GPU busy at batch 4 x 2, so the free memory
buys no speed: a larger micro-batch or generation batch is not faster, and
dropping gradient checkpointing does not fit. The time to cut is evaluation: each one (val/loss on
1,024 rows, answers to 2,048) takes ~3 min, ~45 min of a 2 h 35 min run; raise
`trainer.eval_steps` to evaluate less often.

### LoRA fine-tuning on PathVQA

The example configs (rank 16 on all language-model linear layers, lr 2e-4
cosine, 3 epochs, effective batch 8), scored on PathVQA test (6,719 questions)
with `eval --adapter`. "best" is the checkpoint chosen by `val/closed_acc` on
512 validation rows (seed 0) or 2,048 (seed 1, today's default), "final" the
last one.

| Model | Checkpoint | closed_acc | open_em | open_f1 | open_recall |
|---|---|---|---|---|---|
| always "yes" | | 0.540 | - | - | - |
| `qwen3-vl-2b` | zero-shot | 0.632 | 0.039 | 0.062 | 0.062 |
| `qwen3-vl-2b` | seed 0, best (step 4500) | 0.914 | 0.310 | 0.353 | 0.352 |
| `qwen3-vl-2b` | seed 0, final (step 7371) | 0.918 | 0.326 | 0.369 | 0.371 |
| `qwen3-vl-4b` | zero-shot | 0.684 | 0.030 | 0.058 | 0.059 |
| `qwen3-vl-4b` | seed 0, best (step 5000) | 0.932 | 0.331 | 0.378 | 0.379 |
| `qwen3-vl-4b` | seed 0, final (step 7371) | 0.933 | **0.353** | **0.402** | **0.404** |
| `qwen3-vl-4b` | seed 1, best (step 6500) | **0.940** | 0.338 | 0.384 | 0.386 |
| `qwen3-vl-4b` | seed 1, final (step 7371) | 0.936 | 0.343 | 0.391 | 0.394 |

- Fine-tuning is what matters: +28 / +25 points closed accuracy and ~8x open
  EM over zero-shot. 4B over 2B adds ~1.5-2 points closed and ~2-3 points open
  EM and F1, for 1.7x the training time and twice the adapter size.
- Seed to seed (4B, final checkpoints): 0.3 points closed accuracy, 1 point
  open EM and F1. The 2B-to-4B gap is larger than that; differences under ~1
  point between single runs are not.
- `val/loss` is lowest around 1.8 epochs and rises in epoch 3 (2B: 0.937 to
  1.067, 4B: 0.817 to 0.946), but the generated answers keep improving: the
  final checkpoint beats the "best" one on test for both models. Answer-only
  cross-entropy rises as the model grows more confident on the answers it gets
  wrong; it is not the metric to stop on here.
- 512 validation rows (~270 closed) were too few to choose between late
  checkpoints: "best" lost to "final" on test in both seed-0 runs. With 2,048
  (now the default) the choice held up on closed accuracy (best 0.940 vs final
  0.936) but not on open answers (0.338 vs 0.343 EM): `val/closed_acc` picks
  for closed questions. Within seed noise either way.
- Two seeds for 4B, one for 2B, no hyperparameter search.

## Export

```bash
python -m pathotrain.pathovlm.export runs/<name>/checkpoint-7371 exported/<name>
```

Writes the base model with the adapter merged into its weights (bf16
`model*.safetensors`, `config.json`), the processor with the image settings it
was trained with, and `pathotrain.json` (the checkpoint's provenance: base model
and revision, image settings, LoRA settings, prompt, git commit). Loading it
needs transformers alone, no peft or pathotrain:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained("exported/<name>")
processor = AutoProcessor.from_pretrained("exported/<name>")
```

Ask as in training: the question followed by the instruction in
`pathotrain.json` (`provenance.prompt.instruction`), greedy decoding. The
`checkpoint-N/` folder itself is the adapter alone (`adapter_config.json`,
`adapter_model.safetensors`), for serving several adapters over one base.

Checked on the `qwen3-vl-4b` final checkpoint: an 8.3 GB folder whose weights
equal the in-memory merge tensor for tensor, and whose answers on 512 test
questions equal `eval --adapter`'s run with the same batches. Answers can differ
for ~2% of questions when they are batched differently: in bf16, the padding in
a batch changes the numbers slightly. Quantized exports (FP8, AWQ, GGUF), each
re-scored on PathVQA test, are not built yet.

## Design notes

- **TRL, not pathotrain's engine.** Supervised fine-tuning alone would fit
  either; DPO and GRPO (generation in the training loop, rewards, a reference
  model, vLLM) would be a large build on the engine, and TRL has them.
- **bf16 LoRA, not QLoRA.** The 2B and 4B models fit a 32 GB GPU with LoRA in
  bf16; quantization is for export (FP8, AWQ or GGUF, each re-scored on
  PathVQA test).
- **No Unsloth.** It would pin transformers and huggingface-hub to older
  versions for the whole workspace and patches them at import.

## Tests

```bash
uv run pytest packages/pathotrain/tests/test_pathovlm.py packages/pathotrain/tests/test_pathovlm_train.py packages/pathotrain/tests/test_vqa_dataloader.py
```

The collator tests use the real chat templates; they need the processors in
the Hugging Face cache (a few MB each, no weights) and skip otherwise. The
training test runs TRL end to end on CPU with SmolVLM2-500M's architecture
shrunk to ~3M parameters (random weights, real processor): checkpoints,
provenance, LoRA targets, CSV and MLflow logs, `eval --adapter`, and an export
that reloads with transformers alone and answers as the adapter does.
