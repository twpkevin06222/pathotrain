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
> registry and the training / generation collator. Next: a zero-shot baseline,
> then LoRA fine-tuning with TRL, evaluation and export.

```bash
pip install -e ".[pathovlm]"   # peft, num2words (SmolVLM's processor needs it)
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
- 55% of the closed answers are "yes": always answering "yes" scores 0.55
  closed accuracy, the floor any model must beat.
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

## Training (planned)

```bash
python -m pathotrain.pathovlm.train configs/qwen3-vl-2b.example.yaml
```

- **TRL `SFTTrainer`** (`trl==1.17.1`, pinned): the answer-only loss above, peft
  LoRA on the language model's linear layers, vision tower frozen, projector
  optionally trained. The same trainer family brings DPO and GRPO (e.g. with
  closed-answer rewards) later.
- **pathotrain conventions at the edges:** one YAML per run, validated and
  translated to `SFTConfig`; `experiment.start` copies the config, starts
  `train.log`, seeds and refuses a changed config for an existing run.
- **Logging:** a callback forwards TRL's logs to pathotrain's MLflow and CSV
  loggers under the usual names (`train/loss`, `train/lr`, `val/loss`, ...), into
  the same `mlflow_db`, rather than TRL's own `report_to="mlflow"`.
- **Answer metrics:** the trainer's evaluation computes only the loss, so a
  second callback generates answers for up to `eval.max_samples` validation
  rows and logs `val/closed_acc`, `val/open_em` and `val/open_f1`. The best
  checkpoint follows `val/closed_acc` by default.
- **Checkpoints:** TRL's `checkpoint-N/` folders hold the LoRA adapter as
  `adapter_model.safetensors` (tens of MB, served directly by vLLM), plus a
  `provenance.json`: model and revision, LoRA settings, prompt, image settings,
  git commit, versions.

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
uv run pytest packages/pathotrain/tests/test_pathovlm.py packages/pathotrain/tests/test_vqa_dataloader.py
```

The collator tests use the real chat templates; they need the processors in
the Hugging Face cache (a few MB each, no weights) and skip otherwise.
