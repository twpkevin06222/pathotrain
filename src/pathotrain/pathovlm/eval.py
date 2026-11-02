"""Answers of a VLM on a VQA split, scored::

    python -m pathotrain.pathovlm.eval qwen3-vl-2b pathvqa/test.jsonl --out zero_shot/qwen3-vl-2b
    python -m pathotrain.pathovlm.eval qwen3-vl-2b pathvqa/test.jsonl --out lora/qwen3-vl-2b \
        --adapter runs/vqa01/checkpoint-7371

The pinned base model, zero-shot or with a LoRA adapter from
:mod:`pathotrain.pathovlm.train` (its ``provenance.json`` gives the image settings
and prompt it was trained with). Greedy decoding with one answer-format
instruction after every question (:data:`INSTRUCTION`), so models that would
answer in sentences are scored on their answer, not their style.
Writes ``predictions.csv`` (question, answer, type, prediction) and
``metrics.json`` (:func:`pathotrain.pathovlm.metrics.score`, model, revision,
instruction) under ``--out`` and prints the metrics.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import ProcessorMixin

from pathotrain.dataloader.vqa import VQADataset
from pathotrain.pathovlm.collate import VQACollator
from pathotrain.pathovlm.metrics import score
from pathotrain.pathovlm.models import VLMS, load_vlm

# LLaVA's VQA instruction. One that also asked for yes / no on yes/no questions made
# Qwen3-VL answer yes / no to 98% of open questions (PathVQA validation, 2026-11-02).
INSTRUCTION = "Answer the question using a single word or phrase."


@torch.no_grad()
def generate_answers(
    model: nn.Module,
    processor: ProcessorMixin,
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    max_new_tokens: int = 32,
    system: str | None = None,
    instruction: str | None = INSTRUCTION,
) -> list[str]:
    """Greedy answers of ``model`` to every question of ``dataset``, in order."""
    device = next(model.parameters()).device
    collate = VQACollator(processor, system, generation=True, instruction=instruction)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        multiprocessing_context="forkserver" if num_workers else None,  # CUDA is up
    )
    predictions = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = ids[:, inputs["input_ids"].shape[1] :]  # left padding: prompts end together
        predictions += [
            p.strip() for p in collate.processor.batch_decode(new, skip_special_tokens=True)
        ]
    return predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", choices=sorted(VLMS))
    parser.add_argument("data", help="VQA JSONL (see pathotrain.dataloader.vqa)")
    parser.add_argument("--out", required=True, help="folder for predictions.csv, metrics.json")
    parser.add_argument("--max-samples", type=int, help="score evenly spaced questions only")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--adapter", help="checkpoint-N folder of pathotrain.pathovlm.train")
    args = parser.parse_args(argv)
    prov = {"image": {}, "prompt": {"system": None, "instruction": INSTRUCTION}}
    if args.adapter:
        prov = json.loads((Path(args.adapter) / "provenance.json").read_text())
        if prov["model"] != args.model:
            parser.error(f"--adapter was trained on {prov['model']}, not {args.model}")

    dataset = VQADataset(args.data)
    rows = dataset.rows
    if args.max_samples and len(dataset) > args.max_samples:
        keep = np.linspace(0, len(dataset) - 1, args.max_samples).round().astype(int)
        dataset, rows = Subset(dataset, keep), [rows[i] for i in keep]
    model, processor = load_vlm(args.model, **prov["image"])
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    predictions = generate_answers(
        model,
        processor,
        dataset,
        args.batch_size,
        args.num_workers,
        args.max_new_tokens,
        **prov["prompt"],
    )
    answers, types = [r["answer"] for r in rows], [r["type"] for r in rows]
    metrics = score(predictions, answers, types)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "type", "prediction"])
        writer.writerows(
            (r["question"], r["answer"], r["type"], p)
            for r, p in zip(rows, predictions, strict=True)
        )
    vlm = VLMS[args.model]
    report = {
        "model": args.model,
        "revision": f"{vlm.repo}@{vlm.revision}",
        "adapter": args.adapter,
        "data": args.data,
        **prov["prompt"],
        **metrics,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
