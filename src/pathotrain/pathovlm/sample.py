"""Answers sampled from a fine-tuned VLM: the preference data of the ``dpo`` and
``kto`` stages (:mod:`pathotrain.pathovlm.stages`)::

    python -m pathotrain.pathovlm.sample runs/sft01/checkpoint-7371 pathvqa/train.jsonl \
        samples/train.jsonl --n 4 --temperature 1.0

Every row of the VQA JSONL is written again with ``samples``: ``n`` answers of the
checkpoint (with the prompt and image settings of its ``provenance.json``) and
``image`` relative to the new file. The stages label each sample right or wrong
with :func:`pathotrain.pathovlm.metrics.correct`. ``<out>.provenance.json`` records
the checkpoint and the sampling settings.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from pathotrain.dataloader.vqa import VQADataset
from pathotrain.pathovlm.eval import generate_answers
from pathotrain.pathovlm.export import load_finetuned


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="checkpoint-N folder written by pathovlm.train")
    parser.add_argument("data", help="VQA JSONL (see pathotrain.dataloader.vqa)")
    parser.add_argument("out", help="the JSONL to write")
    parser.add_argument("--n", type=int, default=4, help="answers per question")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, help="evenly spaced questions only")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    dataset = VQADataset(args.data)
    rows = dataset.rows
    if args.max_samples and len(dataset) > args.max_samples:
        keep = np.linspace(0, len(dataset) - 1, args.max_samples).round().astype(int)
        dataset, rows = Subset(dataset, keep), [rows[i] for i in keep]
    model, processor, prov = load_finetuned(args.checkpoint)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    sampling = {"do_sample": True, "temperature": args.temperature, "num_return_sequences": args.n}
    answers = generate_answers(
        model,
        processor,
        dataset,
        args.batch_size,
        args.num_workers,
        args.max_new_tokens,
        **prov["prompt"],
        sampling=sampling,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, row in enumerate(rows):
            image = os.path.relpath(row["image"], out.parent)
            samples = answers[i * args.n : (i + 1) * args.n]
            f.write(json.dumps({**row, "image": image, "samples": samples}) + "\n")
    record = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        **{k: getattr(args, k) for k in ("n", "temperature", "max_samples", "max_new_tokens")},
        "seed": args.seed,
    }
    Path(f"{out}.provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
