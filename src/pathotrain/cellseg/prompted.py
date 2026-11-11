"""Zero-shot nucleus segmentation and typing with SAM 3's text prompts, scored as
:mod:`pathotrain.cellseg.eval` scores trained models::

    python -m pathotrain.cellseg.prompted prompted.yaml --split val
    python -m pathotrain.cellseg.prompted prompted.yaml --split test --out test.json

Start from ``configs/prompted.example.yaml`` beside this module. SAM 3 (``facebook/sam3``,
gated, SAM License; see :mod:`pathotrain.pathofm.sam3`) finds every instance of a text
concept in a tile. With one prompt per nucleus type (``prompts: {1: inflammatory cell,
...}``), each detection takes its prompt's type; with a single prompt (``prompts: cell``),
nuclei are untyped and only the binary metrics mean anything. Where detections overlap,
the higher score keeps the pixels, and a detection left with under ``min_size`` px is
dropped.

Every score threshold in ``thresholds`` is scored in one pass: choose it on ``val``, then
report ``test``. Metrics print as JSON keyed ``<split>/<metric>@<threshold>`` (``bpq``,
``mpq``, ``dq``, ``sq``, ``pq_<type>``, ``nuclei_per_tile``), as
:class:`~pathotrain.cellseg.task.PanopticQuality` computes them.

SAM 3 recognises some phrasings far better than others: on PanNuke, "cell" finds nuclei
and "nucleus" almost nothing, and "neoplastic cell" beats "tumor cell".
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from pathotrain.cellseg.instances import instance_types
from pathotrain.cellseg.task import PanopticQuality
from pathotrain.dataloader import InstanceTileFolderDataset
from pathotrain.engine import load_config
from pathotrain.pathofm.sam3 import SAM3_REPO, SAM3_REVISION


def paint(
    detections: list[tuple[float, np.ndarray, int]], shape: tuple[int, int], min_size: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """An instance map and each id's type (indexed by id) from ``(score, mask, type)``
    detections: higher scores claim their pixels first, and a detection left with fewer
    than ``min_size`` free pixels is dropped."""
    inst = np.zeros(shape, np.int32)
    types = [0]
    for _, mask, kind in sorted(detections, key=lambda d: -d[0]):
        free = mask & (inst == 0)
        if free.sum() < min_size:
            continue
        types.append(kind)
        inst[free] = len(types) - 1
    return inst, np.array(types)


class Sam3Prompter:
    """SAM 3 (the full model: text encoder, detector and mask decoder) for text prompts.
    Each image batch is encoded once and reused for every prompt."""

    def __init__(self, device: str | None = None, revision: str = SAM3_REVISION) -> None:
        from transformers import Sam3Model, Sam3Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam3Processor.from_pretrained(SAM3_REPO, revision=revision)
        self.model = Sam3Model.from_pretrained(SAM3_REPO, revision=revision)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def detect(
        self, images: list[Image.Image], prompts: list[str], threshold: float, mask_threshold: float
    ) -> list[dict[str, list[tuple[float, np.ndarray]]]]:
        """Per image, each prompt's ``(score, mask)`` detections scoring at least
        ``threshold``, masks at the image's size."""
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        autocast = torch.autocast("cuda", torch.bfloat16, enabled=self.device == "cuda")
        out: list[dict] = [{} for _ in images]
        with autocast:
            vision = self.model.get_vision_features(pixel_values=pixels.to(self.device))
            for prompt in prompts:
                text = self.processor(text=[prompt], return_tensors="pt").to(self.device)
                pred = self.model(
                    vision_embeds=vision,
                    input_ids=text["input_ids"].expand(len(images), -1),
                    attention_mask=text["attention_mask"].expand(len(images), -1),
                )
                results = self.processor.post_process_instance_segmentation(
                    pred,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                    target_sizes=[image.size[::-1] for image in images],
                )
                for found, r in zip(out, results, strict=True):
                    scores = r["scores"].float().cpu().numpy()
                    found[prompt] = list(zip(scores, r["masks"].bool().cpu().numpy(), strict=True))
        return out


def evaluate(cfg: dict, split: str, prompter: Sam3Prompter | None = None) -> dict[str, float]:
    """Score SAM 3's prompted detections on ``cfg["data"][split]`` at every threshold."""
    prompts = cfg["prompts"]
    typed = isinstance(prompts, dict)
    by_type = {int(k): v for k, v in prompts.items()} if typed else {1: prompts}
    thresholds = sorted(cfg.get("thresholds", [0.5]))
    num_classes = cfg.get("num_classes", max(by_type) + 1)
    names = cfg.get("class_names") or [str(i) for i in range(num_classes)]
    dataset = InstanceTileFolderDataset(**cfg["data"][split])
    prompter = prompter or Sam3Prompter()
    pq = {t: PanopticQuality(num_classes) for t in thresholds}
    found = dict.fromkeys(thresholds, 0)
    batch = cfg.get("batch_size", 8)
    for start in range(0, len(dataset), batch):
        idx = range(start, min(len(dataset), start + batch))
        images = [Image.open(dataset.samples[i][0]).convert("RGB") for i in idx]
        detections = prompter.detect(
            images, list(by_type.values()), thresholds[0], cfg.get("mask_threshold", 0.5)
        )
        for i, dets in zip(idx, detections, strict=True):
            target = dataset[i][1]
            true = target["inst"].numpy()
            true_types = instance_types(true, target["type"].numpy(), num_classes)
            for t in thresholds:
                kept = [(s, m, kind) for kind, p in by_type.items() for s, m in dets[p] if s >= t]
                pred, pred_types = paint(kept, true.shape, cfg.get("min_size", 10))
                pq[t].update(true, true_types, pred, pred_types)
                found[t] += int(pred.max())
        if start % (batch * 25) == 0:
            logger.info("{}: {}/{} tiles", split, idx[-1] + 1, len(dataset))
    metrics = {}
    for t in thresholds:
        r = pq[t].compute()
        keys = ("bpq", "dq", "sq", "mpq") if typed else ("bpq", "dq", "sq")
        metrics |= {f"{split}/{k}@{t}": r[k].item() for k in keys}
        if typed:
            per_type = zip(names[1:], r["pq"][1:], strict=True)
            metrics |= {f"{split}/pq_{n}@{t}": v.item() for n, v in per_type}
        metrics[f"{split}/nuclei_per_tile@{t}"] = found[t] / len(dataset)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="prompted.yaml")
    parser.add_argument("--split", default="val", help="data split to score (default: val)")
    parser.add_argument("--out", help="also write the metrics to this JSON file")
    args = parser.parse_args(argv)
    metrics = evaluate(load_config(args.config), args.split)
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
