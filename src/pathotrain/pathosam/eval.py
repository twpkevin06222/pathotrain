"""Interactive segmentation of SAM 3, scored on instance-labelled tiles::

    python -m pathotrain.pathosam.eval zeroshot.yaml --split val --out val.json
    python -m pathotrain.pathosam.eval zeroshot.yaml --split test --only tracker
    python -m pathotrain.pathosam.eval zeroshot.yaml --split test --only tracker \
        --checkpoint runs/interactive/checkpoints/best.ckpt   # a fine-tuned model
    python -m pathotrain.pathosam.eval zeroshot.yaml --split test --only tracker --onnx out/

Start from ``configs/zeroshot.example.yaml`` beside this module. Two evaluations, both
with prompts simulated from the ground truth (``data[split]``, an
``InstanceTileFolderDataset``); SAM 3 (``facebook/sam3``) is gated, SAM License:

- ``tracker``, SAM 3's point and box model (``Sam3TrackerModel``), one object per prompt:
  zero-shot, or fine-tuned by :mod:`pathotrain.pathosam.train` with ``--checkpoint``.
  For every nucleus of at least ``min_size`` px:

  - a box, for each jitter in ``box_jitter``: IoU (``iou_box_<jitter>``);
  - a click at its centre, then ``max_clicks - 1`` corrective clicks (see
    :mod:`pathotrain.pathosam.prompts`), each round also given the previous mask:
    mean IoU after ``k`` clicks (``iou_click<k>``), and ``noc85`` / ``noc90``, the mean
    number of clicks to reach that IoU (``max_clicks`` when never), with ``fail85`` /
    ``fail90``, the share that never does.

- ``exemplar``, SAM 3's concept model (``Sam3Model``, zero-shot) given one example box per
  nucleus type present in the tile, from one of that type's nuclei: every detection takes
  the type, overlaps go to the higher score, and the result is scored like
  ``pathotrain.cellseg.eval`` (``bpq``, ``mpq``, ``pq_<type>``, ``dq``, ``sq``) at every
  threshold in ``thresholds``. The example nucleus is itself counted, as an annotator's box
  would be. With ``exemplar.prompts`` (a text per type, as ``pathotrain.cellseg.prompted``'s),
  each example box goes with its type's text, and the metrics are keyed ``exemplar_text``.

The tile is encoded once and reused for every prompt. Metrics print as JSON keyed
``<split>/<evaluation>/<metric>``. Fine-tuning validates with the same ``tracker`` code.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch import Tensor
from torchmetrics import Metric

from pathotrain.cellseg.instances import instance_types
from pathotrain.cellseg.prompted import paint
from pathotrain.cellseg.task import PanopticQuality
from pathotrain.dataloader import InstanceTileFolderDataset
from pathotrain.dataloader.augmentation import eval_transform
from pathotrain.engine import load_config
from pathotrain.pathofm.sam3 import SAM3_REPO, SAM3_REVISION
from pathotrain.pathosam.model import PAD, InteractiveSam, to_tile
from pathotrain.pathosam.prompts import box, center_click, corrective_click

SAM_MEAN = SAM_STD = (0.5, 0.5, 0.5)  # SAM 3's processor normalisation


class Tracker:
    """Masks at tile size for the objects of one tile, from an :class:`InteractiveSam`."""

    def __init__(self, model: InteractiveSam) -> None:
        self.model = model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def encode(self, image: Tensor) -> list[Tensor]:
        """Embeddings of one normalised tile ``(3, H, W)``."""
        return self.model(images=image[None].to(self.device))

    @torch.no_grad()
    def predict(
        self,
        embeddings: list[Tensor],
        size: tuple[int, int],
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        masks: Tensor | None = None,
        multimask: bool = False,
    ) -> tuple[np.ndarray, Tensor]:
        """Masks ``(N, H, W)`` bool for N objects and their low-resolution logits
        ``(N, 1, h, w)``, to feed back as ``masks``. ``points`` ``(N, K, 2)`` and ``boxes``
        ``(N, 4)`` are in tile pixels. With ``multimask``, the mask SAM scores best."""
        n = len(points if points is not None else boxes)
        as_tensor = lambda a, dtype: (
            None if a is None else torch.as_tensor(a, dtype=dtype, device=self.device)
        )
        logits, scores = self.model(
            embeddings=[e.expand(n, *e.shape[1:]) for e in embeddings],
            size=size,
            points=as_tensor(points, torch.float32),
            labels=as_tensor(labels, torch.long),
            boxes=as_tensor(boxes, torch.float32),
            masks=masks,
            multimask=multimask,
        )
        best = (
            scores.argmax(-1) if multimask else torch.zeros(n, dtype=torch.long, device=self.device)
        )
        logits = logits[torch.arange(n), best][:, None]
        return (to_tile(logits, size)[:, 0] > 0).cpu().numpy(), logits


def _iou(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    inter = (true & pred).sum((1, 2))
    union = (true | pred).sum((1, 2))
    return inter / np.maximum(union, 1)


def track_tile(
    tracker: Tracker,
    image: Tensor,
    true: np.ndarray,
    max_clicks: int,
    jitters: list[float],
    rng: np.random.Generator,
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """For the ``(N, H, W)`` masks ``true`` of one normalised tile ``image``: IoU per box
    jitter ``{jitter: (N,)}`` and the IoU after each click ``(N, max_clicks)``."""
    size = true.shape[1:]
    emb = tracker.encode(image)
    box_iou = {}
    for j in jitters:
        boxes = np.stack([box(m, j, rng) for m in true])
        box_iou[j] = _iou(true, tracker.predict(emb, size, boxes=boxes)[0])
    points = np.array([[center_click(m)] for m in true])
    labels = np.ones((len(true), 1), np.int64)
    pred, logits = tracker.predict(emb, size, points, labels, multimask=True)
    curve = [_iou(true, pred)]
    for _ in range(max_clicks - 1):
        clicks = [corrective_click(t, p) for t, p in zip(true, pred, strict=True)]
        new = np.array([(c[0], c[1]) if c else (0.0, 0.0) for c in clicks])
        points = np.concatenate([points, new[:, None]], 1)
        labels = np.concatenate([labels, [[c[2] if c else PAD] for c in clicks]], 1)
        pred, logits = tracker.predict(emb, size, points, labels, masks=logits)
        curve.append(_iou(true, pred))
    return box_iou, np.stack(curve, 1)


class ClickMetrics(Metric):
    """Interactive metrics summed over objects (and ranks): ``iou_box_<jitter>``,
    ``iou_click<k>`` after k clicks, and for 85 and 90% IoU the clicks needed (``noc``, the
    maximum when never) and the share never reaching it (``fail``)."""

    full_state_update = False

    def __init__(self, jitters: list[float], max_clicks: int) -> None:
        super().__init__()
        self.jitters, self.max_clicks = list(jitters), max_clicks
        add = lambda name, n: self.add_state(name, torch.zeros(n, dtype=torch.float64), "sum")
        add("box", len(self.jitters))
        add("iou", max_clicks)
        add("noc", 2)
        add("fail", 2)
        add("objects", 1)

    def update(self, box_iou: dict[float, np.ndarray], curves: np.ndarray) -> None:
        as_t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=self.box.device)
        self.box += as_t([box_iou[j].sum() for j in self.jitters])
        self.iou += as_t(curves.sum(0))
        for i, t in enumerate((0.85, 0.90)):
            reached = curves >= t
            clicks = np.where(reached.any(1), reached.argmax(1) + 1, self.max_clicks)
            self.noc[i] += float(clicks.sum())
            self.fail[i] += float((~reached.any(1)).sum())
        self.objects += len(curves)

    def compute(self) -> dict[str, float]:
        n = self.objects.clamp_min(1)
        out = {f"iou_box_{j}": (v / n).item() for j, v in zip(self.jitters, self.box, strict=True)}
        out |= {f"iou_click{k + 1}": (v / n).item() for k, v in enumerate(self.iou)}
        for i, t in enumerate((85, 90)):
            out[f"noc{t}"] = (self.noc[i] / n).item()
            out[f"fail{t}"] = (self.fail[i] / n).item()
        out["objects"] = self.objects.item()
        return out


def nuclei(inst: np.ndarray, min_size: int) -> list[int]:
    """The instance ids of ``inst`` with at least ``min_size`` pixels."""
    sizes = np.bincount(inst.ravel())
    return [int(k) for k in np.nonzero(sizes >= min_size)[0] if k > 0]


class Exemplar:
    """``Sam3Model`` prompted with example boxes, alone or with text: all objects like the
    example (and the text)."""

    def __init__(self, device: str, revision: str = SAM3_REVISION) -> None:
        from transformers import Sam3Model, Sam3Processor

        self.device = device
        self.processor = Sam3Processor.from_pretrained(SAM3_REPO, revision=revision)
        self.model = Sam3Model.from_pretrained(SAM3_REPO, revision=revision)
        self.model.to(device).eval()

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        boxes: list[np.ndarray],
        threshold: float,
        mask_threshold: float,
        texts: list[str | None] | None = None,
    ) -> list[list[tuple[float, np.ndarray]]]:
        """Per example box, the ``(score, mask)`` detections scoring at least ``threshold``.
        ``texts`` (one per box) prompts each box together with that text; without it, or for a
        None entry, the box alone (SAM 3's processor then uses the text "visual")."""
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"]
        autocast = torch.autocast("cuda", torch.bfloat16, enabled=self.device == "cuda")
        out = []
        with autocast:
            vision = self.model.get_vision_features(pixel_values=pixels.to(self.device))
            for b, text in zip(boxes, texts or [None] * len(boxes), strict=True):
                prompt = self.processor(
                    images=image,
                    text=text,
                    input_boxes=[[b.tolist()]],
                    input_boxes_labels=[[1]],
                    return_tensors="pt",
                ).to(self.device)
                pred = self.model(
                    vision_embeds=vision,
                    input_ids=prompt["input_ids"],
                    attention_mask=prompt["attention_mask"],
                    input_boxes=prompt["input_boxes"],
                    input_boxes_labels=prompt["input_boxes_labels"],
                )
                r = self.processor.post_process_instance_segmentation(
                    pred, threshold, mask_threshold, target_sizes=[image.size[::-1]]
                )[0]
                scores = r["scores"].float().cpu().numpy()
                out.append(list(zip(scores, r["masks"].bool().cpu().numpy(), strict=True)))
        return out


def evaluate(
    cfg: dict,
    split: str,
    only: str | None = None,
    checkpoint: str | Path | None = None,
    onnx: str | Path | None = None,
) -> dict[str, float]:
    """Run the evaluations of ``cfg`` (``only`` one of them) on ``data[split]``. The tracker
    is the fine-tuned model of ``checkpoint``, or the ONNX export in folder ``onnx``
    (:mod:`pathotrain.pathosam.export`), else zero-shot SAM 3."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = eval_transform(mean=SAM_MEAN, std=SAM_STD)
    dataset = InstanceTileFolderDataset(transform=transform, **cfg["data"][split])
    limit = min(len(dataset), cfg.get("limit") or len(dataset))
    rng = np.random.default_rng(cfg.get("seed", 0))
    min_size = cfg.get("min_size", 10)
    run_tracker = only in (None, "tracker")
    run_exemplar = only in (None, "exemplar")
    if run_tracker:
        t_cfg = cfg.get("tracker", {})
        if onnx is not None:
            from pathotrain.pathosam.export import OnnxTracker

            tracker = OnnxTracker(onnx)
        elif checkpoint is not None:
            from pathotrain.pathosam.train import load_model

            tracker = Tracker(load_model(checkpoint)[0].to(device).eval())
        else:
            tracker = Tracker(InteractiveSam().to(device).eval())
        jitters = t_cfg.get("box_jitter", [0.0, 0.05, 0.1])
        clicks = ClickMetrics(jitters, t_cfg.get("max_clicks", 20))
    if run_exemplar:
        e_cfg = cfg.get("exemplar", {})
        exemplar = Exemplar(device)
        texts = {int(k): v for k, v in (e_cfg.get("prompts") or {}).items()}
        e_name = "exemplar_text" if texts else "exemplar"
        thresholds = sorted(e_cfg.get("thresholds", [0.5]))
        num_classes = cfg["num_classes"]
        pq = {t: PanopticQuality(num_classes) for t in thresholds}
    for i in range(limit):
        image, target = dataset[i]
        inst, types = target["inst"].numpy(), target["type"].numpy()
        ids = nuclei(inst, min_size)
        if run_tracker and ids:
            true = np.stack([inst == k for k in ids])
            clicks.update(*track_tile(tracker, image, true, clicks.max_clicks, jitters, rng))
        if run_exemplar:
            kinds = instance_types(inst, types, num_classes)
            present = sorted({int(kinds[k]) for k in ids} - {0})
            examples = [box(inst == rng.choice([k for k in ids if kinds[k] == c])) for c in present]
            pil = Image.open(dataset.samples[i][0]).convert("RGB")
            found = exemplar.detect(
                pil,
                examples,
                thresholds[0],
                e_cfg.get("mask_threshold", 0.5),
                [texts.get(c) for c in present] if texts else None,
            )
            for t in thresholds:
                kept = [
                    (s, m, c)
                    for c, dets in zip(present, found, strict=True)
                    for s, m in dets
                    if s >= t
                ]
                pred, pred_types = paint(kept, inst.shape, min_size)
                pq[t].update(inst, kinds, pred, pred_types)
        if i % 100 == 0:
            logger.info("{}: {}/{} tiles", split, i + 1, limit)
    metrics = {}
    if run_tracker:
        metrics |= {f"{split}/tracker/{k}": v for k, v in clicks.compute().items()}
    if run_exemplar:
        names = cfg.get("class_names") or [str(c) for c in range(num_classes)]
        for t in thresholds:
            r = pq[t].compute()
            metrics |= {
                f"{split}/{e_name}/{k}@{t}": r[k].item() for k in ("bpq", "mpq", "dq", "sq")
            }
            per_type = zip(names[1:], r["pq"][1:], strict=True)
            metrics |= {f"{split}/{e_name}/pq_{n}@{t}": v.item() for n, v in per_type}
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="zeroshot.yaml")
    parser.add_argument("--split", default="val", help="data split to score (default: val)")
    parser.add_argument("--only", choices=["tracker", "exemplar"], help="run one evaluation")
    parser.add_argument("--checkpoint", help="score this fine-tuned tracker (pathosam.train)")
    parser.add_argument("--onnx", help="score this exported tracker folder (pathosam.export)")
    parser.add_argument("--out", help="also write the metrics to this JSON file")
    args = parser.parse_args(argv)
    metrics = evaluate(load_config(args.config), args.split, args.only, args.checkpoint, args.onnx)
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
