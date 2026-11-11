"""Interactive segmentation baselines for SAM 3, scored on instance-labelled tiles::

    python -m pathotrain.pathosam.eval zeroshot.yaml --split val --out val.json
    python -m pathotrain.pathosam.eval zeroshot.yaml --split test --only tracker

Start from ``configs/zeroshot.example.yaml`` beside this module. Two evaluations, both
with prompts simulated from the ground truth (``data[split]``, an
``InstanceTileFolderDataset``); SAM 3 (``facebook/sam3``) is gated, SAM License:

- ``tracker``, SAM 3's point and box model (``Sam3TrackerModel``), one object per prompt.
  For every nucleus of at least ``min_size`` px:

  - a box, for each jitter in ``box_jitter``: IoU (``iou_box@<jitter>``);
  - a click at its centre, then ``max_clicks - 1`` corrective clicks (see
    :mod:`pathotrain.pathosam.prompts`), each round also given the previous mask:
    mean IoU after ``k`` clicks (``iou@<k>``), and ``noc@85`` / ``noc@90``, the mean
    number of clicks to reach that IoU (``max_clicks`` when never), with ``fail@85`` /
    ``fail@90``, the share that never does.

- ``exemplar``, SAM 3's concept model (``Sam3Model``) given one example box per nucleus
  type present in the tile, from one of that type's nuclei: every detection takes the
  type, overlaps go to the higher score, and the result is scored like
  ``pathotrain.cellseg.eval`` (``bpq``, ``mpq``, ``pq_<type>``, ``dq``, ``sq``) at every
  threshold in ``thresholds``. The example nucleus is itself counted, as an annotator's box
  would be.

The tile is encoded once and reused for every prompt. Metrics print as JSON keyed
``<split>/<evaluation>/<metric>``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from pathotrain.cellseg.instances import instance_types
from pathotrain.cellseg.prompted import paint
from pathotrain.cellseg.task import PanopticQuality
from pathotrain.dataloader import InstanceTileFolderDataset
from pathotrain.engine import load_config
from pathotrain.pathofm.sam3 import SAM3_REPO, SAM3_REVISION
from pathotrain.pathosam.prompts import box, center_click, corrective_click

PAD = -10  # the label of a padding point, which SAM's prompt encoder ignores


class Tracker:
    """``Sam3TrackerModel``: masks at tile size for a batch of objects in one tile."""

    def __init__(self, device: str, revision: str = SAM3_REVISION) -> None:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        self.device = device
        self.processor = Sam3TrackerProcessor.from_pretrained(SAM3_REPO, revision=revision)
        self.model = Sam3TrackerModel.from_pretrained(SAM3_REPO, revision=revision)
        self.model.to(device).eval()
        self.side = self.processor.image_processor.size["height"]  # 1008

    @torch.no_grad()
    def encode(self, image: Image.Image) -> list[torch.Tensor]:
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"]
        return self.model.get_image_embeddings(pixels.to(self.device))

    @torch.no_grad()
    def predict(
        self,
        embeddings: list[torch.Tensor],
        size: tuple[int, int],
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        masks: torch.Tensor | None = None,
        multimask: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Masks ``(N, H, W)`` bool for N objects and their low-resolution logits
        ``(N, 1, h, w)``, to feed back as ``masks``. ``points`` ``(N, K, 2)`` and ``boxes``
        ``(N, 4)`` are in tile pixels. With ``multimask``, the mask SAM scores best."""
        n = len(points if points is not None else boxes)
        scale = torch.tensor([self.side / size[1], self.side / size[0]], device=self.device)
        kwargs = {}
        if points is not None:
            pts = torch.as_tensor(points, dtype=torch.float32, device=self.device) * scale
            kwargs["input_points"] = pts[:, None]
            kwargs["input_labels"] = torch.as_tensor(labels, device=self.device)[:, None]
        if boxes is not None:
            b = torch.as_tensor(boxes, dtype=torch.float32, device=self.device) * scale.repeat(2)
            kwargs["input_boxes"] = b[:, None]
        out = self.model(
            image_embeddings=[e.expand(n, *e.shape[1:]) for e in embeddings],
            input_masks=masks,
            multimask_output=multimask,
            **kwargs,
        )
        logits = out.pred_masks[:, 0]  # (N, masks, h, w)
        best = (
            out.iou_scores[:, 0].argmax(-1)
            if multimask
            else torch.zeros_like(logits[:, 0, 0, 0]).long()
        )
        logits = logits[torch.arange(n), best][:, None]
        # As the processor: up to the model's input size, then down to the tile's.
        full = F.interpolate(logits.float(), (self.side, self.side), mode="bilinear")
        full = F.interpolate(full, size, mode="bilinear")[:, 0] > 0
        return full.cpu().numpy(), logits


def _iou(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    inter = (true & pred).sum((1, 2))
    union = (true | pred).sum((1, 2))
    return inter / np.maximum(union, 1)


def track_tile(
    tracker: Tracker,
    image: Image.Image,
    true: np.ndarray,
    max_clicks: int,
    jitters: list[float],
    rng: np.random.Generator,
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """For the ``(N, H, W)`` masks ``true`` of one tile: IoU per box jitter ``{jitter:
    (N,)}`` and the IoU after each click ``(N, max_clicks)``."""
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


def tracker_metrics(box_iou: dict[float, np.ndarray], curves: np.ndarray) -> dict[str, float]:
    """Means over objects: ``iou_box@<jitter>``, ``iou@<k>`` after k clicks, and for 85 and
    90% IoU the clicks needed (``noc@``, the maximum when never) and the share never
    reaching it (``fail@``)."""
    out = {f"iou_box@{j}": float(v.mean()) for j, v in box_iou.items()}
    out |= {f"iou@{k + 1}": float(v) for k, v in enumerate(curves.mean(0))}
    for t in (85, 90):
        reached = curves >= t / 100
        clicks = np.where(reached.any(1), reached.argmax(1) + 1, curves.shape[1])
        out[f"noc@{t}"] = float(clicks.mean())
        out[f"fail@{t}"] = float(1 - reached.any(1).mean())
    out["objects"] = len(curves)
    return out


class Exemplar:
    """``Sam3Model`` prompted with example boxes: all objects like the example."""

    def __init__(self, device: str, revision: str = SAM3_REVISION) -> None:
        from transformers import Sam3Model, Sam3Processor

        self.device = device
        self.processor = Sam3Processor.from_pretrained(SAM3_REPO, revision=revision)
        self.model = Sam3Model.from_pretrained(SAM3_REPO, revision=revision)
        self.model.to(device).eval()

    @torch.no_grad()
    def detect(
        self, image: Image.Image, boxes: list[np.ndarray], threshold: float, mask_threshold: float
    ) -> list[list[tuple[float, np.ndarray]]]:
        """Per example box, the ``(score, mask)`` detections scoring at least ``threshold``."""
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"]
        autocast = torch.autocast("cuda", torch.bfloat16, enabled=self.device == "cuda")
        out = []
        with autocast:
            vision = self.model.get_vision_features(pixel_values=pixels.to(self.device))
            for b in boxes:
                prompt = self.processor(
                    images=image,
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


def evaluate(cfg: dict, split: str, only: str | None = None) -> dict[str, float]:
    """Run the evaluations of ``cfg`` (``only`` one of them) on ``data[split]``."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = InstanceTileFolderDataset(**cfg["data"][split])
    limit = min(len(dataset), cfg.get("limit") or len(dataset))
    rng = np.random.default_rng(cfg.get("seed", 0))
    min_size = cfg.get("min_size", 10)
    run_tracker = only in (None, "tracker")
    run_exemplar = only in (None, "exemplar")
    if run_tracker:
        t_cfg = cfg.get("tracker", {})
        tracker = Tracker(device)
        jitters = t_cfg.get("box_jitter", [0.0, 0.05, 0.1])
        boxes_all, curves_all = {j: [] for j in jitters}, []
    if run_exemplar:
        e_cfg = cfg.get("exemplar", {})
        exemplar = Exemplar(device)
        thresholds = sorted(e_cfg.get("thresholds", [0.5]))
        num_classes = cfg["num_classes"]
        pq = {t: PanopticQuality(num_classes) for t in thresholds}
    for i in range(limit):
        image = Image.open(dataset.samples[i][0]).convert("RGB")
        target = dataset[i][1]
        inst, types = target["inst"].numpy(), target["type"].numpy()
        ids = [k for k in np.unique(inst) if k > 0 and (inst == k).sum() >= min_size]
        if run_tracker and ids:
            true = np.stack([inst == k for k in ids])
            box_iou, curves = track_tile(
                tracker, image, true, t_cfg.get("max_clicks", 20), jitters, rng
            )
            for j in jitters:
                boxes_all[j].append(box_iou[j])
            curves_all.append(curves)
        if run_exemplar:
            kinds = instance_types(inst, types, num_classes)
            present = sorted({int(kinds[k]) for k in ids} - {0})
            examples = [box(inst == rng.choice([k for k in ids if kinds[k] == c])) for c in present]
            found = exemplar.detect(
                image, examples, thresholds[0], e_cfg.get("mask_threshold", 0.5)
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
        boxes = {j: np.concatenate(v) for j, v in boxes_all.items()}
        result = tracker_metrics(boxes, np.concatenate(curves_all))
        metrics |= {f"{split}/tracker/{k}": v for k, v in result.items()}
    if run_exemplar:
        names = cfg.get("class_names") or [str(c) for c in range(num_classes)]
        for t in thresholds:
            r = pq[t].compute()
            metrics |= {
                f"{split}/exemplar/{k}@{t}": r[k].item() for k in ("bpq", "mpq", "dq", "sq")
            }
            per_type = zip(names[1:], r["pq"][1:], strict=True)
            metrics |= {f"{split}/exemplar/pq_{n}@{t}": v.item() for n, v in per_type}
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="zeroshot.yaml")
    parser.add_argument("--split", default="val", help="data split to score (default: val)")
    parser.add_argument("--only", choices=["tracker", "exemplar"], help="run one evaluation")
    parser.add_argument("--out", help="also write the metrics to this JSON file")
    args = parser.parse_args(argv)
    metrics = evaluate(load_config(args.config), args.split, args.only)
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
