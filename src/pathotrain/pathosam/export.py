"""Export SAM 3's point and box model as a model folder for serving, or to ONNX::

    python -m pathotrain.pathosam.export out/ --checkpoint best.ckpt --mpp 0.25 \\
        --register pathosam                                       # the model folder
    python -m pathotrain.pathosam.export out/ --mpp 0.25          # zero-shot SAM 3
    python -m pathotrain.pathosam.export onnx/ --checkpoint best.ckpt --onnx

The model folder (:func:`export`), for patholamina's sam3 service: ``model.safetensors``
(SAM 3 with the fine-tuned tracker merged in) with SAM 3's config and processor files, and
``model_card.json`` (task ``promptable_segmentation``, ``input.mpp`` the training mpp,
``prompt_vocabulary``: ``[{phrase, synonyms, class_id}]``, one class per ``class_id``). ``--register NAME`` adds it to the MLflow registry with the card at
the version's artifact root, as patholamina's ``register_model.py`` expects.

ONNX (:func:`export_onnx`, ``--onnx``) needs the export extra
(``pip install 'pathotrain[export]'``). As SAM's own ONNX export, the heavy image encoder
runs once per tile and the light decoder once per click:

- ``encoder.onnx`` (1.9 GB; weights over 2 GB would go to ``encoder.onnx.data`` beside it):
  ``image`` (batch, 3, size, size)
  float32, the tile normalised with mean = std = 0.5, to ``fpn0`` (batch, 32, 288, 288),
  ``fpn1`` (batch, 64, 144, 144) and ``fpn2`` (batch, 256, 72, 72).
- ``decoder.onnx``, for N objects of one tile:

  - ``fpn0``, ``fpn1``, ``fpn2``: the tile's encoder outputs, repeated N times;
  - ``points`` (N, K, 2) float32, ``(x, y)`` in tile pixels, with ``labels`` (N, K) int64:
    1 positive click, 0 negative click, 2 and 3 a box's top-left and bottom-right corners
    (after the clicks), -10 padding;
  - ``mask_input`` (N, 1, 288, 288) float32, the previous round's ``low_res`` logits, and
    ``has_mask`` (N,) float32, 1 where it holds one, else 0;

  to ``low_res`` (N, 1, 288, 288) and ``iou`` (N, 1), the single mask (SAM's choice: token 0
  unless it is unstable), and ``multi_low_res`` (N, 3, 288, 288) and ``multi_iou`` (N, 3),
  three candidate masks for an ambiguous single click (keep the best ``multi_iou``).

Mask logits are at 288 x 288: resize them bilinearly to 1008 x 1008, then to the tile, and
threshold at 0 (:func:`pathotrain.pathosam.model.to_tile`). :class:`OnnxTracker` runs the
pair as :class:`pathotrain.pathosam.eval.Tracker` runs the PyTorch model, so
``pathosam.eval --onnx out/`` scores the export with the same click protocol.

Export checks both graphs against PyTorch with ONNX Runtime before returning, and writes
``pathosam.json`` (the metadata, also stored in each graph) beside them. The weights are SAM
3's, so the files fall under the SAM License; a fine-tuned checkpoint also carries its
training data's terms.
"""

import argparse
import itertools
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import Tensor, nn

from patchology import TilingConfig
from pathotrain import experiment
from pathotrain.engine import MLflowLogger
from pathotrain.engine.logger import sqlite_store
from pathotrain.pathofm.sam3 import SAM3_REPO, SAM3_REVISION, cache_rope
from pathotrain.pathosam.model import PAD, InteractiveSam, to_tile

# Text prompts SAM 3 finds nuclei with on H&E (PanNuke presence: "cell" 0.99, "neoplastic
# cell" 0.85; "nucleus" 0.08 and "tumor cell" 0.36 are left out).
PROMPT_VOCABULARY = ("cell=nucleus,nuclei", "neoplastic cell=tumour cell,tumor cell")
"""Default ``--prompt`` values: ``phrase[=synonym,...][:class_id]``."""
IMAGE_SIZE = 1008
"""SAM 3 resizes every tile to this side inside the model."""
FPN = ("fpn0", "fpn1", "fpn2")
DECODER_INPUTS = (*FPN, "points", "labels", "mask_input", "has_mask")
DECODER_OUTPUTS = ("low_res", "iou", "multi_low_res", "multi_iou")
BOX_LABELS = (2, 3)  # a box as its top-left and bottom-right corners


class _Encoder(nn.Module):
    def __init__(self, model: InteractiveSam) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(self.model(images=image))


class _Decoder(nn.Module):
    """The prompt encoder and mask decoder for N objects with points (a box as two corner
    points), an optional previous mask, and both the single and the three-mask outputs."""

    def __init__(self, model: InteractiveSam, size: int) -> None:
        super().__init__()
        self.sam, self.scale = model.sam, model.side / size

    def forward(self, fpn0, fpn1, fpn2, points, labels, mask_input, has_mask):
        sam, n = self.sam, points.shape[0]
        sparse, dense = sam.prompt_encoder(
            input_points=(points * self.scale)[:, None],
            input_labels=labels[:, None],
            input_boxes=None,
            input_masks=mask_input,
        )
        no_mask = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand_as(dense)
        dense = torch.where(has_mask[:, None, None, None] > 0.5, dense, no_mask)
        decode = lambda multimask: sam.mask_decoder(
            image_embeddings=fpn2,
            image_positional_embeddings=sam.get_image_wide_positional_embeddings().repeat(
                n, 1, 1, 1
            ),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
            high_resolution_features=[fpn0, fpn1],
        )
        single, multi = decode(False), decode(True)
        return single[0][:, 0], single[1][:, 0], multi[0][:, 0], multi[1][:, 0]


def _decoder_inputs(
    embeddings: list[Tensor], size: int, n: int = 3, k: int = 4
) -> tuple[Tensor, ...]:
    """Example decoder inputs for N objects with K points: clicks, padding, a box."""
    g = torch.Generator().manual_seed(0)
    points = torch.rand(n, k, 2, generator=g) * size
    labels = torch.tensor([[1, 0, PAD, 1], [1, 1, 2, 3], [0, 1, 1, PAD]])[:n, :k]
    mask_input = torch.randn(n, 1, *embeddings[0].shape[-2:], generator=g)
    has_mask = torch.tensor([0.0, 1.0, 1.0])[:n]
    fpn = [e.expand(n, *e.shape[1:]).contiguous() for e in embeddings]
    return (*fpn, points, labels, mask_input, has_mask)


def _load(checkpoint: str | Path | None) -> tuple[InteractiveSam, dict, str]:
    """The model, its training config (``{}`` zero-shot) and its SAM 3 revision."""
    if checkpoint is None:
        return InteractiveSam(), {}, SAM3_REVISION
    from pathotrain.pathosam.train import load_model

    model, cfg = load_model(checkpoint)
    return model, cfg, cfg.get("model", {}).get("revision", SAM3_REVISION)


def _sam3_snapshot(revision: str) -> Path:
    """The local ``facebook/sam3`` snapshot (config, processor, tokenizer, weights).
    Not ``sam3.pt``: Meta's original 3.5 GB checkpoint, which transformers never reads (it
    was downloaded, then copied into every exported folder)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(SAM3_REPO, revision=revision, ignore_patterns=["*.pt"]))


def merge_tracker(tracker: nn.Module, base: dict[str, Tensor]) -> dict[str, Tensor]:
    """SAM 3's full state dict ``base`` (detector and tracker) with ``tracker``'s
    (``Sam3TrackerModel``) weights put in, so text, box and click prompts all load from it.

    Raises:
        KeyError: a tracker weight has no place in ``base``.
    """
    merged = dict(base)
    for key, value in tracker.state_dict().items():
        if key.startswith("vision_encoder.backbone."):  # the ViT is the detector's
            full = f"detector_model.{key}"
        elif key.startswith("vision_encoder.neck."):
            full = "tracker_neck." + key.removeprefix("vision_encoder.neck.")
        else:
            full = f"tracker_model.{key}"
        if full not in merged:
            raise KeyError(f"tracker weight {key} has no {full} in SAM 3's checkpoint")
        merged[full] = value.detach().to("cpu", merged[full].dtype).contiguous()
    return merged


def training_mpp(cfg: dict, mpp: float | None) -> float:
    """The resolution the model trained at: the config's when it tiled slides, else ``mpp``.

    Raises:
        ValueError: ``mpp`` unknown, or it contradicts the training config.
    """
    train = cfg.get("data", {}).get("train", {})
    if "slides" in train:
        trained = TilingConfig(**train.get("tiling", {})).tile_mpp
        if mpp is not None and mpp != trained:
            raise ValueError(f"the model trained at {trained} mpp; got mpp={mpp}")
        return trained
    if mpp is None:
        raise ValueError(
            "tile folders (and zero-shot SAM 3) record no resolution: pass mpp (--mpp)"
        )
    return mpp


def vocabulary(prompts: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    """``phrase[=synonym,...][:class_id]`` prompts → the card's ``prompt_vocabulary``. A
    prompt without ``:class_id`` gets the next unused id; phrases and synonyms must be unique
    (case-insensitive), as the sam3 service maps any of them back to one entry."""
    entries: list[dict[str, Any]] = []
    for prompt in prompts:
        text, _, class_id = prompt.partition(":")
        phrase, _, synonyms = text.partition("=")
        used = {e["class_id"] for e in entries}
        entries.append(
            {
                "phrase": phrase.strip(),
                "synonyms": [s.strip() for s in synonyms.split(",") if s.strip()],
                "class_id": int(class_id)
                if class_id.strip()
                else next(i for i in itertools.count() if i not in used),
            }
        )
    words = [w.lower() for e in entries for w in (e["phrase"], *e["synonyms"])]
    if not entries or not all(e["phrase"] for e in entries):
        raise ValueError("every prompt needs a phrase")
    if len(words) != len(set(words)):
        raise ValueError(f"phrases and synonyms must be unique: {words}")
    return entries


def export(
    out_dir: str | Path,
    checkpoint: str | Path | None = None,
    mpp: float | None = None,
    size: int = 256,
    prompt_vocabulary: list[str] | tuple[str, ...] = PROMPT_VOCABULARY,
    name: str | None = None,
    register: str | None = None,
    mlflow_db: str | Path | None = None,
) -> Path:
    """Write the model folder for serving (patholamina's sam3 service) to ``out_dir``:

    - ``model.safetensors``: SAM 3 (``facebook/sam3`` at the checkpoint's revision) with the
      fine-tuned tracker merged in (:func:`merge_tracker`), with SAM 3's ``config.json``,
      processor and tokenizer files and ``LICENSE``, so ``Sam3TrackerModel`` (clicks, boxes)
      and ``Sam3Model`` (text) both load the folder with ``from_pretrained``;
    - ``model_card.json``: task ``promptable_segmentation``, ``input.mpp`` (the training
      mpp, :func:`training_mpp`), ``input.tile_size`` and ``input.image_size``, and
      ``prompt_vocabulary`` (:func:`vocabulary`), the text prompts SAM 3 finds nuclei with on
      H&E, each with synonyms and the class it suggests (``classes``).

    With ``register``, also log the folder to MLflow and register it as a new version of
    that model, with the card at the version's artifact root, as patholamina's
    ``register_model.py`` does. Returns ``out_dir``."""
    from safetensors.torch import load_file, save_file

    model, cfg, revision = _load(checkpoint)
    mpp = training_mpp(cfg, mpp)
    entries = vocabulary(prompt_vocabulary)
    snapshot = _sam3_snapshot(revision)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for file in snapshot.iterdir():
        if file.name != "model.safetensors":
            shutil.copyfile(file, out / file.name)
    merged = merge_tracker(model.sam, load_file(snapshot / "model.safetensors"))
    save_file(merged, out / "model.safetensors", metadata={"format": "pt"})
    run_id = experiment.checkpoint_run_id(checkpoint) if checkpoint else None
    card = {
        "schema_version": 1,
        "name": name or register or cfg.get("name", "sam3"),
        "version": "1",
        "task": "promptable_segmentation",
        "framework": "transformers",
        "file": "model.safetensors",
        "precision": "fp32",
        "input": {
            "tile_size": size,
            "image_size": IMAGE_SIZE,
            "mpp": mpp,
            "channel_order": "RGB",
            "normalisation": {"mean": [0.5] * 3, "std": [0.5] * 3, "scale": 1 / 255},
        },
        "prompt_vocabulary": entries,
        # One class per class_id, named after the first phrase that suggests it.
        "classes": [
            {"id": cid, "name": next(e["phrase"] for e in entries if e["class_id"] == cid)}
            for cid in sorted({e["class_id"] for e in entries})
        ],
        "provenance": {
            "training_run_id": run_id,
            "base_model": f"{SAM3_REPO}@{revision}",
            "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
        },
        "licence": "SAM License (facebook/sam3); a fine-tuned checkpoint adds its data's terms",
    }
    (out / experiment.CARD_FILE).write_text(json.dumps(card, indent=2) + "\n")
    if register is not None:
        _register(out, register, cfg, mlflow_db)
    return out


def _register(folder: Path, name: str, cfg: dict, mlflow_db: str | Path | None) -> str:
    """Log ``folder`` as the ``model`` artifacts of an ``export`` run and register them as a
    new version of ``name``; returns the version. The store is ``mlflow_db`` (default the
    config's), else ``MLFLOW_TRACKING_URI``."""
    from mlflow import MlflowClient

    mlflow_db = mlflow_db or cfg.get("mlflow_db")
    run = MLflowLogger(
        cfg.get("name", "pathosam"), "export", **(sqlite_store(mlflow_db) if mlflow_db else {})
    )
    run.log_hyperparams({"folder": str(folder.resolve())})  # starts the run
    run_id = run.run_id
    client = MlflowClient()  # the tracking URI the run set
    client.log_artifacts(run_id, str(folder), artifact_path="model")
    if not client.search_registered_models(filter_string=f"name = '{name}'"):
        client.create_registered_model(name)
    # Plain run artifacts, not a logged MLflow model: MLflow 3 registers them this way.
    uri = f"{client.get_run(run_id).info.artifact_uri}/model"
    version = client.create_model_version(name, uri, run_id=run_id).version
    run.finalize("success")
    logger.info(f"registered {folder} as {name} version {version}")
    return str(version)


def export_onnx(
    out_dir: str | Path,
    checkpoint: str | Path | None = None,
    size: int = 256,
    atol: float = 1e-3,
    encoder_atol: float = 1e-2,
) -> Path:
    """Write ``encoder.onnx``, ``decoder.onnx`` and ``pathosam.json`` to ``out_dir`` for
    ``size`` px tiles, from ``checkpoint`` (a :mod:`pathotrain.pathosam.train` checkpoint)
    or zero-shot SAM 3, and check both against PyTorch (the decoder to ``atol``, the encoder
    to ``encoder_atol``, relative to each output's largest magnitude above 1). Returns
    ``out_dir``."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "ONNX export needs the export extra: pip install 'pathotrain[export]'"
        ) from e
    model, _, revision = _load(checkpoint)
    model = model.eval().float()
    cache_rope(model.sam.vision_encoder.backbone)  # SAM 3's own RoPE does not export
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "task": "interactive_segmentation",
        "model": "facebook/sam3 point and box model (Sam3TrackerModel)",
        "revision": revision,
        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
        "tile_size": size,
        "normalisation": {"mean": [0.5] * 3, "std": [0.5] * 3, "scale": 1 / 255},
        "encoder": {"inputs": ["image"], "outputs": list(FPN)},
        "decoder": {"inputs": list(DECODER_INPUTS), "outputs": list(DECODER_OUTPUTS)},
        "labels": {"positive": 1, "negative": 0, "box": list(BOX_LABELS), "padding": PAD},
        "licence": "SAM License (facebook/sam3); a fine-tuned checkpoint adds its data's terms",
    }
    # A 32-layer ViT at 1008 px drifts more in float32: on random input up to 5e-3; on a real
    # PanNuke tile at most 1.3e-3 (3e-4 of each output's range), on CPU.
    experiment.export_onnx(
        _Encoder(model), out / "encoder.onnx", size, list(FPN), meta, atol=encoder_atol
    )

    with torch.no_grad():
        embeddings = model(images=torch.randn(1, 3, size, size))
        example = _decoder_inputs(embeddings, size)
        decoder = _Decoder(model, size).eval()
        torch.onnx.export(
            decoder,
            example,
            str(out / "decoder.onnx"),
            input_names=list(DECODER_INPUTS),
            output_names=list(DECODER_OUTPUTS),
            dynamic_axes={
                **{name: {0: "objects"} for name in (*DECODER_INPUTS, *DECODER_OUTPUTS)},
                "points": {0: "objects", 1: "points"},
                "labels": {0: "objects", 1: "points"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        graph = onnx.load(str(out / "decoder.onnx"))
        graph.metadata_props.add(key="pathotrain", value=json.dumps(meta))
        onnx.save(graph, str(out / "decoder.onnx"))
        # Other object and point counts than traced: checks the dynamic axes.
        check = _decoder_inputs(embeddings, size, n=2, k=3)
        expected = decoder(*check)
    session = ort.InferenceSession(str(out / "decoder.onnx"), providers=["CPUExecutionProvider"])
    got = session.run(
        None, {name: t.numpy() for name, t in zip(DECODER_INPUTS, check, strict=True)}
    )
    for name, want, have in zip(DECODER_OUTPUTS, expected, got, strict=True):
        scale = max(1.0, float(want.abs().max()))
        err = float(np.abs(want.numpy() - have).max())
        assert err <= atol * scale, f"decoder output {name} differs from PyTorch by {err:.2e}"
    (out / "pathosam.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


class OnnxTracker:
    """The exported encoder and decoder behind :class:`pathotrain.pathosam.eval.Tracker`'s
    interface (``encode``, ``predict``), so ``track_tile`` runs them unchanged."""

    def __init__(self, folder: str | Path, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        folder = Path(folder)
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(folder / "encoder.onnx"), providers=providers)
        self.decoder = ort.InferenceSession(str(folder / "decoder.onnx"), providers=providers)
        self.size = json.loads((folder / "pathosam.json").read_text())["tile_size"]
        self.mask_shape = self.decoder.get_inputs()[DECODER_INPUTS.index("mask_input")].shape[-2:]

    def encode(self, image: Tensor) -> list[np.ndarray]:
        """Encoder outputs of one normalised tile ``(3, H, W)``."""
        return self.encoder.run(None, {"image": image[None].cpu().numpy()})

    def predict(
        self,
        embeddings: list[np.ndarray],
        size: tuple[int, int],
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        masks: Tensor | None = None,
        multimask: bool = False,
    ) -> tuple[np.ndarray, Tensor]:
        """As ``Tracker.predict``: masks ``(N, H, W)`` bool and low-resolution logits to
        feed back. A box becomes two corner points after the clicks."""
        if tuple(size) != (self.size, self.size):
            raise ValueError(f"the export takes {self.size} px tiles, got {tuple(size)}")
        n = len(points if points is not None else boxes)
        pts, lab = np.zeros((n, 0, 2), np.float32), np.zeros((n, 0), np.int64)
        if points is not None:
            pts, lab = np.asarray(points, np.float32), np.asarray(labels, np.int64)
        if boxes is not None:
            corners = np.asarray(boxes, np.float32).reshape(n, 2, 2)
            pts = np.concatenate([pts, corners], 1)
            lab = np.concatenate([lab, np.tile(BOX_LABELS, (n, 1))], 1)
        mask_input = np.zeros((n, 1, *self.mask_shape), np.float32)
        has_mask = np.zeros(n, np.float32)
        if masks is not None:
            mask_input, has_mask = masks.float().cpu().numpy(), np.ones(n, np.float32)
        feeds = dict(zip(FPN, (np.repeat(e, n, 0) for e in embeddings), strict=True))
        feeds |= {"points": pts, "labels": lab, "mask_input": mask_input, "has_mask": has_mask}
        low_res, _, multi, multi_iou = self.decoder.run(None, feeds)
        logits = multi[np.arange(n), multi_iou.argmax(1)][:, None] if multimask else low_res
        logits = torch.from_numpy(np.ascontiguousarray(logits))
        return (to_tile(logits, tuple(size))[:, 0] > 0).numpy(), logits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", help="the model folder to write")
    parser.add_argument(
        "--checkpoint", help="a pathosam.train checkpoint (default: zero-shot SAM 3)"
    )
    parser.add_argument("--size", type=int, default=256, help="tile side in px (default: 256)")
    parser.add_argument(
        "--mpp", type=float, help="training resolution; required unless it tiled slides"
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="a text prompt for the card's prompt_vocabulary, repeatable: "
        "phrase[=synonym,...][:class_id] "
        f"(default: {' '.join(repr(p) for p in PROMPT_VOCABULARY)})",
    )
    parser.add_argument("--name", help="the card's model name (default: --register, else run)")
    parser.add_argument("--register", help="also register the folder in MLflow under this name")
    parser.add_argument(
        "--mlflow-db", help="MLflow SQLite file for --register (default: the config's mlflow_db)"
    )
    parser.add_argument(
        "--onnx", action="store_true", help="write encoder.onnx and decoder.onnx instead"
    )
    args = parser.parse_args(argv)
    if args.onnx:
        export_onnx(args.out_dir, args.checkpoint, args.size)
        return
    export(
        args.out_dir,
        args.checkpoint,
        args.mpp,
        args.size,
        args.prompts or PROMPT_VOCABULARY,
        args.name,
        args.register,
        args.mlflow_db,
    )


if __name__ == "__main__":
    main()
