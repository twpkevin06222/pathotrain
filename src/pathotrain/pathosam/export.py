"""Export SAM 3's point and box model to ONNX as an encoder and a decoder::

    python -m pathotrain.pathosam.export out/                     # zero-shot SAM 3
    python -m pathotrain.pathosam.export out/ --checkpoint best.ckpt --size 256

Needs the export extra (``pip install 'pathotrain[export]'``). As SAM's own ONNX export,
the heavy image encoder runs once per tile and the light decoder once per click:

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
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from pathotrain import experiment
from pathotrain.pathofm.sam3 import SAM3_REVISION, cache_rope
from pathotrain.pathosam.model import PAD, InteractiveSam, to_tile

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


def export(
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
    if checkpoint is not None:
        from pathotrain.pathosam.train import load_model

        model, cfg = load_model(checkpoint)
        revision = cfg.get("model", {}).get("revision", SAM3_REVISION)
    else:
        model, revision = InteractiveSam(), SAM3_REVISION
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
    parser.add_argument("out_dir", help="folder for encoder.onnx, decoder.onnx and pathosam.json")
    parser.add_argument(
        "--checkpoint", help="a pathosam.train checkpoint (default: zero-shot SAM 3)"
    )
    parser.add_argument("--size", type=int, default=256, help="tile side in px (default: 256)")
    args = parser.parse_args(argv)
    export(args.out_dir, args.checkpoint, args.size)


if __name__ == "__main__":
    main()
