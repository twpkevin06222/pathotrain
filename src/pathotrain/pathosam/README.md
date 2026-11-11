# pathotrain - pathosam

> [!WARNING]
> For research use only. Not a medical device.

SAM 3 for prompted and interactive segmentation of pathology images: an annotator clicks
or boxes a cell and gets its mask. The package builds on transformers' SAM 3 models. So
far it has zero-shot baselines, with prompts simulated from the ground truth; fine-tuning
comes next.

SAM 3 (`facebook/sam3`) is gated on Hugging Face and released under the SAM License (see
the [pathofm README](../pathofm/README.md#sam-3-dense-backbone-gated)).

## Zero-shot baselines

```bash
cp src/pathotrain/pathosam/configs/zeroshot.example.yaml zeroshot.yaml   # set the paths
python -m pathotrain.pathosam.eval zeroshot.yaml --split val --out val.json
python -m pathotrain.pathosam.eval zeroshot.yaml --split test --only tracker
```

The data is an `InstanceTileFolderDataset` (see the dataloader README). Each tile is
encoded once and reused for every prompt.

**`tracker`: SAM 3's point and box model (`Sam3TrackerModel`), one nucleus per prompt.**
Every nucleus of at least `min_size` px is prompted with:

- a box, for each jitter in `tracker.box_jitter` (each side moved by up to that share of
  the box's size);
- a click at its centre (the pixel farthest from its boundary), then corrective clicks up
  to `tracker.max_clicks`. Each round adds one click and feeds the previous mask back in.

A corrective click follows RITM's NoC protocol: it goes to the deepest pixel of either the
missed or the extra region, whichever is deeper. A click in a missed region is positive,
one in an extra region negative. The simulation is in `pathosam.prompts`, so fine-tuning
can reuse it.

| Metric | Meaning |
|---|---|
| `iou_box@<jitter>` | mean IoU from one box |
| `iou@<k>` | mean IoU after k clicks |
| `noc@85`, `noc@90` | mean clicks to reach 85% or 90% IoU (`max_clicks` when never) |
| `fail@85`, `fail@90` | share of nuclei that never reach it |

**`exemplar`: SAM 3's concept model (`Sam3Model`) with one example box per nucleus type.**
For each type present in a tile, the box of one of its nuclei asks for "all nuclei like
this one". Every detection takes that type, and overlaps go to the higher score. The result
is scored like `cellseg.eval` (`bpq`, `mpq`, `pq_<type>`, `dq`, `sq`) at every threshold in
`exemplar.thresholds`, so it compares directly with the trained cellseg models. The example
nucleus itself counts, as an annotator's box would.

Metrics are keyed `<split>/<evaluation>/<metric>`.

## Notes

- SAM 3's point/box model reads the same image-encoder ViT, with identical weights, as
  `pathofm`'s `sam3` backbone. An encoder fine-tuned in cellseg could therefore be loaded
  into it (not done yet).
- SAM 3 resizes every tile to 1008 px, so a 256 px PanNuke tile is upsampled about 4x.
