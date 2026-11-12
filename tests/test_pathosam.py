import numpy as np
import pytest
import torch

from pathotrain.pathosam import box, center_click, corrective_click
from pathotrain.pathosam.eval import ClickMetrics
from pathotrain.pathosam.model import InteractiveSam
from pathotrain.pathosam.task import InteractiveTask, _boxes, _dice, _sample


def _square(y0, y1, x0, x1, size=32):
    mask = np.zeros((size, size), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_box_is_exclusive_and_jitter_stays_within_bounds():
    mask = _square(4, 14, 8, 28)  # 20 wide, 10 high
    assert box(mask).tolist() == [8, 4, 28, 14]
    rng = np.random.default_rng(0)
    for _ in range(100):
        x0, y0, x1, y1 = box(mask, 0.1, rng)
        assert abs(x0 - 8) <= 2 and abs(x1 - 28) <= 2 and abs(y0 - 4) <= 1 and abs(y1 - 14) <= 1
    edge = box(_square(0, 5, 0, 5), 0.5, np.random.default_rng(1))
    assert edge.min() >= 0 and edge.max() <= 32  # clipped to the tile


def test_center_click_is_the_deepest_pixel_and_tile_edges_count_as_boundary():
    assert center_click(_square(10, 21, 4, 15)) == (9.0, 15.0)
    x, _ = center_click(_square(0, 32, 0, 3))  # a strip along the left edge
    assert x == 1.0


def test_corrective_click_goes_to_the_deeper_error():
    true = _square(4, 24, 4, 24)
    missed = true.copy()
    missed[4:24, 14:24] = False  # half missed, nothing extra
    assert corrective_click(true, missed)[2] == 1
    extra = true | _square(4, 24, 24, 32)  # a deep extra region, nothing missed
    x, _, label = corrective_click(true, extra)
    assert label == 0 and x >= 24
    small_miss = true.copy()
    small_miss[4, 4] = False  # a 1 px miss beside a deep extra: the extra wins
    assert corrective_click(true, small_miss | _square(4, 24, 24, 32))[2] == 0
    assert corrective_click(true, true) is None


def test_click_metrics_count_clicks_and_failures():
    curves = np.array(
        [
            [0.80, 0.86, 0.91],  # 85% at click 2, 90% at click 3
            [0.92, 0.95, 0.96],  # both at click 1
            [0.50, 0.60, 0.70],  # never: counts as 3 clicks
        ]
    )
    metric = ClickMetrics([0.0], 3)
    metric.update({0.0: np.array([0.9, 0.8])}, curves[:2])  # summed over updates (and ranks)
    metric.update({0.0: np.array([0.7])}, curves[2:])
    m = metric.compute()
    assert m["iou_box_0.0"] == pytest.approx(0.8)
    assert m["iou_click1"] == pytest.approx(0.74)
    assert m["noc85"] == pytest.approx((2 + 1 + 3) / 3)
    assert m["noc90"] == pytest.approx((3 + 1 + 3) / 3)
    assert m["fail90"] == pytest.approx(1 / 3) and m["objects"] == 3


def test_training_boxes_match_the_eval_boxes_and_clicks_land_in_the_region():
    masks = torch.from_numpy(np.stack([_square(4, 14, 8, 28), _square(0, 5, 0, 5)]))
    assert _boxes(masks, 0.0).tolist() == [box(m).tolist() for m in masks.numpy()]
    region = torch.from_numpy(np.stack([_square(10, 12, 20, 22), np.zeros((32, 32), bool)]))
    for _ in range(20):
        xy, some = _sample(region)
        x, y = xy[0].long()
        assert region[0, y, x] and some.tolist() == [True, False]
    assert _dice(torch.full((1, 4, 4), 20.0), torch.ones(1, 4, 4)).item() < 1e-3


def _tiny_sam(monkeypatch):
    """InteractiveSam on a tiny random Sam3TrackerModel (112 px input), built offline."""
    from transformers import Sam3TrackerConfig, Sam3TrackerModel, Sam3VisionConfig, Sam3ViTConfig

    vit = Sam3ViTConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        global_attn_indexes=[1, 3],
        window_size=4,
        image_size=112,
        pretrain_image_size=56,
    )
    vision = Sam3VisionConfig(
        backbone_config=vit, fpn_hidden_size=32, backbone_feature_sizes=[[32, 32], [16, 16], [8, 8]]
    )
    config = Sam3TrackerConfig(
        vision_config=vision,
        prompt_encoder_config={"hidden_size": 32, "image_size": 112, "patch_size": 14},
        mask_decoder_config={
            "hidden_size": 32,
            "num_attention_heads": 4,
            "mlp_dim": 64,
            "iou_head_hidden_dim": 32,
        },
    )

    def build(*args, **kwargs):
        torch.manual_seed(0)
        return Sam3TrackerModel(config)

    monkeypatch.setattr(Sam3TrackerModel, "from_pretrained", build)
    return InteractiveSam()


def _batch():
    inst = torch.zeros(2, 64, 64, dtype=torch.long)
    inst[0, 4:20, 4:20], inst[0, 30:50, 10:30], inst[1, 8:40, 20:44] = 1, 2, 5
    return torch.randn(2, 3, 64, 64), {"inst": inst, "type": (inst > 0).long()}


@pytest.mark.parametrize("box_share", [0.0, 1.0])  # a first click, or a first box
def test_training_step_trains_the_decoder_only(monkeypatch, box_share):
    task = InteractiveTask(_tiny_sam(monkeypatch), objects=2, rounds=3, box_share=box_share)
    loss, logs = task.training_step(_batch())
    assert torch.isfinite(loss) and set(logs) == {"loss", "focal", "dice", "iou"}
    loss.backward()
    sam = task.model.sam
    assert all(p.grad is None for p in sam.vision_encoder.parameters())
    assert any(p.grad is not None for p in sam.mask_decoder.parameters())


def test_iou_loss_trains_the_predicted_iou_of_every_mask(monkeypatch):
    # Inference keeps the mask with the best predicted IoU, so every mask's score must train.
    task = InteractiveTask(_tiny_sam(monkeypatch), objects=2, rounds=1, box_share=0.0)
    task.weights = {"focal": 0.0, "dice": 0.0, "iou": 1.0}
    scores = []

    def keep(module, args, kwargs, output):
        if isinstance(output, tuple):  # a decoder call: (mask logits, predicted IoUs)
            output[1].retain_grad()
            scores.append(output[1])

    task.model.register_forward_hook(keep, with_kwargs=True)
    loss, _ = task.training_step(_batch())
    loss.backward()
    (iou_pred,) = scores  # one round, three masks per object
    assert iou_pred.shape[1] == 3 and (iou_pred.grad.abs() > 0).all()


def test_validation_scores_clicks_and_checkpoints_hold_the_trained_parts(monkeypatch, tmp_path):
    from mlflow.utils.validation import _validate_metric_name

    from pathotrain.engine import trainable_state
    from pathotrain.pathosam.train import load_model

    task = InteractiveTask(_tiny_sam(monkeypatch), val_clicks=3)
    with torch.no_grad():
        task.validation_step(_batch())
    metrics = task.compute_metrics()
    assert {"val/iou_box_0.0", "val/iou_click3", "val/noc90", "val/fail85"} <= set(metrics)
    for name in metrics:  # MLflow refuses some characters (e.g. "@") and stops the run
        _validate_metric_name(name)
    state = trainable_state(task.model)
    assert state and not any(k.startswith("sam.vision_encoder.") and "weight" in k for k in state)
    for v in state.values():
        if v.is_floating_point():
            v.add_(1.0)  # differ from a freshly built model
    torch.save({"model": state, "config": {}}, tmp_path / "m.ckpt")
    loaded, _ = load_model(tmp_path / "m.ckpt")
    for k, v in state.items():
        torch.testing.assert_close(loaded.state_dict()[k], v)


def test_exemplar_sends_each_box_with_its_types_text(monkeypatch, tmp_path):
    from PIL import Image

    from pathotrain.pathosam import eval as pathosam_eval

    inst = np.zeros((32, 32), np.uint8)
    inst[2:10, 2:10], inst[20:30, 20:30] = 1, 2
    types = np.where(inst == 1, 1, np.where(inst == 2, 4, 0)).astype(np.uint8)
    for folder, array in (
        ("img", np.zeros((32, 32, 3), np.uint8)),
        ("inst", inst),
        ("type", types),
    ):
        (tmp_path / folder).mkdir()
        Image.fromarray(array).save(tmp_path / folder / "a.png")
    calls = []

    class FakeExemplar:  # finds exactly the true nuclei of the example's type
        def __init__(self, device):
            pass

        def detect(self, image, boxes, threshold, mask_threshold, texts=None):
            calls.append(texts)
            return [[(0.9, inst == (1 if t == "inflammatory cell" else 2))] for t in texts]

    monkeypatch.setattr(pathosam_eval, "Exemplar", FakeExemplar)
    data = {"images": tmp_path / "img", "instances": tmp_path / "inst", "types": tmp_path / "type"}
    cfg = {
        "num_classes": 6,
        "data": {"val": data},
        "exemplar": {
            "thresholds": [0.5],
            "prompts": {1: "inflammatory cell", 4: "epithelial cell"},
        },
    }
    metrics = pathosam_eval.evaluate(cfg, "val", only="exemplar")
    assert calls == [["inflammatory cell", "epithelial cell"]]  # types present, in order
    assert metrics["val/exemplar_text/bpq@0.5"] == 1 and metrics["val/exemplar_text/mpq@0.5"] == 1


def test_onnx_export_runs_a_click_session_like_pytorch(monkeypatch, tmp_path):
    pytest.importorskip("onnxruntime")
    import json

    from pathotrain.pathosam.eval import Tracker, track_tile
    from pathotrain.pathosam.export import OnnxTracker, export_onnx

    model = _tiny_sam(monkeypatch)
    monkeypatch.setattr("pathotrain.pathosam.export.InteractiveSam", lambda: model)
    out = export_onnx(tmp_path / "onnx", size=64)
    meta = json.loads((out / "pathosam.json").read_text())
    assert meta["tile_size"] == 64 and meta["labels"]["box"] == [2, 3]
    assert {p.name for p in out.iterdir()} >= {"encoder.onnx", "decoder.onnx", "pathosam.json"}

    image, target = _batch()
    true = np.stack([target["inst"][0].numpy() == k for k in (1, 2)])
    runs = []
    for tracker in (Tracker(model.eval()), OnnxTracker(out, ["CPUExecutionProvider"])):
        rng = np.random.default_rng(0)  # same box jitter for both
        runs.append(track_tile(tracker, image[0], true, 4, [0.0, 0.1], rng))
    (box_t, curve_t), (box_o, curve_o) = runs
    for j in (0.0, 0.1):
        np.testing.assert_allclose(box_o[j], box_t[j], atol=0.02)
    np.testing.assert_allclose(curve_o, curve_t, atol=0.02)  # IoU after each click


def test_model_folder_merges_the_tracker_into_sam3_and_registers_with_its_card(
    monkeypatch, tmp_path
):
    import json

    from mlflow import MlflowClient
    from mlflow.artifacts import load_text
    from safetensors.torch import load_file, save_file

    from pathotrain.engine.logger import sqlite_store
    from pathotrain.pathosam import export as pathosam_export

    model = _tiny_sam(monkeypatch)
    monkeypatch.setattr(pathosam_export, "InteractiveSam", lambda: model)
    trained = {k: v.clone() for k, v in model.sam.state_dict().items()}
    # A fake facebook/sam3 snapshot: the full checkpoint's key layout, other weights.
    names = {
        "vision_encoder.backbone.": "detector_model.vision_encoder.backbone.",
        "vision_encoder.neck.": "tracker_neck.",
    }

    def full(key):
        for prefix, to in names.items():
            if key.startswith(prefix):
                return to + key.removeprefix(prefix)
        return f"tracker_model.{key}"

    snapshot = tmp_path / "sam3"
    snapshot.mkdir()
    base = {full(k): torch.zeros_like(v) for k, v in trained.items()}
    base["detector_model.text_encoder.w"] = torch.ones(2)
    save_file(base, snapshot / "model.safetensors")
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(pathosam_export, "_sam3_snapshot", lambda revision: snapshot)

    with pytest.raises(ValueError, match="mpp"):
        pathosam_export.export(tmp_path / "out")
    db = tmp_path / "mlflow.db"
    out = pathosam_export.export(
        tmp_path / "out", mpp=0.25, prompt_vocabulary=["cell"], register="sam", mlflow_db=db
    )
    assert {p.name for p in out.iterdir()} == {
        "model.safetensors",
        "config.json",
        "model_card.json",
    }
    merged = load_file(out / "model.safetensors")
    assert merged.keys() == base.keys()
    assert torch.equal(merged["detector_model.text_encoder.w"], torch.ones(2))
    for key, value in trained.items():
        assert torch.equal(merged[full(key)], value), key
    card = json.loads((out / "model_card.json").read_text())
    assert card["task"] == "interactive_segmentation" and card["file"] == "model.safetensors"
    assert card["input"]["mpp"] == 0.25 and card["prompt_vocabulary"] == ["cell"]

    client = MlflowClient(sqlite_store(db)["tracking_uri"])
    uri = client.get_model_version_download_uri("sam", "1")
    assert json.loads(load_text(f"{uri}/model_card.json")) == card


def test_example_config_is_valid():
    import inspect
    from importlib.resources import files

    from pathotrain.dataloader import InstanceTileFolderDataset
    from pathotrain.engine import load_config

    configs = files("pathotrain.pathosam") / "configs"
    args = set(inspect.signature(InstanceTileFolderDataset.__init__).parameters)
    cfg = load_config(configs / "zeroshot.example.yaml")
    for split in ("val", "test"):
        assert set(cfg["data"][split]) <= args
    assert len(cfg["class_names"]) == cfg["num_classes"]
    train = load_config(configs / "interactive.example.yaml")
    assert set(train["task"]) <= set(inspect.signature(InteractiveTask.__init__).parameters)
    for split in ("train", "val"):
        assert set(train["data"][split]) <= args
