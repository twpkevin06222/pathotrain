import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from pathotrain.dataloader.vqa import VQADataset, answer_type, export_vqa_parquet


def _jpeg(value: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def _parquet(path, rows):
    table = pa.table(
        {
            "image": [{"bytes": image, "path": None} for image, _, _ in rows],
            "question": [q for _, q, _ in rows],
            "answer": [a for _, _, a in rows],
        }
    )
    pq.write_table(table, path)
    return path


def test_export_stores_each_image_once_and_reports_shared_images(tmp_path):
    a, b, c = _jpeg(10), _jpeg(120), _jpeg(240)
    train = _parquet(
        tmp_path / "train.parquet",
        [(a, "is this a carcinoma?", "yes"), (a, "what is present?", "necrosis"), (b, "q?", "no")],
    )
    test = _parquet(tmp_path / "test.parquet", [(b, "is it benign?", "no"), (c, "where?", "lung")])

    stats = export_vqa_parquet({"train": [train], "test": [test]}, tmp_path / "out")

    assert stats["train"] == {"questions": 3, "images": 2, "shared_images": {"test": 1}}
    assert stats["test"] == {"questions": 2, "images": 2, "shared_images": {"train": 1}}
    assert len(list((tmp_path / "out/images").iterdir())) == 3  # a, b, c once each
    rows = [json.loads(line) for line in (tmp_path / "out/train.jsonl").read_text().splitlines()]
    assert [r["type"] for r in rows] == ["closed", "open", "closed"]
    assert rows[0]["image"] == rows[1]["image"] and rows[0]["image"].endswith(".jpg")

    ds = VQADataset(tmp_path / "out/train.jsonl")
    sample = ds[1]
    assert sample["question"] == "what is present?" and sample["answer"] == "necrosis"
    assert sample["image"].mode == "RGB" and sample["image"].size == (8, 8)


def test_vqa_rows_infer_the_type_and_reject_bad_rows(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "x.png")
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"image": "x.png", "question": "q", "answer": "Yes "})
        + "\n\n"
        + json.dumps({"image": "x.png", "question": "q", "answer": "a", "type": "open"})
        + "\n"
    )
    ds = VQADataset(good, transform=lambda im: im.size)
    assert [r["type"] for r in ds.rows] == ["closed", "open"] and ds[0]["image"] == (4, 4)
    assert answer_type("No") == "closed" and answer_type("nope") == "open"

    for row, match in [
        ({"image": "x.png", "question": "q"}, "lacks"),
        ({"image": "x.png", "question": "q", "answer": "a", "type": "free"}, "type"),
    ]:
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match=match):
            VQADataset(bad)
    (tmp_path / "empty.jsonl").write_text("")
    with pytest.raises(ValueError, match="no rows"):
        VQADataset(tmp_path / "empty.jsonl")
