import pytest
import torch

from pathotrain.detection.train import check_eval_crop
from pathotrain.segmentation.task import SegmentationTask, _starts, sliding_window
from pathotrain.segmentation.train import task_kwargs


@pytest.mark.parametrize("length, size", [(100, 32), (64, 32), (33, 32), (20, 32)])
def test_windows_cover_the_whole_length(length, size):
    starts = _starts(length, size, 0.25)
    covered = torch.zeros(length, dtype=torch.bool)
    for s in starts:
        covered[s : s + size] = True
    assert covered.all() and starts[0] == 0 and starts[-1] == max(0, length - size)


def test_context_free_model_gives_the_whole_image_result():
    # A 1x1 convolution sees no context, so every window agrees on every pixel.
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 4, 1)
    image = torch.randn(2, 3, 70, 100)
    windowed = sliding_window(model, image, 32)
    torch.testing.assert_close(windowed, model(image).log_softmax(1), atol=1e-5, rtol=1e-5)


def test_validation_scores_every_pixel_of_a_large_tile():
    torch.manual_seed(0)
    image = torch.randn(1, 3, 70, 100)
    mask = torch.randint(0, 3, (1, 70, 100))
    mask[0, :5] = 255
    seen = []
    model = torch.nn.Conv2d(3, 3, 1)
    task = SegmentationTask(model, 3, window=32)
    task.model.register_forward_hook(lambda m, i, o: seen.append(tuple(i[0].shape[-2:])))
    task.validation_step((image, mask))
    assert set(seen) == {(32, 32)}  # the model only ever sees training-size windows
    assert task.iou.confmat.sum() == (mask != 255).sum()  # yet the whole tile is scored
    whole = SegmentationTask(model, 3)
    whole.validation_step((image, mask))
    assert task.compute_metrics() == pytest.approx(whole.compute_metrics(), abs=1e-5)


def test_window_defaults_to_the_training_crop():
    assert task_kwargs({"augmentation": {"crop": 256}})["window"] == 256
    assert task_kwargs({})["window"] is None
    assert task_kwargs({"augmentation": {"crop": 256}, "task": {"window": None}})["window"] is None
    with pytest.raises(ValueError, match="window_overlap"):
        SegmentationTask(torch.nn.Conv2d(3, 2, 1), 2, window=32, window_overlap=1.0)


def test_detection_refuses_a_crop_when_it_evaluates():
    check_eval_crop({"augmentation": {"crop": None}})
    with pytest.raises(ValueError, match="windowed"):
        check_eval_crop({"augmentation": {"crop": 512}})
