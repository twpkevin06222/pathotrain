import pickle

import pytest
from PIL import Image

from pathotrain.pathovlm import VLMS, VQACollator, load_processor
from pathotrain.pathovlm.collate import IGNORE


def _processor(name):
    try:  # tokenizer, chat template and image settings only: a few MB, no weights
        return load_processor(name, local_files_only=True)
    except OSError:
        pytest.skip(f"{name}'s processor is not in the Hugging Face cache")


SAMPLES = [
    {
        "image": Image.new("RGB", (792, 528), "pink"),
        "question": "is this a carcinoma?",
        "answer": "yes",
        "type": "closed",
    },
    {
        "image": Image.new("RGB", (300, 200), "red"),
        "question": "what is present?",
        "answer": "necrosis of the liver",
        "type": "open",
    },
]
END = {"qwen3-vl-2b": "<|im_end|>\n", "smolvlm2-500m": "<end_of_utterance>\n"}


@pytest.mark.parametrize("name", ["qwen3-vl-2b", "smolvlm2-500m"])
def test_only_the_answer_and_its_end_of_turn_are_trained_on(name):
    processor = _processor(name)
    batch = VQACollator(processor, system="You are a pathologist.")(SAMPLES)

    labels, ids = batch["labels"], batch["input_ids"]
    assert labels.shape == ids.shape and "pixel_values" in batch
    for i, sample in enumerate(SAMPLES):
        trained = labels[i][labels[i] != IGNORE]
        assert (
            processor.tokenizer.decode(trained).strip() == f"{sample['answer']}{END[name]}".strip()
        )
        kept = labels[i] != IGNORE
        assert (labels[i][kept] == ids[i][kept]).all()
    assert (labels[batch["attention_mask"] == 0] == IGNORE).all()  # padding
    assert "pathologist" in processor.tokenizer.decode(ids[0])


@pytest.mark.parametrize("name", ["qwen3-vl-2b", "smolvlm2-500m"])
def test_generation_batches_are_left_padded_prompts_with_their_answers(name):
    processor = _processor(name)
    batch = VQACollator(processor, generation=True)(SAMPLES)

    assert "labels" not in batch and batch["answers"] == ["yes", "necrosis of the liver"]
    assert batch["types"] == ["closed", "open"]
    assert (batch["attention_mask"][:, -1] == 1).all()  # prompts end together
    assert "necrosis" not in processor.tokenizer.decode(batch["input_ids"][1])
    assert processor.tokenizer.padding_side == "right"  # the caller's processor is untouched


def test_image_settings_cap_qwen_tokens_and_keep_smolvlm_to_one_image():
    big = [{**SAMPLES[0], "image": Image.new("RGB", (2048, 2048))}]
    qwen = VQACollator(_processor("qwen3-vl-2b"))(big)
    assert (qwen["image_grid_thw"][0].prod() / 4) <= 1024  # 2 x 2 patches per token
    smol = VQACollator(_processor("smolvlm2-500m"))(big)
    assert smol["pixel_values"].shape[1] == 1  # no sub-images
    split = VQACollator(_processor_with("smolvlm2-500m", do_image_splitting=True))(big)
    assert split["pixel_values"].shape[1] > 1


def _processor_with(name, **image):
    _processor(name)  # skips when not cached
    return load_processor(name, local_files_only=True, **image)


def test_collator_pickles_for_dataloader_workers():
    collator = pickle.loads(pickle.dumps(VQACollator(_processor("smolvlm2-500m"))))
    assert collator(SAMPLES)["labels"].shape[0] == 2


def test_registry_rejects_unknown_models_and_pins_revisions():
    assert all(len(v.revision) == 40 for v in VLMS.values())
    with pytest.raises(ValueError, match="unknown VLM"):
        load_processor("qwen9-vl")
