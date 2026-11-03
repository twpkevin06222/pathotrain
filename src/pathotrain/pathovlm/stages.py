"""The post-training stages :mod:`pathotrain.pathovlm.train` can run, each a TRL
trainer and its config class.

A config's ``stage`` section picks one: ``{name: sft, preset: null, args: {},
parent: null}`` (the default). ``preset`` names a set of TRL config values the
stage defines (e.g. MPO for DPO); ``args`` go straight into the stage's TRL config
class after the preset, for anything the pathotrain sections do not cover.
``parent`` is a ``checkpoint-N`` folder to start from: its adapter is merged into
the base and a new LoRA adapter trained on top, so the merged parent is the
reference model of ``dpo`` and ``kto``.

- ``sft``: VQA rows, the answer trained on.
- ``dpo`` (presets ``mpo``): rows written by :mod:`pathotrain.pathovlm.sample`;
  one pair per question and distinct wrong sample, the gold answer chosen and the
  sample rejected. Questions the parent always gets right give no pair.
- ``kto``: the same rows; every distinct sample, labelled desirable when right
  (:func:`~pathotrain.pathovlm.metrics.correct`). Right samples usually far
  outnumber wrong ones: see ``desirable_weight`` / ``undesirable_weight`` in
  ``stage.args``.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from trl import DPOConfig, DPOTrainer, KTOConfig, KTOTrainer, SFTConfig, SFTTrainer

from pathotrain.pathovlm.collate import VQACollator
from pathotrain.pathovlm.metrics import correct, normalize


@dataclass(frozen=True)
class StageSpec:
    config_cls: type
    trainer_cls: type
    # TRL config values the stage needs to work with pathotrain's data; not settable.
    fixed: dict = field(default_factory=dict)
    # (parsed config, processor) -> extra trainer kwargs, e.g. the data collator.
    trainer_kwargs: Callable[..., dict] = lambda c, processor: {}
    presets: dict[str, dict] = field(default_factory=dict)
    # (VQA records, prompt section) -> the stage's TRL dataset records; None: the records.
    build_dataset: Callable[..., list[dict]] | None = None


def _prompt(row: dict, prompt) -> list[dict]:
    """TRL's conversational prompt; TRL puts the image before the question, as
    :func:`~pathotrain.pathovlm.collate.chat` does."""
    question = f"{row['question']}\n{prompt.instruction}" if prompt.instruction else row["question"]
    system = [{"role": "system", "content": prompt.system}] if prompt.system else []
    return [*system, {"role": "user", "content": question}]


def _samples(row: dict) -> dict[str, str]:
    """``row``'s sampled answers, one per normalised answer."""
    if "samples" not in row:
        raise ValueError(f"{row['image']}: no samples; write the rows with pathovlm.sample")
    return {normalize(s): s for s in row["samples"]}


def _nonempty(records: list[dict], stage: str) -> list[dict]:
    if not records:
        raise ValueError(f"no {stage} rows: every sample is right (or no samples)")
    return records


def preference_pairs(rows: list[dict], prompt) -> list[dict]:
    pairs = [
        {
            "image": r["image"],
            "prompt": _prompt(r, prompt),
            "chosen": [{"role": "assistant", "content": r["answer"]}],
            "rejected": [{"role": "assistant", "content": s}],
        }
        for r in rows
        for s in _samples(r).values()
        if not correct(s, r["answer"], r["type"])
    ]
    return _nonempty(pairs, "dpo")


def unpaired_preferences(rows: list[dict], prompt) -> list[dict]:
    records = [
        {
            "image": r["image"],
            "prompt": _prompt(r, prompt),
            "completion": [{"role": "assistant", "content": s}],
            "label": correct(s, r["answer"], r["type"]),
        }
        for r in rows
        for s in _samples(r).values()
    ]
    # KTO reads in order (its KL term pairs neighbours), so a question's samples must not be adjacent.
    random.Random(0).shuffle(records)
    return _nonempty(records, "kto")


# Truncation would cut image tokens (up to 1024 for Qwen3-VL) or the answer.
NO_TRUNCATION = {"max_length": None}


STAGES = {
    "sft": StageSpec(
        SFTConfig,
        SFTTrainer,
        # The collator needs the raw rows (image, question, answer) and labels them itself.
        fixed={
            "remove_unused_columns": False,
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "max_length": None,
        },
        trainer_kwargs=lambda c, processor: {
            "data_collator": VQACollator(
                processor, c.prompt.system, instruction=c.prompt.instruction
            )
        },
    ),
    "dpo": StageSpec(
        DPOConfig,
        DPOTrainer,
        fixed=NO_TRUNCATION,
        # MPO (Wang et al. 2024): preference (sigmoid), quality (BCO) and generation (SFT) losses.
        presets={
            "mpo": {"loss_type": ["sigmoid", "bco_pair", "sft"], "loss_weights": [0.8, 0.2, 1.0]}
        },
        build_dataset=preference_pairs,
    ),
    "kto": StageSpec(
        KTOConfig, KTOTrainer, fixed=NO_TRUNCATION, build_dataset=unpaired_preferences
    ),
}
