"""The post-training stages :mod:`pathotrain.pathovlm.train` can run, each a TRL
trainer and its config class.

A config's ``stage`` section picks one: ``{name: sft, preset: null, args: {}}``
(the default). ``preset`` names a set of TRL config values the stage defines
(e.g. MPO for DPO); ``args`` go straight into the stage's TRL config class
after the preset, for anything the pathotrain sections do not cover.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from trl import SFTConfig, SFTTrainer

from pathotrain.pathovlm.collate import VQACollator


@dataclass(frozen=True)
class StageSpec:
    config_cls: type
    trainer_cls: type
    # TRL config values the stage needs to work with pathotrain's data; not settable.
    fixed: dict = field(default_factory=dict)
    # (parsed config, processor) -> extra trainer kwargs, e.g. the data collator.
    trainer_kwargs: Callable[..., dict] = lambda c, processor: {}
    presets: dict[str, dict] = field(default_factory=dict)


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
}
