"""Export a LoRA checkpoint written by :mod:`pathotrain.pathovlm.train` as a plain
Hugging Face model::

    python -m pathotrain.pathovlm.export runs/vqa01/checkpoint-7371 exported/vqa01

Writes the base model with the adapter merged into its weights (bf16,
``model*.safetensors``, ``config.json``), the processor with the image settings
it was trained with, and ``pathotrain.json``: the checkpoint's provenance (base
model and revision, image settings, LoRA settings, prompt, git commit). Load it
anywhere with transformers alone, no peft, and ask as in training (the
instruction from ``pathotrain.json`` after the question)::

    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained("exported/vqa01")
    processor = AutoProcessor.from_pretrained("exported/vqa01")

The ``checkpoint-N/`` folder itself is the adapter alone (``adapter_config.json``,
``adapter_model.safetensors``) for serving many adapters over one base.
"""

import argparse
import json
from pathlib import Path

from peft import PeftModel
from torch import nn
from transformers import ProcessorMixin

from pathotrain.pathovlm.models import load_vlm


def load_finetuned(checkpoint: str | Path) -> tuple[nn.Module, ProcessorMixin, dict]:
    """The pinned base model with ``checkpoint``'s adapter merged in (bf16, eval
    mode), the processor with its image settings, and its ``provenance.json``."""
    provenance = json.loads((Path(checkpoint) / "provenance.json").read_text())
    model, processor = load_vlm(provenance["model"], **provenance["image"])
    model = PeftModel.from_pretrained(model, str(checkpoint)).merge_and_unload()
    return model.eval(), processor, provenance


def export(checkpoint: str | Path, out_dir: str | Path) -> Path:
    """Write ``checkpoint`` merged into its base to the folder ``out_dir``; return it."""
    model, processor, provenance = load_finetuned(checkpoint)
    out = Path(out_dir)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    record = {"provenance": provenance, "checkpoint": str(Path(checkpoint).resolve())}
    (out / "pathotrain.json").write_text(json.dumps(record, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="checkpoint-N folder written by pathovlm.train")
    parser.add_argument("output", help="the folder to write the model to")
    args = parser.parse_args(argv)
    print(export(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
