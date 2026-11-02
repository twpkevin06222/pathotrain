"""Batches of :class:`~pathotrain.dataloader.vqa.VQADataset` samples for a VLM.

Each sample becomes one chat: an optional system message, a user turn with the
image and the question, and (for training) the assistant's answer, rendered by
the model's own chat template. Only the answer's tokens (and the template's
end-of-turn tokens after it) are trained on: every other label is -100.
"""

import copy

import torch
from transformers import ProcessorMixin

IGNORE = -100


def chat(question: str, answer: str | None = None, system: str | None = None) -> list[dict]:
    """The messages of one VQA sample, with the answer as the assistant's turn."""
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []
    messages.append(
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    )
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


class VQACollator:
    """``samples -> batch`` for a processor's model.

    Training (``generation=False``): the processor's inputs, right-padded, plus
    ``labels``. Generation: the prompts only, left-padded so new tokens follow
    every prompt, plus the ``answers`` and ``types`` to score against.
    ``instruction`` follows every question, e.g. the answer format (the same for
    every question: a per-type one would give away the gold answer's type).

    Raises:
        ValueError: A chat template whose prompt is not a prefix of the full
            chat, or whose answer tokenises differently inside the chat than
            alone (the labels would be wrong).
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        system: str | None = None,
        generation: bool = False,
        instruction: str | None = None,
    ) -> None:
        if generation:
            processor = copy.deepcopy(processor)  # the caller's keeps its padding side
            processor.tokenizer.padding_side = "left"
        elif processor.tokenizer.padding_side != "right":
            raise ValueError("training batches need a right-padding tokenizer")
        self.processor, self.system, self.generation = processor, system, generation
        self.instruction = instruction

    def _question(self, sample: dict) -> str:
        q = sample["question"]
        return f"{q}\n{self.instruction}" if self.instruction else q

    def _render(self, messages: list[dict], generation_prompt: bool) -> str:
        # A non-empty processor_kwargs stops SmolVLM's override from putting its video
        # settings in **kwargs, which transformers warns about on every call; with
        # tokenize=False the processor is not called, so the kwargs are unused.
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation_prompt,
            processor_kwargs={"return_tensors": None},
        )

    def __call__(self, samples: list[dict]) -> dict:
        images = [[s["image"]] for s in samples]
        prompts = [self._render(chat(self._question(s), system=self.system), True) for s in samples]
        if self.generation:
            batch = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
            return {
                **batch,
                "answers": [s["answer"] for s in samples],
                "types": [s["type"] for s in samples],
            }
        texts = [
            self._render(chat(self._question(s), s["answer"], self.system), False) for s in samples
        ]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = torch.full_like(batch["input_ids"], IGNORE)
        tokenizer = self.processor.tokenizer
        for i, (text, prompt) in enumerate(zip(texts, prompts, strict=True)):
            if not text.startswith(prompt):
                raise ValueError(
                    f"the chat template's prompt is not a prefix of the chat: {text!r}"
                )
            suffix = tokenizer(text[len(prompt) :], add_special_tokens=False)["input_ids"]
            end = int(batch["attention_mask"][i].sum())  # right padding: real tokens first
            tail = batch["input_ids"][i, end - len(suffix) : end]
            if tail.tolist() != suffix:
                raise ValueError(f"the answer tokenises differently inside the chat: {text!r}")
            labels[i, end - len(suffix) : end] = tail
        return {**batch, "labels": labels}
