"""Scores of generated VQA answers against the gold ones.

- ``closed_acc``: closed (yes / no) questions answered right. The prediction's
  first "yes" or "no" word counts, so "Yes, the image shows ..." is a yes; a
  prediction with neither is wrong. ``closed_parsed`` is the share of closed
  predictions that had one.
- ``open_em``: open answers equal to the gold after :func:`normalize`.
- ``open_f1``: token F1 between prediction and gold (SQuAD style).
- ``open_recall``: share of gold tokens found in the prediction (LLaVA-Med
  style); it rewards long answers, so read it next to EM and F1, never alone.
"""

import re
import string
from collections import Counter

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCTUATION = str.maketrans(string.punctuation, " " * len(string.punctuation))


def normalize(text: str) -> str:
    """Lowercase, punctuation and articles removed, whitespace collapsed."""
    return " ".join(ARTICLES.sub(" ", text.lower().translate(PUNCTUATION)).split())


def yes_no(text: str) -> str | None:
    """The first "yes" or "no" word of ``text``, else None."""
    return next((w for w in normalize(text).split() if w in ("yes", "no")), None)


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(gold).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, gold: str) -> float:
    pred, ref = set(normalize(prediction).split()), normalize(gold).split()
    return sum(t in pred for t in ref) / len(ref) if ref else 0.0


def score(predictions: list[str], answers: list[str], types: list[str]) -> dict[str, float]:
    """The metrics above over a split (each key only when its type is present),
    with the question counts ``n_closed`` and ``n_open``."""
    closed = [(p, a) for p, a, t in zip(predictions, answers, types, strict=True) if t == "closed"]
    open_ = [(p, a) for p, a, t in zip(predictions, answers, types, strict=True) if t == "open"]
    out: dict[str, float] = {"n_closed": len(closed), "n_open": len(open_)}
    if closed:
        parsed = [yes_no(p) for p, _ in closed]
        right = sum(y == normalize(a) for y, (_, a) in zip(parsed, closed, strict=True))
        out["closed_acc"] = right / len(closed)
        out["closed_parsed"] = sum(y is not None for y in parsed) / len(closed)
    if open_:
        out["open_em"] = sum(normalize(p) == normalize(a) for p, a in open_) / len(open_)
        out["open_f1"] = sum(token_f1(p, a) for p, a in open_) / len(open_)
        out["open_recall"] = sum(token_recall(p, a) for p, a in open_) / len(open_)
    return out
