"""Contrastive image-text fine-tuning (SigLIP 2) on pathology captions, for
zero-shot tile classification and image-text retrieval. LoRA needs the
``histossl`` extra (``pip install 'pathotrain[histossl]'``), whose probe code it
shares."""

from pathotrain.pathoclip.models import CLIPS, load_clip

__all__ = ["CLIPS", "load_clip"]
