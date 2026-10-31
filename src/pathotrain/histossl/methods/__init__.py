"""The histossl methods: each module's ``build(cfg, ssl, probe)`` makes its task."""

from pathotrain.histossl.methods import teacher_jepa
from pathotrain.histossl.methods.base import SSLTask

BUILDERS = {"teacher_jepa": teacher_jepa.build}

__all__ = ["BUILDERS", "SSLTask"]
