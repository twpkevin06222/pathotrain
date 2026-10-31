"""Whole-slide image registration: align a non-H&E slide (e.g. IHC) to its H&E slide,
e.g. to tile both over the same tissue for virtual staining."""

from pathotrain.registration.affine import Registration, overlay, register

__all__ = ["Registration", "overlay", "register"]
