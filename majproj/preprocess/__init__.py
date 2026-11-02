"""Preprocessing v2: one exported `_shapenet.npy` file -> canonical, distance-consistent partial cloud.

The same function is used for training (on simulated exports), inference, and (ported) in the browser.
"""
from .core import (CANON_POINTS, from_canonical, locate_sensor, preprocess, segment_object, to_canonical)

__all__ = ['preprocess', 'to_canonical', 'from_canonical', 'locate_sensor', 'segment_object', 'CANON_POINTS']
