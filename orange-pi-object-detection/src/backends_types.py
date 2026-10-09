"""Plain data types shared by every inference backend."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
