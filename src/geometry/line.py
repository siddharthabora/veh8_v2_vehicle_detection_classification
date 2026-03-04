# src/geometry/line.py

from dataclasses import dataclass


@dataclass(frozen=True)
class HorizontalLine:
    """A horizontal line at y = constant in image coordinates (y increases downward)."""
    y: int

    @staticmethod
    def from_height(height: int, frac: float = 0.75) -> "HorizontalLine":
        if height <= 0:
            raise ValueError("height must be > 0")
        if not (0.0 < frac < 1.0):
            raise ValueError("frac must be between 0 and 1")
        return HorizontalLine(y=int(height * frac))
