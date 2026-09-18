"""OpenCV image I/O that supports non-ASCII Windows paths."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageRepository:
    """Read and write images without relying on OpenCV's path encoding support."""

    @staticmethod
    def read(path: Path) -> np.ndarray | None:
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
        except OSError:
            return None
        return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def write(path: Path, image: np.ndarray) -> bool:
        success, encoded = cv2.imencode(path.suffix, image)
        if not success:
            return False
        try:
            encoded.tofile(path)
        except OSError:
            return False
        return True
