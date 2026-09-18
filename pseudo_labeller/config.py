"""Configuration models for pseudo-label generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})


@dataclass(frozen=True)
class GaussianDifferenceConfig:
    """All settings needed to generate masks and YOLO segmentation labels."""

    input_root: Path
    mask_output_root: Path
    label_output_root: Path
    class_id: int = 0
    class_name: str = "Exposure"
    min_contour_area: float = 10.0
    contour_approximation_ratio: float = 0.002
    gaussian_kernel_size: int = 21
    gaussian_sigma: float = 0.0
    mask_threshold: int | None = None
    image_extensions: frozenset[str] = DEFAULT_IMAGE_EXTENSIONS

    def validate(self) -> None:
        """Raise a helpful exception when a configuration value is invalid."""
        if not self.input_root.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {self.input_root}")
        if self.gaussian_kernel_size <= 0 or self.gaussian_kernel_size % 2 == 0:
            raise ValueError("gaussian_kernel_size must be a positive odd integer")
        if self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be zero or positive")
        if self.mask_threshold is not None and not 0 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be None or an integer between 0 and 255")
        if self.class_id < 0:
            raise ValueError("class_id must be zero or positive")
        if not self.class_name.strip():
            raise ValueError("class_name must not be empty")
        if self.min_contour_area < 0:
            raise ValueError("min_contour_area must be zero or positive")
        if self.contour_approximation_ratio < 0:
            raise ValueError("contour_approximation_ratio must be zero or positive")
