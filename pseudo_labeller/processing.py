"""Gaussian-difference mask creation and YOLO polygon conversion."""

from __future__ import annotations

import cv2
import numpy as np

from .config import GaussianDifferenceConfig


class GaussianDifferenceMaskMaker:
    """Create filled binary regions from local Gaussian-difference responses."""

    def __init__(self, config: GaussianDifferenceConfig) -> None:
        self.config = config

    def create(self, image: np.ndarray) -> np.ndarray:
        """Return a binary mask whose retained external contours are filled."""
        blurred = cv2.GaussianBlur(image, (self.config.gaussian_kernel_size,) * 2, self.config.gaussian_sigma)
        difference = image.astype(np.int16) - blurred.astype(np.int16)
        scalar_difference = difference.mean(axis=2) if difference.ndim == 3 else difference
        difference_map = np.abs(scalar_difference).clip(0, 255).astype(np.uint8)
        threshold = self.config.mask_threshold
        threshold_type = cv2.THRESH_BINARY if threshold is not None else cv2.THRESH_BINARY + cv2.THRESH_OTSU
        _, mask = cv2.threshold(difference_map, threshold or 0, 255, threshold_type)
        return self.fill_regions(mask)

    def fill_regions(self, mask: np.ndarray) -> np.ndarray:
        """Fill retained contours, turning edge responses into solid regions."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        retained = [c for c in contours if cv2.contourArea(c) >= self.config.min_contour_area]
        filled_mask = np.zeros_like(mask)
        if retained:
            cv2.drawContours(filled_mask, retained, -1, 255, thickness=cv2.FILLED)
        return filled_mask


class YoloSegmentationExporter:
    """Convert binary mask regions to normalized YOLO segmentation records."""

    def __init__(self, config: GaussianDifferenceConfig) -> None:
        self.config = config

    def create_labels(self, mask: np.ndarray) -> list[str]:
        height, width = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        labels: list[str] = []
        for contour in contours:
            if cv2.contourArea(contour) < self.config.min_contour_area:
                continue
            epsilon = self.config.contour_approximation_ratio * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(polygon) < 3:
                continue
            coordinates: list[str] = []
            for x, y in polygon:
                coordinates.extend((f"{min(max(float(x) / width, 0.0), 1.0):.6f}", f"{min(max(float(y) / height, 0.0), 1.0):.6f}"))
            labels.append(f"{self.config.class_id} " + " ".join(coordinates))
        return labels
