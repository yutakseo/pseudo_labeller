"""Batch orchestration for pseudo-mask and YOLO-label generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .config import GaussianDifferenceConfig
from .image_io import ImageRepository
from .processing import GaussianDifferenceMaskMaker, YoloSegmentationExporter


@dataclass
class BatchResult:
    saved_masks: int = 0
    saved_regions: int = 0
    failures: list[tuple[Path, str]] = field(default_factory=list)


class PseudoLabelGenerator:
    """Generate pseudo masks and matching YOLO segmentation files for a tree."""

    def __init__(self, config: GaussianDifferenceConfig, image_repository: ImageRepository | None = None) -> None:
        self.config = config
        self.images = image_repository or ImageRepository()
        self.mask_maker = GaussianDifferenceMaskMaker(config)
        self.label_exporter = YoloSegmentationExporter(config)

    def input_paths(self) -> list[Path]:
        return sorted(path for path in self.config.input_root.rglob("*") if path.is_file() and path.suffix.lower() in self.config.image_extensions)

    def write_labelmap(self) -> None:
        self.config.label_output_root.mkdir(parents=True, exist_ok=True)
        contents = self.config.class_name.strip() + "\n"
        for filename in ("labelmap.txt", "classes.txt"):
            path = self.config.label_output_root / filename
            try:
                path.write_text(contents, encoding="utf-8")
            except OSError as error:
                raise OSError(f"Cannot write label map: {path}") from error

    def run(self) -> BatchResult:
        self.config.validate()
        self.write_labelmap()
        result = BatchResult()
        for input_path in self.input_paths():
            try:
                self._process_one(input_path, result)
            except (OSError, cv2.error) as error:
                result.failures.append((input_path, str(error)))
        return result

    def _process_one(self, input_path: Path, result: BatchResult) -> None:
        image = self.images.read(input_path)
        if image is None:
            raise OSError(f"Cannot read image: {input_path}")
        relative_path = input_path.relative_to(self.config.input_root)
        mask_path = self.config.mask_output_root / relative_path
        label_path = (self.config.label_output_root / relative_path).with_suffix(".txt")
        mask = self.mask_maker.create(image)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.images.write(mask_path, mask):
            raise OSError(f"Cannot write mask: {mask_path}")
        labels = self.label_exporter.create_labels(mask)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        except OSError as error:
            raise OSError(f"Cannot write label: {label_path}") from error
        result.saved_masks += 1
        result.saved_regions += len(labels)
