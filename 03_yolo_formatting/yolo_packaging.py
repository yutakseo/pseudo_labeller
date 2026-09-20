"""Create Roboflow-compatible YOLO Segmentation ZIP archives."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
DEFAULT_LABELMAP_FILENAMES = ("labelmap.txt", "classes.txt")


@dataclass(frozen=True)
class YoloPackagingConfig:
    """Locations and file types used when packaging a YOLO dataset."""

    image_root: Path
    label_root: Path
    output_zip: Path
    image_extensions: frozenset[str] = DEFAULT_IMAGE_EXTENSIONS
    labelmap_filenames: tuple[str, ...] = DEFAULT_LABELMAP_FILENAMES

    def validate(self) -> None:
        if not self.image_root.is_dir():
            raise NotADirectoryError(f"Image directory does not exist: {self.image_root}")
        if not self.label_root.is_dir():
            raise NotADirectoryError(f"Label directory does not exist: {self.label_root}")
        if self.output_zip.suffix.lower() != ".zip":
            raise ValueError("output_zip must have a .zip extension")
        self.find_labelmap()

    def find_labelmap(self) -> Path:
        """Return the preferred available class-ID-to-name mapping file."""
        for filename in self.labelmap_filenames:
            path = self.label_root / filename
            if path.is_file():
                return path
        expected = ", ".join(self.labelmap_filenames)
        raise FileNotFoundError(
            f"No label map found in {self.label_root}. Expected one of: {expected}"
        )

    def classNames(self) -> tuple[str, ...]:
        """Return non-empty class names ordered by their zero-based IDs."""
        labelmap_path = self.find_labelmap()
        class_names = tuple(
            line.strip()
            for line in labelmap_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not class_names:
            raise ValueError(f"Label map is empty: {labelmap_path}")
        return class_names


@dataclass(frozen=True)
class PackagingResult:
    """Summary of a completed ZIP packaging operation."""

    output_zip: Path
    image_count: int
    label_count: int


class RoboflowYoloPackager:
    """Package images and polygon labels as a standard YOLO dataset ZIP."""

    def __init__(self, config: YoloPackagingConfig) -> None:
        self.config = config

    def image_paths(self) -> list[Path]:
        return sorted(
            path for path in self.config.image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.config.image_extensions
        )

    def label_path_for(self, image_path: Path) -> Path:
        """Map an image path to its label path, preserving the image tree."""
        relative_path = image_path.relative_to(self.config.image_root)
        return (self.config.label_root / relative_path).with_suffix(".txt")

    def package(self) -> PackagingResult:
        """Write ``data.yaml`` and flattened ``train/images`` and ``train/labels``."""
        self.config.validate()
        image_paths = self.image_paths()
        if not image_paths:
            return PackagingResult(self.config.output_zip, image_count=0, label_count=0)

        self.validateNames(image_paths)
        missing_labels = [path for path in image_paths if not self.label_path_for(path).is_file()]
        if missing_labels:
            raise FileNotFoundError(self._missing_labels_message(missing_labels))

        self.config.output_zip.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(self.config.output_zip, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("data.yaml", self.dataYaml())
            archive.writestr("valid/images/", "")
            archive.writestr("valid/labels/", "")
            for image_path in image_paths:
                archive.write(image_path, Path("train/images") / image_path.name)
                label_name = image_path.with_suffix(".txt").name
                archive.write(
                    self.label_path_for(image_path),
                    Path("train/labels") / label_name,
                )
        return PackagingResult(self.config.output_zip, len(image_paths), len(image_paths))

    def dataYaml(self) -> str:
        """Create the class mapping required for YOLO format detection."""
        class_names = self.config.classNames()
        names = ", ".join(
            json.dumps(class_name, ensure_ascii=False)
            for class_name in class_names
        )
        return (
            "path: .\n"
            "train: train/images\n"
            "val: valid/images\n"
            f"nc: {len(class_names)}\n"
            f"names: [{names}]\n"
        )

    @staticmethod
    def validateNames(image_paths: list[Path]) -> None:
        """Reject names that would collide after flattening the source tree."""
        names = [path.name.casefold() for path in image_paths]
        label_names = [path.with_suffix(".txt").name.casefold() for path in image_paths]
        duplicate_images = RoboflowYoloPackager.duplicateNames(names)
        duplicate_labels = RoboflowYoloPackager.duplicateNames(label_names)
        if duplicate_images or duplicate_labels:
            duplicates = sorted(duplicate_images | duplicate_labels)
            preview = ", ".join(duplicates[:10])
            raise ValueError(f"Duplicate flattened filenames: {preview}")

    @staticmethod
    def duplicateNames(names: list[str]) -> set[str]:
        """Return values that occur more than once."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        return duplicates

    @staticmethod
    def _missing_labels_message(missing_labels: list[Path]) -> str:
        preview = "\n".join(f"  {path}" for path in missing_labels[:10])
        remaining = len(missing_labels) - 10
        suffix = f"\n  ... and {remaining} more" if remaining > 0 else ""
        return f"{len(missing_labels)} image(s) do not have matching .txt label files:\n{preview}{suffix}"
