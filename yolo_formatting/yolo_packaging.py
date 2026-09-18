"""Create Roboflow-compatible YOLO Segmentation ZIP archives."""

from __future__ import annotations

from dataclasses import dataclass
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
        raise FileNotFoundError(f"No label map found in {self.label_root}. Expected one of: {expected}")


@dataclass(frozen=True)
class PackagingResult:
    """Summary of a completed ZIP packaging operation."""

    output_zip: Path
    image_count: int
    label_count: int


class RoboflowYoloPackager:
    """Package images, matching labels, and a label map into a ZIP archive."""

    def __init__(self, config: YoloPackagingConfig) -> None:
        self.config = config

    def image_paths(self) -> list[Path]:
        return sorted(
            path for path in self.config.image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.config.image_extensions
        )

    def label_path_for(self, image_path: Path) -> Path:
        """Map an image path to its label path, preserving the image tree."""
        return (self.config.label_root / image_path.relative_to(self.config.image_root)).with_suffix(".txt")

    def package(self) -> PackagingResult:
        """Validate and write an archive with ``images/``, ``labels/``, and labelmap."""
        self.config.validate()
        image_paths = self.image_paths()
        if not image_paths:
            return PackagingResult(self.config.output_zip, image_count=0, label_count=0)

        missing_labels = [path for path in image_paths if not self.label_path_for(path).is_file()]
        if missing_labels:
            raise FileNotFoundError(self._missing_labels_message(missing_labels))

        labelmap_path = self.config.find_labelmap()
        self.config.output_zip.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(self.config.output_zip, "w", compression=ZIP_DEFLATED) as archive:
            for image_path in image_paths:
                relative_path = image_path.relative_to(self.config.image_root)
                archive.write(image_path, Path("images") / relative_path)
                archive.write(self.label_path_for(image_path), Path("labels") / relative_path.with_suffix(".txt"))
            archive.write(labelmap_path, "labelmap.txt")
        return PackagingResult(self.config.output_zip, len(image_paths), len(image_paths))

    @staticmethod
    def _missing_labels_message(missing_labels: list[Path]) -> str:
        preview = "\n".join(f"  {path}" for path in missing_labels[:10])
        remaining = len(missing_labels) - 10
        suffix = f"\n  ... and {remaining} more" if remaining > 0 else ""
        return f"{len(missing_labels)} image(s) do not have matching .txt label files:\n{preview}{suffix}"
