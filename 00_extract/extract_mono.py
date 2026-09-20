"""Extract mono images captured at one exposure while preserving the source tree."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Optional, Sequence


DEFAULT_DATASET_NAME = "20260918_데이터취득"
DEFAULT_OUTPUT_NAME = "20260918_extracted"
DEFAULT_EXPOSURE = "10000us"
DEFAULT_MONO_PREFIX = "mono_"
DEFAULT_IMAGE_EXTENSIONS: FrozenSet[str] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
)
LOG_FORMAT = "%(levelname)s: %(message)s"


@dataclass(frozen=True)
class ExtractionConfig:
    """Paths and selection rules for one extraction run."""

    source_root: Path
    output_root: Path
    exposure: str = DEFAULT_EXPOSURE
    mono_prefix: str = DEFAULT_MONO_PREFIX
    image_extensions: FrozenSet[str] = DEFAULT_IMAGE_EXTENSIONS

    def validate(self) -> None:
        """Raise an exception when the extraction configuration is unsafe."""
        if not self.source_root.is_dir():
            raise NotADirectoryError(
                f"Source dataset does not exist: {self.source_root}"
            )
        if not self.exposure.strip():
            raise ValueError("Exposure directory name must not be empty")
        if not self.mono_prefix.strip():
            raise ValueError("Mono filename prefix must not be empty")

        source_path = self.source_root.resolve()
        output_path = self.output_root.resolve()
        if source_path == output_path:
            raise ValueError("Source and output directories must be different")
        if source_path in output_path.parents:
            raise ValueError("Output directory must not be inside the source dataset")


@dataclass(frozen=True)
class ExtractionResult:
    """Summary of copied files."""

    copied_count: int
    output_root: Path


class MonoExtractor:
    """Copy matching mono images without changing relative paths or filenames."""

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config

    def selectPaths(self) -> list[Path]:
        """Return sorted images matching both exposure and mono constraints."""
        return sorted(
            path
            for path in self.config.source_root.rglob("*")
            if self.isTarget(path)
        )

    def isTarget(self, path: Path) -> bool:
        """Return whether a path is a mono image at the configured exposure."""
        return (
            path.is_file()
            and path.parent.name.casefold() == self.config.exposure.casefold()
            and path.name.casefold().startswith(self.config.mono_prefix.casefold())
            and path.suffix.casefold() in self.config.image_extensions
        )

    def extract(self) -> ExtractionResult:
        """Copy selected images and preserve their paths relative to the source."""
        self.config.validate()
        selected_paths = self.selectPaths()
        for source_path in selected_paths:
            self.copyOne(source_path)
        return ExtractionResult(len(selected_paths), self.config.output_root)

    def copyOne(self, source_path: Path) -> None:
        """Copy one source image with its metadata and relative directory tree."""
        relative_path = source_path.relative_to(self.config.source_root)
        output_path = self.config.output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)


def normalizeName(name: str) -> str:
    """Normalize a name so macOS decomposed Hangul compares reliably."""
    return unicodedata.normalize("NFC", name)


def findDataset(data_root: Path, dataset_name: str) -> Path:
    """Find a dataset by Unicode-normalized directory name."""
    expected_name = normalizeName(dataset_name)
    matches = sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir() and normalizeName(path.name) == expected_name
    )
    if not matches:
        raise FileNotFoundError(f"Dataset not found under {data_root}: {dataset_name}")
    if len(matches) > 1:
        paths = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"Multiple matching datasets found: {paths}")
    return matches[0]


def parseArgs(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "__data"
    parser = argparse.ArgumentParser(
        description=(
            "Copy mono images from one exposure directory while preserving "
            "the source dataset tree."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Source dataset root (default: __data/20260918_데이터취득)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_root / DEFAULT_OUTPUT_NAME,
        help="Output root (default: __data/20260918_extracted)",
    )
    parser.add_argument(
        "--exposure",
        default=DEFAULT_EXPOSURE,
        help="Exact exposure directory name (default: 10000us)",
    )
    return parser.parse_args(arguments)


def createConfig(arguments: argparse.Namespace) -> ExtractionConfig:
    """Create a validated extraction configuration from CLI arguments."""
    project_root = Path(__file__).resolve().parents[1]
    source_root = arguments.source
    if source_root is None:
        source_root = findDataset(project_root / "__data", DEFAULT_DATASET_NAME)
    return ExtractionConfig(
        source_root=source_root,
        output_root=arguments.output,
        exposure=arguments.exposure,
    )


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Run extraction and return a process exit code."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    try:
        config = createConfig(parseArgs(arguments))
        result = MonoExtractor(config).extract()
    except (OSError, RuntimeError, ValueError) as error:
        logging.error("Extraction failed: %s", error)
        return 1

    logging.info("Source: %s", config.source_root)
    logging.info("Output: %s", result.output_root)
    logging.info("Copied %d mono image(s)", result.copied_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
