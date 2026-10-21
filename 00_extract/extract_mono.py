"""Extract mono images at one exposure while preserving the source tree.

Usually only edit ``DATASET_NAME`` and ``EXPOSURE`` below, then run this file.
All paths are built from the project directory, so the current working directory
does not matter.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Optional, Sequence


# ---------------------------------------------------------------------------
# Run settings: change these two values for a new extraction run.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "__data"
DATASET_NAME = "20261010_case16"
EXPOSURE = "10000us"

# Input:  __data/<DATASET_NAME>
# Output: __data/<DATASET_NAME>_(1)extracted
DEFAULT_SOURCE_ROOT = DATA_ROOT / DATASET_NAME
DEFAULT_OUTPUT_ROOT = DATA_ROOT / f"{DATASET_NAME}_(1)extracted"

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
    exposure: str = EXPOSURE
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


def parseArgs(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy mono images from one exposure directory while preserving "
            "the source dataset tree."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Source dataset root (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--exposure",
        default=EXPOSURE,
        help=f"Exact exposure directory name (default: {EXPOSURE})",
    )
    return parser.parse_args(arguments)


def createConfig(arguments: argparse.Namespace) -> ExtractionConfig:
    """Create a validated extraction configuration from CLI arguments."""
    return ExtractionConfig(
        source_root=arguments.source,
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
