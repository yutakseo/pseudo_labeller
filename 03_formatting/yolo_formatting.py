"""Package images and YOLO Segmentation labels for Roboflow upload.

Edit ``CONFIG`` below, then run ``python -m 03_formatting.yolo_formatting``.
"""

from importlib import import_module
import logging
from pathlib import Path
import sys


LOG_FORMAT = "%(levelname)s: %(message)s"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "__data"

yolo_package = import_module("03_formatting")
RoboflowYoloPackager = yolo_package.RoboflowYoloPackager
YoloPackagingConfig = yolo_package.YoloPackagingConfig


CONFIG = YoloPackagingConfig(
    image_root=DATA_ROOT / "20260918_(2)cropped" / "crops" / "auto",
    label_root=DATA_ROOT / "20260922_(3)labelled" / "outputs",
    output_zip=DATA_ROOT / "20260922_yolo_roboflow_upload.zip",
)


def main() -> int:
    """Create the upload archive and return a process exit code."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    try:
        result = RoboflowYoloPackager(CONFIG).package()
    except (OSError, ValueError) as error:
        logging.error("Packaging failed: %s", error)
        return 1

    if result.image_count == 0:
        logging.warning("No supported image files found in: %s", CONFIG.image_root)
        return 0

    logging.info("Created Roboflow upload ZIP: %s", result.output_zip)
    logging.info(
        "Packaged %d image(s) and %d label file(s)",
        result.image_count,
        result.label_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
