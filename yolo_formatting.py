"""Package images and YOLO Segmentation labels for Roboflow upload.

Edit ``CONFIG`` below, then run ``python yolo_formatting.py``.  The reusable
packaging implementation lives in :mod:`formatting.yolo_packaging`.
"""

from pathlib import Path
from yolo_formatting import RoboflowYoloPackager, YoloPackagingConfig


CONFIG = YoloPackagingConfig(
    image_root=Path(r"F:\0918_data\0918_data_raw"),
    label_root=Path(r"F:\__pseudo_labeller\02_pseudo_labeller\processed_mask\0918_pseudo_yolo_labels"),
    output_zip=Path(r".0918_pseudo_yolo_roboflow_upload.zip"),
)


if __name__ == "__main__":
    result = RoboflowYoloPackager(CONFIG).package()
    if result.image_count == 0:
        print(f"No supported image files found in: {CONFIG.image_root}")
    else:
        print(f"Created Roboflow upload ZIP: {result.output_zip}")
        print(f"Packaged {result.image_count} image(s) and {result.label_count} label file(s).")
        print("Upload this ZIP to an Instance Segmentation project as YOLO Segmentation.")
