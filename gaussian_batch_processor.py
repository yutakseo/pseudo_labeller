"""Create pseudo-segmentation labels and optionally export a dataset format.

Edit ``DATASET_NAME`` below, then run ``python gaussian_batch_processor.py``.
All paths are built from the project directory, so the script works from an
IDE as well as a terminal. The implementation is in ``02_pseudo_labeller``
and can be reused from notebooks.
"""

from importlib import import_module
from pathlib import Path

pseudo_labeller = import_module("02_pseudo_labeller")
formatters = import_module("03_formatting")
GaussianDifferenceConfig = pseudo_labeller.GaussianDifferenceConfig
PseudoLabelGenerator = pseudo_labeller.PseudoLabelGenerator
RoboflowYoloPackager = formatters.RoboflowYoloPackager
YoloPackagingConfig = formatters.YoloPackagingConfig
build_label_studio_dataset = formatters.build_dataset

# ---------------------------------------------------------------------------
# Run settings: change this value for a new dataset.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "__data"
DATASET_NAME = "20261010_case16"

# Input is the crop stage output; labelled data is the next pipeline stage.
CROPPED_ROOT = DATA_ROOT / f"{DATASET_NAME}_(2)cropped"
LABELLED_ROOT = DATA_ROOT / f"{DATASET_NAME}_(3)labelled"


CONFIG = GaussianDifferenceConfig(
    input_root=CROPPED_ROOT,
    mask_output_root=LABELLED_ROOT,
    label_output_root=LABELLED_ROOT,
    class_id=0,
    class_name="Exposure",
    min_contour_area=0.0,
    contour_approximation_ratio=0.0,
    gaussian_kernel_size=101,
    gaussian_sigma=10.0,
    mask_threshold=None,
    # Choose one: "none", "yolo", or "label_studio".
    formatting="none",
)

# Export destinations used only when their matching `formatting` mode is selected.
LABEL_STUDIO_OUTPUT_ROOT = DATA_ROOT / f"{DATASET_NAME}_label_studio"
YOLO_OUTPUT_ZIP = DATA_ROOT / f"{DATASET_NAME}_yolo_roboflow_upload.zip"


def export_format() -> None:
    """Export the generated labels in the format selected by ``CONFIG``."""
    if CONFIG.formatting == "none":
        print("Formatting disabled; generated masks and YOLO segmentation labels only.")
        return

    if CONFIG.formatting == "yolo":
        result = RoboflowYoloPackager(
            YoloPackagingConfig(
                image_root=CONFIG.input_root,
                label_root=CONFIG.label_output_root,
                output_zip=YOLO_OUTPUT_ZIP,
            )
        ).package()
        print(f"Created YOLO/Roboflow ZIP: {result.output_zip}")
        print(f"Packaged {result.image_count} image(s) and {result.label_count} label file(s)")
        return

    images, masks = build_label_studio_dataset(
        source_images=CONFIG.input_root,
        source_masks=CONFIG.mask_output_root,
        source_labelmap=CONFIG.label_output_root / "labelmap.txt",
        output_root=LABEL_STUDIO_OUTPUT_ROOT,
    )
    print(f"Created Label Studio dataset: {LABEL_STUDIO_OUTPUT_ROOT}")
    print(f"Label Studio tasks: {images}; editable brush masks: {masks}")


def main() -> int:
    """Generate labels and perform the requested optional dataset export."""
    generator = PseudoLabelGenerator(CONFIG)
    try:
        result = generator.run()
    except (OSError, ValueError) as error:
        print(f"Labelling failed: {error}")
        return 1

    if result.saved_masks == 0 and not result.failures:
        print(f"No supported image files found in: {CONFIG.input_root}")
        return 0

    print(f"Saved {result.saved_masks} mask(s) under: {CONFIG.mask_output_root}")
    print(f"Saved {result.saved_regions} YOLO segmentation region(s) under: {CONFIG.label_output_root}")
    print(f"Saved labelmap.txt for class {CONFIG.class_id}: {CONFIG.class_name!r}")
    if result.failures:
        print(f"Failed to process {len(result.failures)} file(s):")
        for path, error in result.failures:
            print(f"  {path}: {error}")
        return 1

    try:
        export_format()
    except (OSError, ValueError) as error:
        print(f"Formatting failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    
