"""Create pseudo-segmentation masks and YOLO/Roboflow labels.

Edit the settings below, then run ``python gaussian_batch_processor.py``.  The
implementation is in ``02_pseudo_labeller`` and can be reused from notebooks.
"""

from importlib import import_module
from pathlib import Path


pseudo_labeller = import_module("02_pseudo_labeller")
GaussianDifferenceConfig = pseudo_labeller.GaussianDifferenceConfig
PseudoLabelGenerator = pseudo_labeller.PseudoLabelGenerator


CONFIG = GaussianDifferenceConfig(
    input_root=Path(r"__data/20260918_extracted"),
    mask_output_root=Path(r"__data/20260918_labelled/masks"),
    label_output_root=Path(r"__data/20260918_labelled/outputs"),
    class_id=0,
    class_name="Exposure",
    min_contour_area=0.0,
    contour_approximation_ratio=0.0,
    gaussian_kernel_size=101,
    gaussian_sigma=10.0,
    mask_threshold=None,
)




if __name__ == "__main__":
    generator = PseudoLabelGenerator(CONFIG)
    result = generator.run()
    if result.saved_masks == 0 and not result.failures:
        print(f"No supported image files found in: {CONFIG.input_root}")
    else:
        print(f"Saved {result.saved_masks} mask(s) under: {CONFIG.mask_output_root}")
        print(f"Saved {result.saved_regions} YOLO segmentation region(s) under: {CONFIG.label_output_root}")
        print(f"Saved labelmap.txt for class {CONFIG.class_id}: {CONFIG.class_name!r}")
        if result.failures:
            print(f"Failed to process {len(result.failures)} file(s):")
            for path, error in result.failures:
                print(f"  {path}: {error}")
    
