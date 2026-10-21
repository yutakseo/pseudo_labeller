"""Run the complete pseudo-labelling pipeline from one source directory.

Edit the two paths in ``CONFIG`` below, then run this file from your IDE.
The output directory may contain the original ``_(0)raw`` folder. Generated
folders follow the existing ``_(1)``, ``_(2)``, ``_(3)``, ``_(4)`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
import sys


@dataclass(frozen=True)
class PipelineConfig:
    """Settings to edit before running the pipeline."""

    source_dataset: str
    output_root: str
    exposure: str = "10000us"
    export_format: str = "label-studio"  # label-studio, yolo, none
    class_name: str = "Exposure"


# ---------------------------------------------------------------------------
# Run settings: paste the two paths here, then run this file.
# ---------------------------------------------------------------------------
CONFIG = PipelineConfig(
    source_dataset=r"F:\[뿌리]비전검사\Pseudo-labeller-v2\pseudo_labeller\__data\20261010_case16\20261010_case16_(0)raw",  # Example: r"F:\data\source_dataset"
    output_root=r"F:\[뿌리]비전검사\Pseudo-labeller-v2\pseudo_labeller\__data\20261010_case16",
)


def ensure_safe_paths(source: Path, output: Path) -> tuple[Path, Path]:
    """Validate paths before any stage creates or overwrites files."""
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Source dataset does not exist: {source}")
    if source == output:
        raise ValueError("Output directory must be different from the source dataset")
    if source in output.parents:
        raise ValueError("Output directory must not be inside the source dataset")
    return source, output


def dataset_name(source_root: Path) -> str:
    """Derive ``20261010_case16`` from ``20261010_case16_(0)raw``."""
    name = source_root.name.split("_(0)", maxsplit=1)[0].rstrip("_")
    return name or source_root.name


def ensure_new_stage_paths(*paths: Path) -> None:
    """Prevent a rerun from silently mixing results with a previous run."""
    existing = [path for path in paths if path.exists()]
    if existing:
        names = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(f"Output stage already exists. Choose another output folder or remove it:\n  {names}")


def run_pipeline(config: PipelineConfig) -> None:
    """Run stages 00--03 using isolated directories below ``config.output_root``."""
    if not config.source_dataset.strip():
        raise ValueError("Set CONFIG.source_dataset at the top of main.py")
    if not config.output_root.strip():
        raise ValueError("Set CONFIG.output_root at the top of main.py")
    if config.export_format not in {"label-studio", "yolo", "none"}:
        raise ValueError("CONFIG.export_format must be: label-studio, yolo, or none")

    source_root, output_root = ensure_safe_paths(
        Path(config.source_dataset), Path(config.output_root)
    )
    name = dataset_name(source_root)
    extracted_root = output_root / f"{name}_(1)extracted"
    cropped_root = output_root / f"{name}_(2)cropped"
    labelled_root = output_root / f"{name}_(3)labelled"
    label_studio_root = output_root / f"{name}_(4)label_studio"
    yolo_zip = output_root / f"{name}_(4)yolo_roboflow_upload.zip"
    stage_paths = [extracted_root, cropped_root, labelled_root]
    if config.export_format == "label-studio":
        stage_paths.append(label_studio_root)
    elif config.export_format == "yolo":
        stage_paths.append(yolo_zip)
    ensure_new_stage_paths(*stage_paths)
    output_root.mkdir(parents=True, exist_ok=True)

    extract_module = import_module("00_extract.extract_mono")
    crop_module = import_module("01_crop.main")
    pseudo_module = import_module("02_pseudo_labeller")
    formatting_module = import_module("03_formatting")

    print(f"[00_extract] {source_root} -> {extracted_root}")
    extraction_config = extract_module.ExtractionConfig(
        source_root=source_root,
        output_root=extracted_root,
        exposure=config.exposure,
    )
    extraction_result = extract_module.MonoExtractor(extraction_config).extract()
    if extraction_result.copied_count == 0:
        raise RuntimeError(
            f"No mono images found in exposure directory: {config.exposure}"
        )
    print(f"  copied {extraction_result.copied_count} image(s)")

    print(f"[01_crop] {extracted_root} -> {cropped_root}")
    crop_config = replace(
        crop_module.CONFIG,
        source=extracted_root,
        output=cropped_root,
        reference_exposure=config.exposure,
    )
    crop_module.main(crop_config)

    # ROI crop images live here; using this exact root avoids processing reports
    # or other non-image artifacts that may be written beside ``crops``.
    crop_images_root = cropped_root / "crops" / crop_config.mode
    print(f"[02_pseudo_labeller] {crop_images_root} -> {labelled_root}")
    label_config = pseudo_module.GaussianDifferenceConfig(
        input_root=crop_images_root,
        mask_output_root=labelled_root,
        label_output_root=labelled_root,
        class_id=0,
        class_name=config.class_name,
        min_contour_area=0.0,
        contour_approximation_ratio=0.0,
        gaussian_kernel_size=101,
        gaussian_sigma=10.0,
    )
    label_result = pseudo_module.PseudoLabelGenerator(label_config).run()
    if label_result.failures:
        details = "\n".join(f"  {path}: {error}" for path, error in label_result.failures)
        raise RuntimeError(f"Pseudo-labelling failed for {len(label_result.failures)} file(s):\n{details}")
    if label_result.saved_masks == 0:
        raise RuntimeError(f"No crop images found under: {crop_images_root}")
    print(f"  saved {label_result.saved_masks} mask(s), {label_result.saved_regions} region(s)")

    if config.export_format == "none":
        return
    if config.export_format == "label-studio":
        print(f"[03_formatting] Label Studio export -> {label_studio_root}")
        image_count, mask_count = formatting_module.build_dataset(
            source_images=crop_images_root,
            source_masks=labelled_root,
            source_labelmap=labelled_root / "labelmap.txt",
            output_root=label_studio_root,
        )
        print(f"  exported {image_count} image(s), {mask_count} editable mask(s)")
        return

    print(f"[03_formatting] YOLO/Roboflow export -> {yolo_zip}")
    package_result = formatting_module.RoboflowYoloPackager(
        formatting_module.YoloPackagingConfig(
            image_root=crop_images_root,
            label_root=labelled_root,
            output_zip=yolo_zip,
        )
    ).package()
    print(
        f"  packaged {package_result.image_count} image(s), "
        f"{package_result.label_count} label file(s)"
    )


def main(config: PipelineConfig = CONFIG) -> int:
    """Run the pipeline and return a shell-friendly status code."""
    try:
        run_pipeline(config)
    except SystemExit as error:
        # The existing ROI command uses SystemExit for its validation errors.
        # Convert those into the same clean result as errors from other stages.
        if isinstance(error.code, int):
            return error.code
        print(f"Pipeline failed: {error.code}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
