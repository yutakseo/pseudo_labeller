"""Configure and run ROI cropping without writing a long command line.

Edit ``CONFIG`` below, then run ``python 01_crop/main.py`` from the project
root.  ``extra_args`` accepts any advanced option exposed by
``auto_roi_local_std.py --help`` that is not represented as a config field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

if __package__:  # Supports `python -m 01_crop.main`.
    from . import auto_roi_local_std
else:  # Supports `python 01_crop/main.py` and direct IDE execution.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import auto_roi_local_std


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "__data"
DATASET_ROOT = DATA_ROOT / "20261002_누락시편추가촬영"


@dataclass(frozen=True)
class RoiCropConfig:
    """All commonly adjusted ROI-cropping options."""

    source: Path
    output: Path
    mode: str = "auto"  # std, fft, gabor, fft+gabor, all, auto, gabor_boundary
    reference_exposure: str | None = "10000us"
    exposure: str = "10000us"  # Used only when reference_exposure is None.
    prefix: str | None = "mono_"
    extensions: str = ".png,.jpg,.jpeg,.bmp,.webp"
    image_output: str = "masked"  # masked, crop, rectified
    margin: int = 20

    # Auto-mode controls.
    auto_max_size: int = 1024
    fusion_exposures: int = 0
    auto_min_quality: float = 0.55
    auto_fine_band: float = 0.06
    registration_min_correlation: float = 0.65

    # Boundary and quality controls shared by the standard modes.
    min_height_ratio: float = 0.32
    max_height_ratio: float = 0.55
    max_tilt_degrees: float = 20.0
    strips: int = 24
    ransac_residual: float = 8.0
    ransac_max_residual: float = 16.0
    min_inlier_ratio: float = 0.55
    min_x_coverage: float = 0.75
    max_gap_ratio: float = 0.25
    max_median_residual: float = 8.0
    max_height_change_ratio: float = 0.15
    min_pair_inlier_ratio: float = 0.40

    # Optional review/override inputs.  Set to None when unused.
    roi_overrides: Path | None = None
    ground_truth: Path | None = None
    apply_ground_truth: bool = False
    limit: int | None = None
    dry_run: bool = False
    save_debug: bool = False  # Valid only with mode="gabor_boundary".
    boundary_config: Path | None = None

    # Pass any less-common CLI options here, e.g. ("--gabor-kernel", "51").
    extra_args: tuple[str, ...] = ()


CONFIG = RoiCropConfig(
    source=DATASET_ROOT / "20261002_(1)extracted",
    # Must be new or empty: ROI processing intentionally will not mix runs.
    output=DATASET_ROOT / "20261002_(2)cropped_new",
)


def _option(flag: str, value: object) -> list[str]:
    return [flag, str(value)]


def build_arguments(config: RoiCropConfig) -> list[str]:
    """Translate the editable config into the established ROI CLI options."""
    arguments = [
        *_option("--source", config.source),
        *_option("--output", config.output),
        *_option("--mode", config.mode),
        *_option("--extensions", config.extensions),
        *_option("--image-output", config.image_output),
        *_option("--margin", config.margin),
        *_option("--auto-max-size", config.auto_max_size),
        *_option("--fusion-exposures", config.fusion_exposures),
        *_option("--auto-min-quality", config.auto_min_quality),
        *_option("--auto-fine-band", config.auto_fine_band),
        *_option("--registration-min-correlation", config.registration_min_correlation),
        *_option("--min-height-ratio", config.min_height_ratio),
        *_option("--max-height-ratio", config.max_height_ratio),
        *_option("--max-tilt-degrees", config.max_tilt_degrees),
        *_option("--strips", config.strips),
        *_option("--ransac-residual", config.ransac_residual),
        *_option("--ransac-max-residual", config.ransac_max_residual),
        *_option("--min-inlier-ratio", config.min_inlier_ratio),
        *_option("--min-x-coverage", config.min_x_coverage),
        *_option("--max-gap-ratio", config.max_gap_ratio),
        *_option("--max-median-residual", config.max_median_residual),
        *_option("--max-height-change-ratio", config.max_height_change_ratio),
        *_option("--min-pair-inlier-ratio", config.min_pair_inlier_ratio),
    ]
    if config.prefix:
        arguments.extend(_option("--prefix", config.prefix))
    if config.reference_exposure is None:
        arguments.extend(_option("--exposure", config.exposure))
    else:
        arguments.extend(_option("--reference-exposure", config.reference_exposure))
    if config.roi_overrides is not None:
        arguments.extend(_option("--roi-overrides", config.roi_overrides))
    if config.ground_truth is not None:
        arguments.extend(_option("--ground-truth", config.ground_truth))
    if config.apply_ground_truth:
        arguments.append("--apply-ground-truth")
    if config.limit is not None:
        arguments.extend(_option("--limit", config.limit))
    if config.dry_run:
        arguments.append("--dry-run")
    if config.save_debug:
        arguments.append("--save-debug")
    if config.boundary_config is not None:
        arguments.extend(_option("--boundary-config", config.boundary_config))
    arguments.extend(config.extra_args)
    return arguments


def main(config: RoiCropConfig = CONFIG) -> None:
    """Run the existing ROI implementation with the configured arguments."""
    arguments = build_arguments(config)
    print("ROI crop configuration:")
    print(f"  source: {config.source}")
    print(f"  output: {config.output}")
    print(f"  mode: {config.mode}")
    auto_roi_local_std.main(arguments)


if __name__ == "__main__":
    main()
