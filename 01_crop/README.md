# ROI package

All ROI extraction implementation lives in this package.

- `auto_roi_local_std.py`: CLI, shared image geometry, masking, CSV, and batch helpers.
- `roi_boundary.py`: Gabor/STD boundary candidates, path selection, RANSAC fitting, and recovery.
- `roi_boundary_quality.py`: configuration loading and independent quality gates.
- `auto_roi_group.py`: multi-image and multi-exposure ROI grouping.
- `roi_annotations.py`: manual line and quadrilateral annotation parsing.
- `evaluate_roi_boundaries.py`: boundary metrics and held-out evaluation.
- `prepare_roi_gt.py`: create tune/validation annotation manifests.

Run the extractor from the repository root:

```powershell
uv run python -m src.roi.auto_roi_local_std --help
```

Use module entry points from the repository root.
