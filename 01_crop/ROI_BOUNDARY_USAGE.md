# Boundary Quality Mode (v5)

## Scope

This version adds `gabor_boundary`. It does not replace `auto`, `all`, `std`,
`fft`, `gabor` or `fft+gabor`. All detection decisions in the new mode are in
native source coordinates. Exports must be `masked`, never warped or cropped.
The existing source files and previous output directories are not changed.

For Gaussian-difference candidate labels after ROI extraction, see
[ROI_PSEUDO_LABELS_USAGE.md](ROI_PSEUDO_LABELS_USAGE.md).

## PowerShell

From `D:\Code\vision_data`:

```powershell
$out = ".\data\roied\gabor_boundary_$(Get-Date -Format yyyyMMdd_HHmmss)"
uv run python -m src.roi.auto_roi_local_std `
  --source .\data\0918 `
  --output $out `
  --reference-exposure 10000us `
  --mode gabor_boundary `
  --boundary-config .\config\roi_gabor_boundary.yaml `
  --image-output masked `
  --save-debug
```

Use a new output folder every time. Omitting `--save-debug` writes only the
masked PNGs and CSV. `--fusion-exposures 0` (default) checks all exposures;
`--fusion-exposures 3` limits detection assistance to three exposures but still
exports every target. Gabor features are calculated at native resolution, so
this mode can take longer than the downscaled v4 auto pipeline.

Output layout:

- `roi_results.csv`: one row per target, including any skipped target.
- `crops/gabor_boundary/<original relative folder>/<filename>_roi.png`.
- `debug/gabor_boundary/<reference relative folder>/<filename>_debug.png`.

Only reference images receive debug panels. They contain the original, Gabor
envelope, brightness edges, raw candidates, pair-DP path, fit inliers/outliers,
final lines, mask overlay, competing candidate, and outside validation bands.

## Algorithm Changes

1. Optional centered Gaussian smoothing, illumination subtraction and CLAHE
   affect analysis only. The output uses the untouched input pixels.
2. Eight Gabor orientations form an absolute-response maximum and local envelope.
   Signed envelope transitions, brightness edges, local inside/outside contrast
   and adjacent-bin consistency generate up to five candidates per side/bin.
   STD is auxiliary. The whole fabric interior is not thresholded or required
   to have uniform texture.
3. Top/bottom pairs are selected together by global second-order DP. Position,
   thickness and slope-change penalties have separate YAML controls. Independent
   lines allow modest trapezoids; parallelism is not forced.
4. RANSAC tries 8/12/16 native pixels while retaining the same quality limits.
   A supported side is preserved exactly when only its counterpart is searched
   again. Recovery ranks peaks only within its allowed corridor, so stronger
   unrelated peaks outside it cannot consume the candidate budget. If a failed
   line crosses the supported line, only the failed side is discarded.
   If a side remains unknown, its canvas edge is left untrimmed and the
   result requires review. With no supported boundaries the original frame is
   preserved; this is not reported as successful ROI detection.
5. Alternative DP paths compete with the primary path. Geometry duplicates are
   ignored by ambiguity checks. Every near-score, spatially different competitor
   is checked, not just a duplicate second-place entry.
6. A scalar score cannot override ambiguity, per-side coverage, missing-third
   support, unsupported gaps, residuals, geometry, consensus or outside-texture
   gates. Outside validation uses local bands, not uniform-interior assumptions.
7. Multiple exposures supply corroborating endpoints, not ground truth. Final
   geometry is selected from reference-image candidates and reused unchanged for
   that scene/camera's targets. No cross-scene registration is used by this mode.
   An exposure contributes a vote only after its own fitting, geometry, ambiguity,
   score and outside checks pass without a consensus bonus. The denominator is
   all readable, same-size exposures selected for assistance, including those
   unable to provide a validated vote. Single-exposure runs receive no bonus.

## Status and Pixel Semantics

Automatic `status` / `quality_status` values are `PASS`, `RECOVERED` and
`NEEDS_REVIEW`. Recovery must pass every gate. All three export a PNG; review
does not mean acceptance. `auto_accepted=0` and `needs_review=1` identify review
outputs. An explicit override retains the legacy `status=manual` and uses
`quality_status=MANUAL`; it is never counted as automatic acceptance.

`roi_detected=0` distinguishes missing-boundary/full-frame preservation from
localized ROI. Unreadable/missing/ambiguous references and incompatible target
sizes remain explicit `skipped` rows, not fabricated images. The 696-output
expectation assumes the same readable, compatible source population as before.

For each column, the new mode retains:

```text
round(top_y) - margin_top through round(bottom_y) + margin_bottom, inclusive
```

NumPy nearest-even rounding is used. Bounds are clipped to the image; internally
the lower array index is exclusive. `y1_exclusive` retains its existing meaning.
Shape, channels and dtype are unchanged, as are all retained source pixels.
`--roi-margin-top` and `--roi-margin-bottom` override YAML margins independently.
Existing modes keep their previous floor/ceil and shared-margin semantics.

## Configuration and CSV

`config/roi_gabor_boundary.yaml` is the complete starting configuration. A
custom file can override selected sections/keys; unknown keys, invalid types,
nonfinite values and invalid ranges are rejected. All pixel distances are
native pixels, not percentages of the resized image.

Priority tuning parameters, to be calibrated using tuning GT only:

- `quality.ambiguity_score_gap: 0.02`, `ambiguity_boundary_gap_px: 30`.
- `quality.max_angle_diff_deg: 2.5`, `max_height_variation_px: 100`.
- `quality.min_x_coverage: 0.70`, `max_gap_ratio: 0.25`.
- `outside.ratio_threshold: 0.80`, `min_bin_fraction: 0.20` and
  `min_adjacent_bins: 3`; isolated scratches should not alone trigger clipping.
- Candidate windows, Gabor wavelength, bin width, and independent mask margins.

Existing CSV fields remain. Added aliases are `scene_id`, `image_path`,
`exposure_us`, `reference_image`, per-side `*_longest_gap` (pixels),
`*_ransac_rmse`, and `roi_height_left/center/right`. New diagnostics include:

- `angle_diff_deg`, `height_variation_px`.
- `best_candidate_score`, `second_candidate_score`, `candidate_score_gap`,
  `candidate_boundary_gap_px`: the highest-scoring distinct second geometry.
- `ambiguous_candidate_score`, `ambiguous_boundary_gap_px`: the conflicting
  near-score candidate, which may rank below second place.
- `outside_texture_top/bottom`: P90 outside/inside texture-energy ratio.
  The denominator is floored at `outside.min_inside_energy` to avoid division
  by near-zero energy. Weak interior texture does not excuse strong exterior texture.
  Detailed per-bin ratios/flags are in reference `quality_details_json`.
- `consensus_count/total/ratio`, `was_recovered`, `recovery_method`.
- `needs_review_reason`: semicolon-separated independent failures.
- `manual_override`, `roi_margin_top/bottom`, `mask_rounding`, `debug_path`.
  The legacy `margin` column equals the shared margin, or is blank for asymmetric
  margins. Reconstruct new masks using the separate margins and rounding field.
- `automatic_prediction_json`: reference automatic geometry and status before
  any override, for unbiased evaluator input. It also retains the automatic
  quality details, score components and consensus count.

Manual rows leave automatic score/gap, outside-ratio and consensus diagnostics
blank rather than attaching them to the corrected geometry. Their saved
automatic snapshot retains these diagnostics. Candidate/fusion JSON describes
automatic processing; RANSAC and outside-band debug panels also show automatic
geometry, while final-line and mask panels show the manual correction.
The runtime `skipped_images` count includes missing and unreadable references,
unreadable targets, and target-size mismatches, matching skipped CSV rows.

Example reasons: `AMBIGUOUS_CANDIDATES`, `LOW_X_COVERAGE_BOTTOM`,
`LARGE_ANGLE_DIFFERENCE`, `LARGE_HEIGHT_VARIATION`, `FABRIC_OUTSIDE_BOTTOM`,
`LOW_MULTI_EXPOSURE_CONSENSUS`, `OUTSIDE_CHECK_UNAVAILABLE_TOP`.

## Manual Overrides

Legacy `shape/top_y/bottom_y` files remain valid. Two points per line or four
corners are also supported, keyed by reference-relative source and mode:

```json
{
  "case01/01/000/10000us/mono_example.png": {
    "gabor_boundary": {
      "shape": [2046, 2046],
      "corners": [[0, 700], [2045, 710], [2045, 1600], [0, 1590]]
    }
  }
}
```

This is an illustration, not an annotation of a real specimen. Corner order is
TL, TR, BR, BL, or use a dictionary with those keys. Alternatively replace
`corners` with `top_points` and `bottom_points`, each two increasing-X `[x,y]`
points. Lines are evaluated across the full image width; vertical lines,
crossings and out-of-image extrapolation are rejected. Add
`--roi-overrides .\config\my_roi_overrides.json` to the extraction command.
Overrides propagate to all target exposures, without changing original files.

## Ground Truth Evaluation

Use approved human annotations, not automatically predicted coordinates. Both
the existing GT CSV and a JSON mapping are accepted. JSON values have `shape`,
`split` (`tune` or `validation`), `status` (`approved` or `pending`), and the
two-line or four-corner fields above, without a mode wrapper. Movement variants
and all captures of a base case must stay in one split.

`prepare_roi_gt.py --selection all` prepares pending reference records including
automatic passes. Review-only annotations cannot estimate population false-pass
rate. Use a validation count compatible with whole-case grouping.

After annotating GT, run separately:

```powershell
uv run python -m src.roi.evaluate_roi_boundaries `
  --report "$out\roi_results.csv" `
  --ground-truth .\config\roi_gt.json `
  --source .\data\0918 `
  --mode gabor_boundary `
  --output ".\data\roi_evaluation\$(Get-Date -Format yyyyMMdd_HHmmss)"
```

This writes `scene_metrics.csv` and `summary.json`, separately for tuning and
validation: top/bottom MAE, RMSE, maximum/P95 error; under-crop and over-crop
pixels/fractions; mask IoU; automatic scene-pass rate; and false-pass rate.
False-pass means an automatic PASS/RECOVERED that fails the declared GT criteria:
default IoU >= .98, under-crop fraction <= .005, both boundary P95 errors <= 15px.
These evaluation thresholds are CLI-configurable, not validated guarantees.
False-pass rate divides by automatic passes in the annotated split; it is null
when no automatic passes exist. Area fractions divide by GT ROI area. Margins
affect mask metrics, but geometric boundary errors compare unmargined lines.
Manual replacements are never substituted for the saved automatic prediction.

`--ground-truth` / `--evaluate-gt` on extraction also adds reference GT metrics
to `roi_results.csv` before overrides. A held-out evaluation is still required.

## Files and Verification Scope

- `src/roi/auto_roi_local_std.py`: CLI, masking, overrides and CSV implementation.
- `src/roi/roi_boundary.py`: features, pairs, global DP, side-only recovery, export/debug.
- `src/roi/roi_boundary_quality.py`: YAML validation and independent quality gates.
- `src/roi/auto_roi_group.py`: group and multi-exposure ROI logic.
- `src/roi/roi_annotations.py`: line/corner annotation parsing and JSON GT splits.
- `src/roi/evaluate_roi_boundaries.py`: standalone GT metrics and false-pass reporting.
- `prepare_roi_gt.py`: supports new mode and inclusion of automatic passes.
- `tests/test_roi_boundary.py`: synthetic pipeline, unit and regression tests.

```powershell
uv run python -m unittest discover -s tests -q
```

The regression fixtures contain measured v4_1 geometry/quality from
`case19/03/000`, `case13-moved/02/000`, `case04/02/000`, `case01/02/000` and
`case05/01/180`. The new gates must reject their recorded problematic geometry.
These are diagnostic regression tests, not new detections on the real images.

This change was requested as code plus commands only. Therefore the real 696
images were not reprocessed, and there are no new PASS/RECOVERED/NEEDS_REVIEW
counts, real success/failure image paths or new accuracy claims yet. The prior
v4_1 baseline had 82 internal passes and 4 review groups, not 95.3% GT accuracy.
The runtime `boundary_summary` and CSV provide the new counts after execution.

Remaining risks: glare or scratches that resemble a boundary, weak evidence on
one side, genuine curvature, and uncertain inclusion of loose fibers. Curved
models are intentionally not implemented/enabled in this version. Straight
lines plus explicit review are retained; no spline is silently applied.
