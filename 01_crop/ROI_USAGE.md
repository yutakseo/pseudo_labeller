# Reference-Exposure ROI Masking

For the new boundary-focused mode with false-pass prevention gates, native-size
masked outputs, debug panels and a GT evaluator, see [ROI_BOUNDARY_USAGE.md](ROI_BOUNDARY_USAGE.md).
The original modes described below retain their previous behavior.

## Complete Group Auto Mode

For a single selected ROI per group and an output for every readable,
size-compatible target, use the new `auto` mode:

```powershell
uv run python -m auto_roi_local_std  --source "F:\2차 라벨러\pseudo_labeller\__data\20260918_(1)extracted"  --output "F:\2차 라벨러\pseudo_labeller\__data\20260918_(2)cropped"  --reference-exposure 10000us   --mode auto   --image-output masked
```

Use a different, empty output directory for each run. This command writes only
one CSV and one full-resolution black-masked PNG per compatible input, not tiles
or rectified images. Source dtype, dimensions and all pixels inside the selected
ROI are preserved. Only detection uses resized/normalized images.

Pipeline:

1. Keep the existing scene/capture/direction/camera grouping and reference identity.
2. Percentile-normalize the exposures independently for analysis; compute their
   local STD, FFT and brightness-edge profiles on corresponding X strips.
3. Take the median of these **feature profiles across exposures**, not just the
   median of raw intensities. By default all matching exposures contribute.
4. Generate STD, FFT coarse + STD, FFT coarse + Gabor, FFT, Gabor fallback and
   Edge/Hough candidates. Each texture path also offers a constant-thickness
   alternative fitted to pair midpoints; this does not force all ROIs to be
   parallel. Gabor uses a CLAHE-enhanced median normalized image.
5. Also generate an exposure-consensus candidate after rejecting endpoint
   outliers, requiring a majority of the contributing exposures. Consistency
   scores also include the fraction of all exposures that agree: two matching
   exposures out of six must not earn the same support as six out of six.
   Score all candidates against the same fused feature evidence.
6. For low-scoring groups, try ECC affine ROI transfer from up to three high-
   scoring groups of the **same base case, camera, direction and image shape**.
   The template set is frozen before transfers, preventing chained fallback drift.
   Correlation, transform scale, orientation, displacement and geometry are checked.
   Transferred candidates must still compete on the target's local evidence.
7. Select the highest-scoring geometrically valid ROI and reuse its native-size
   coordinates unchanged across that group's exposures.

The score combines texture contrast (0.25), boundary evidence (0.20), line fit
(0.15), spatial coverage/gaps (0.10), exposure consistency (0.15), and joint
inliers (0.05), plus a soft thickness-change preference (0.10). The latter
compares the independent-line model with constant-thickness alternatives instead
of forcing every accepted ROI to be parallel. Coverage and joint-inlier ratios are continuous terms, so crossing
74.975% vs 75% does not independently veto a candidate. Crossing boundaries,
nonfinite geometry, unreasonable heights/tilts and excessive thickness change
remain disallowed. The score is **not a calibrated probability or GT accuracy**.

New statuses and output policy:

| Status | Meaning | Image exported? |
| --- | --- | --- |
| `auto` | Selected candidate meets the quality threshold | Yes |
| `auto_fallback` | Gabor/Edge/registration alternative meets the threshold | Yes |
| `low_confidence` | Best valid candidate is below the quality threshold | Yes, `needs_review=1` |
| `full_frame_fallback` | No supported valid ROI; preserve the entire original frame | Yes, `roi_detected=0`, `needs_review=1` |
| `manual` | Explicitly supplied manual reference boundaries | Yes |
| `skipped` | Missing/ambiguous reference, unreadable file, or incompatible size | No |

The last fallback deliberately does **not** invent a centered crop that could
erase the inspected material. It preserves all pixels, so it is an output but
not successful ROI localization. An unreadable file cannot yield a valid image.
The existing four strict modes still skip failed ROI exports as before.

Additional CSV diagnostics:

- `auto_accepted`, `export_eligible`, `roi_detected`, `roi_quality`,
  `roi_candidate_count`, `selection_method`, `fallback_used`, `output_warning`.
- `fusion_count`, `consensus_count`, `consensus_spread_px` (native pixels).
- `registration_source`, `registration_correlation`.
- On reference rows only: `quality_components_json`, `candidates_json`,
  `fusion_sources_json`, `registration_transform_json`.

Candidate endpoints in `candidates_json` are explicitly named
`endpoints_analysis_px` and use the resized analysis grid. Final top/bottom
coordinates and point/residual diagnostics use native source pixels.
The registration matrix maps template source coordinates to target source
coordinates. `found` indicates available output geometry, including fallback;
use `roi_detected`, `needs_review` and `status` to distinguish actual localization
from an unverified or full-frame result. Do not equate output completeness with
boundary accuracy.

Controls:

- `--auto-max-size 1024`: largest analysis dimension; feature sizes are scaled
  to preserve approximately the same physical pixel scale as the strict modes.
- `--fusion-exposures 0`: use all exposures. A positive value limits the subset,
  always retaining the reference plus exposures sampled across the range.
- `--auto-min-quality 0.55`: heuristic review threshold, not an export veto.
- `--auto-fine-band 0.06`: fine search radius / image height around FFT estimates.
- `--registration-min-correlation 0.65`: minimum ECC correlation; not sufficient
  by itself to certify registration or ROI accuracy.

Stationary camera/object geometry across exposures is still required. Exposure
consensus is corroborating evidence, not proof of registration. The optional
manual GT/override workflow below also supports `auto`; evaluate held-out GT
before using these heuristic scores as production acceptance criteria.

Reference: [OpenCV ECC registration](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).

## Strict Single-Feature Modes

Run from the project root in PowerShell. Each real run requires a new or empty
output directory. Existing results are not deleted or mixed with a new CSV.

```powershell
uv run python -m src.roi.auto_roi_local_std `
  --source .\data\0918 `
  --output .\data\roied\reference_10000us_gabor_v3 `
  --reference-exposure 10000us `
  --mode gabor `
  --image-output masked
```

The reference image is processed once per scene/direction/camera. Its accepted
geometry is reused unchanged for all exposures of that group. The camera and
object must remain stationary between exposures. Images with different spatial
dimensions are skipped. Source images are never modified.

## Detection and Outputs

- All four modes remain available: `std`, `fft`, `gabor`, `fft+gabor`.
- Candidate evidence combines texture change, original-image Scharr Y magnitude,
  and signed inside/outside texture contrast (weights 0.45 / 0.15 / 0.40).
  Top candidates favor higher texture below; bottom candidates favor higher
  texture above. Brightness alone cannot create a boundary. Texture comparison
  uses offsets 15..60 px on each side, reduced for very small images.
- Top/bottom candidates form pairs constrained by height and interior/exterior
  texture. Second-order DP penalizes position, thickness and slope changes,
  not constant tilt itself. A high-confidence central X bin joins left/right DP
  messages; backtracking expands from that bin to both ends. All anchor choices
  and joining slope states are retained, not greedily fixed to one central peak.
  This preserves the global path objective; changing traversal alone is not an
  accuracy improvement.
- Top and bottom lines are fitted independently; they need not be parallel.
- Failed fits retry at 8, 12 and 16 pixels by default. Every attempt must meet
  all quality gates below; increasing tolerance never bypasses them.
- When only one side fails, the successful side is preserved. The other side is
  searched again using the same combined boundary evidence near the predicted
  band. When both sides fail, or their pair is inconsistent, a corridor-constrained
  pair search is attempted. Continued failure remains `needs_review`.
- Boundaries are straight lines in this version. Strongly curved or uncertain
  boundaries require review; they are not automatically replaced with horizontal
boundaries or an unverified curve.

Initial quality thresholds (not yet calibrated against human GT):

| Gate | Default | Option |
| --- | --- | --- |
| Inlier / selected point count | >= 0.55, at least 4 | `--min-inlier-ratio` |
| Inlier X span / full image width | >= 0.75 | `--min-x-coverage` |
| Support in left, center, right thirds | At least one each | Always required |
| Longest unsupported gap / full width | <= 0.25 | `--max-gap-ratio` |
| Inlier vertical RMSE | <= 8 px | `--ransac-residual` |
| Median absolute inlier vertical residual | <= 8 px | `--max-median-residual` |
| Endpoint ROI heights / image height | 0.32..0.55 | `--min-height-ratio`, `--max-height-ratio` |
| Endpoint height change / image height | <= 0.15 | `--max-height-change-ratio` |
| X positions supporting both lines / sampled X union | >= 0.40 | `--min-pair-inlier-ratio` |
| Absolute angle of either line | <= 20 degrees | `--max-tilt-degrees` |

Gap includes the distances from the image ends to the first/last inliers and
between adjacent inlier centers, not just consecutive empty-bin counts.
Coverage uses the full width, including the excluded 5% at each side.
Changing `--center-x-ratio` or `--strips` may make coverage unattainable; this
requires adjustment of sampling, not silent relaxation of the quality gate.
There is no forced parallelism. Margin remains 20 px and is applied only after
geometry validation; choose its size according to whether loose fibers are
part of the intended defect inspection region.

Status values:

- `fitted`: both original line fits passed.
- `recovered`: a failed fit passed a bounded retry or guided re-detection.
- `manual`: explicitly supplied reference boundaries passed geometry validation.
- `needs_review` / `no_roi`: no image is exported for this mode at any exposure
  of the group. Every affected image/mode is still recorded in the CSV.
- `skipped`: missing/ambiguous reference, unreadable file, or size mismatch.
  Unknown geometry and statistics are blank, not zero.

Exactly one image is exported for each accepted image/mode. Select its format:

- `--image-output masked` (default): original dimensions, dtype and retained
  pixels; the exterior is black. This preserves the original masking workflow.
- `--image-output crop`: black-mask the exterior, then trim to the ROI's Y
  bounding box. Width and dtype remain unchanged. No resampling is performed.
  Black triangular corners can remain around sloped boundaries.
- `--image-output rectified` (or `--rectify`): export only the perspective-
  rectified, resampled ROI. A homography is recorded in the CSV.

`--margin` adds vertical padding (20 pixels by default). Coordinates in the CSV
remain in the source image coordinate system regardless of output format.

Only a CSV and the accepted ROI images are written. No visualization, comparison
image, per-image JSON, reference_run.json, or review preview is generated.

```text
<output>/roi_results.csv
<output>/crops/<mode>/<source-relative-parent>/<original-name>_roi.png
```

The CSV is UTF-8 with BOM for Excel and has one row per image/mode, including
failures. It is flushed after each row, so completed rows remain readable if a
run is interrupted. An interrupted CSV is partial, not proof of run completion.
`--dry-run` computes results but writes neither CSV nor images.

The rejection/export behavior in this section applies to the original four
strict modes. Auto mode follows the distinct output policy documented above.

## CSV Analysis

- Identity: `source`, `scene`, `camera`, `exposure`, `mode`, `roi_source`,
  `reference_exposure`, `is_reference`, `roi_reused`.
- Decision: `status`, `reason`, `found`, `needs_review`, `output_written`.
- Output: `output_path` relative to the output root, `image_output`, input/output
  dimensions, dtype, channels and `output_origin_y` (blank for rectified output).
- ROI: `y0`, `y1_exclusive` (bounding box including margin), `margin`,
  `roi_area_ratio`, left/right height and mean height ratio (before margin).
- Each boundary (`top_*`, `bottom_*`): validity, slope, angle in degrees,
  intercept, endpoint Y positions before margin, point/inlier counts and ratio,
  inlier X span divided by the central search width, RMSE in pixels, tolerance,
  attempt count, reacquisition flag, and initial/final rejection reasons.
- Additional boundary quality: `x_coverage` (full width, unlike the legacy
  `span_ratio`), `max_gap_ratio`, `left_support`, `center_support`, `right_support`,
  and `median_residual_px`. Rejected fits retain available residual/support
  measurements rather than losing them on early rejection.
- Pair quality: `height_mean_px`, `height_std_px` of the fitted line separation
  across all source columns before margin, `height_change_ratio`, and
  `pair_inlier_ratio`. These can diagnose rejected fits too; they are not pixel
  masks or proof that the physical boundaries were located correctly.
- Reference rows only: `top_points_json`, `top_inliers_json` and bottom equivalents
  contain point lists and inlier flags for deeper diagnosis without extra files.
- Candidate diagnostics: selected pair count, empty column count, mean pair score,
  and `anchor_x` of the initial pair path. Reacquired fits have their own point
  lists, while these candidate statistics continue to describe the initial path.
- Target statistics: `roi_gray_mean`, `roi_gray_std` use original source pixels
  inside the retained ROI including margin, not the black-filled exterior.
  `roi_zero_fraction` / `roi_saturated_fraction` are the fractions of retained
  intensity-channel samples at zero / dtype maximum; alpha is excluded.
- Reproducibility: detector version, UTC run start, parameters as one JSON-valued
  CSV cell, and optional `homography_json`. No separate JSON files are written.

Boolean fields use 1/0; unavailable values are blank. Zero RMSE and zero
saturation are valid measured values, distinct from missing measurements.
Manual boundaries have no measured RANSAC RMSE or tolerance.
They also have no automatic coverage, gap, or joint-inlier quality measurements.

For detector success rates, filter `is_reference=1` so repeated exposure copies
do not inflate the sample count. Use all exposure rows for brightness statistics.
Prioritize `needs_review=1`, then inspect `status=recovered` and large residuals
or low support ratios. These are diagnostics, not calibrated accuracy scores:
even a well-fitted line may follow the wrong physical edge. Compare with source
images or annotated ground truth before claiming ROI accuracy.

## Manual Reference Corrections

`--roi-overrides .\config\roi_overrides.json` accepts this JSON structure. Replace
the example path and coordinates with measurements from the actual reference.
Paths are relative to `--source` and must identify the reference exposure.

```json
{
  "case01/01/000/10000us/mono_example.png": {
    "gabor": {
      "shape": [420, 640],
      "top_y": [100, 145],
      "bottom_y": [285, 315]
    }
  }
}
```

`shape` is `[height, width]`. Each boundary contains the Y coordinate at `x=0`
and `x=width-1`, before margin is added. Coordinates are zero-based; the bottom
boundary is exclusive. The two lines may have different slopes. Only the listed
mode is overridden. Shape, coordinate, tilt and height constraints are checked
before applying the reference geometry to other exposures.

Relevant controls are `--ransac-residual`, `--ransac-max-residual`,
`--recovery-band-ratio`, `--slope-weight`, `--max-tilt-degrees`,
`--min-height-ratio`, and `--max-height-ratio`. Raising tolerances does not by
itself establish accuracy. Inspect representative ROI images against their
sources, especially `recovered` results, before using the results downstream.

## Manual GT and Held-Out Evaluation

Prepare a review CSV from the previous result, without running image detection:

```powershell
uv run python -m src.roi.prepare_roi_gt `
  --report .\data\roied\reference_10000us_gabor_csv\roi_results.csv `
  --output .\config\roi_gt_review.csv `
  --mode gabor `
  --validation-count 4
```

For the existing report this selects 11 failed references, with 7 tuning and
4 validation images. The split is deterministic, case-disjoint (including
movement variants), not a stratified sample of all 86 references. An impossible
case-disjoint count produces an error instead of leaking cases across splits.
Use only `tune` images to adjust thresholds; freeze settings before inspecting
held-out validation errors. Because these are previously failed images, also
review representative previously accepted images for regressions.

The helper creates a new input manifest, never overwrites an existing one, and
does not copy predictions into GT. All rows start `pending`; this is NOT a
completed manual correction. The actual reference must be inspected and its
boundary coordinates entered before changing a row to `approved`.

CSV columns: `source,split,status,height,width,top_points_json,bottom_points_json`.
For a **640-pixel-wide example only**, the two point-list cells might contain:

```json
[[0, 100], [639, 145]]
```

```json
[[0, 285], [639, 315]]
```

These are top then bottom, expressed as `[x,y]`, from `x=0` to `x=width-1`.
Use measured source-image coordinates, with no margin; bottom is exclusive.
The CSV point-list fields must be quoted as CSV cells. For actual curved GT,
more points with strictly increasing X are allowed and linearly interpolated
for evaluation. Two endpoints alone cannot establish whether the real edge is
curved. Set the physical boundary definition consistently, including the policy
for loose fibers.

Evaluate automatic results without replacing them:

```powershell
uv run python -m src.roi.auto_roi_local_std `
  --source .\data\0918 `
  --output .\data\roied\reference_10000us_gabor_gt_eval `
  --reference-exposure 10000us `
  --mode gabor `
  --ground-truth .\config\roi_gt_review.csv `
  --image-output masked
```

To also apply approved **two-endpoint** GT as manual masks for every exposure,
add `--apply-ground-truth` and use a different, empty output directory. This
overrides all selected modes, is incompatible with `--roi-overrides`, and refuses
dense/curved GT: a curve must not silently become a straight-line manual mask.
`pending` rows are neither evaluated nor applied.

Additional result CSV fields, on reference rows only:

- `gt_split`: `tune` or `validation`.
- `gt_auto_status`, `gt_auto_found`: status before manual replacement.
- `gt_top_mae_px`, `gt_bottom_mae_px`: mean absolute Y error at every source X,
  against human GT, before margin and **before any manual override**.

Missing automatic boundaries yield blank MAE, not zero; a valid surviving side
of a rejected ROI can still have its own MAE. Count `gt_auto_found` failures
alongside errors so rejecting hard images does not appear to improve accuracy.
Target-exposure rows leave GT fields blank to prevent counting the same geometry
multiple times. With `--apply-ground-truth`, `status=manual` describes the exported
mask but `gt_auto_*` still describes the uncorrected algorithm.

The regular batch continues to write only `roi_results.csv` and ROI PNGs. No GT
manifest is created automatically during a batch. Curves/splines are intentionally
not implemented until dense human annotations demonstrate a need and held-out
boundary error improves, not merely the candidate fitting residual.

Reference: [OpenCV Scharr derivative documentation](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html).
