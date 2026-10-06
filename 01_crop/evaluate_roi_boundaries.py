"""Evaluate automatic ROI boundaries, excluding manual replacement leakage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import auto_roi_local_std as core


METRIC_FIELDS = ('top_mae_px', 'bottom_mae_px', 'top_rmse_px', 'bottom_rmse_px',
                 'top_max_error_px', 'bottom_max_error_px', 'top_p95_error_px', 'bottom_p95_error_px',
                 'under_crop_pixels', 'over_crop_pixels', 'under_crop_fraction', 'over_crop_fraction', 'mask_iou')


def boundary_metrics(prediction: dict, spec: dict) -> dict:
    h, w = spec['shape']
    xs = np.arange(w, dtype=float)
    predictions, truth = {}, {}
    metrics = {}
    for side in ('top', 'bottom'):
        line = prediction[side + '_line']
        points = np.asarray(spec[side], dtype=float)
        values = line['slope'] * xs + line['intercept']
        if not np.isfinite(values).all():
            raise ValueError('nonfinite prediction')
        gt = np.interp(xs, points[:, 0], points[:, 1])
        error = np.abs(values - gt)
        metrics.update({side + '_mae_px': float(error.mean()), side + '_rmse_px': float(np.sqrt(np.mean(error**2))),
                        side + '_max_error_px': float(error.max()), side + '_p95_error_px': float(np.percentile(error, 95))})
        predictions[side], truth[side] = values, gt
    mt, mb = prediction.get('roi_margin_top', 0), prediction.get('roi_margin_bottom', 0)
    if prediction.get('mask_rounding') == 'nearest_inclusive':
        top, bottom = np.rint(predictions['top']) - mt, np.rint(predictions['bottom']) + mb + 1
    else:
        top, bottom = np.floor(predictions['top'] - mt), np.ceil(predictions['bottom'] + mb)
    top, bottom = np.clip(top, 0, h), np.clip(bottom, 0, h)
    gt_top, gt_bottom = np.clip(np.rint(truth['top']), 0, h), np.clip(np.rint(truth['bottom']) + 1, 0, h)
    # Column interval arithmetic is exactly equivalent to full pixel masks.
    area = np.maximum(0, bottom - top)
    gt_area = np.maximum(0, gt_bottom - gt_top)
    overlap = np.maximum(0, np.minimum(bottom, gt_bottom) - np.maximum(top, gt_top))
    under, over = int(np.sum(gt_area - overlap)), int(np.sum(area - overlap))
    union = float(np.sum(gt_area + area - overlap))
    metrics.update(under_crop_pixels=under, over_crop_pixels=over,
                   under_crop_fraction=under / max(1, int(gt_area.sum())),
                   over_crop_fraction=over / max(1, int(gt_area.sum())),
                   mask_iou=float(overlap.sum() / union) if union else 1.)
    return metrics


def prediction_from_row(row: dict) -> dict | None:
    if row.get('automatic_prediction_json'):
        return json.loads(row['automatic_prediction_json'])
    if row.get('manual_override') == '1' or row['status'] == 'manual':
        return None
    if not all(row.get(s + '_' + k) for s in ('top', 'bottom') for k in ('slope', 'intercept')):
        return None
    return {**{s + '_line': {k: float(row[s + '_' + k]) for k in ('slope', 'intercept')}
               for s in ('top', 'bottom')},
            'quality_status': row.get('quality_status') or row['status'],
            'roi_detected': row.get('roi_detected', row.get('found')) == '1',
            'roi_margin_top': float(row.get('roi_margin_top') or row.get('margin') or 0),
            'roi_margin_bottom': float(row.get('roi_margin_bottom') or row.get('margin') or 0),
            'mask_rounding': row.get('mask_rounding') or 'floor_ceil_exclusive'}


def evaluate_report(report: Path, annotations: dict, mode: str, min_iou: float, max_under: float,
                    max_p95: float) -> tuple[list, dict]:
    with report.open(encoding='utf-8-sig', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r.get('is_reference') == '1' and r['mode'] == mode]
    by_source = {}
    for row in rows:
        key = row['source'].casefold()
        if key in by_source:
            raise ValueError(f'duplicate reference result: {key}')
        by_source[key] = row
    results = []
    for name, spec in annotations.items():
        row = by_source.get(name)
        out = {'source': name, 'split': spec['split'], 'evaluation_status': 'unavailable',
               'auto_pass': False, 'geometry_evaluated': False, 'acceptable': False, 'false_pass': False}
        if row is not None:
            dimensions = [row.get('input_height'), row.get('input_width')]
            if all(dimensions) and [int(v) for v in dimensions] != spec['shape']:
                raise ValueError(f'GT shape mismatch: {name}')
            prediction = prediction_from_row(row)
            if prediction is not None:
                accepted = prediction['quality_status'] in {'PASS', 'RECOVERED', 'auto', 'auto_fallback', 'fitted', 'recovered'}
                if prediction.get('roi_detected', False):
                    metrics = boundary_metrics(prediction, spec)
                    acceptable = (metrics['mask_iou'] >= min_iou and metrics['under_crop_fraction'] <= max_under
                                  and max(metrics['top_p95_error_px'], metrics['bottom_p95_error_px']) <= max_p95)
                    out.update(metrics, geometry_evaluated=True, acceptable=acceptable, evaluation_status='evaluated')
                else:
                    out['evaluation_status'] = 'no_localized_roi'
                out.update(auto_pass=accepted, false_pass=accepted and not out['acceptable'])
        results.append(out)
    summary = {'criteria': {'min_iou': min_iou, 'max_under_crop_fraction': max_under, 'max_boundary_p95_px': max_p95},
               'false_pass_rate_definition': 'false_pass_count / automatic_pass_count on approved GT scenes',
               'splits': {}}
    for split in ('tune', 'validation'):
        subset = [r for r in results if r['split'] == split]
        measured = [r for r in subset if r['geometry_evaluated']]
        passes = sum(r['auto_pass'] for r in subset)
        false_passes = sum(r['false_pass'] for r in subset)
        summary['splits'][split] = {'approved_gt_scenes': len(subset), 'geometry_evaluated': len(measured),
                                    'automatic_pass_count': passes, 'false_pass_count': false_passes,
                                    'scene_pass_rate': passes / len(subset) if subset else None,
                                    'false_pass_rate': false_passes / passes if passes else None,
                                    'false_pass_per_gt_scene': false_passes / len(subset) if subset else None,
                                    'metric_means': {k: float(np.mean([r[k] for r in measured])) if measured else None
                                                     for k in METRIC_FIELDS}}
    return results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('data/0918'))
    parser.add_argument('--reference-exposure', default='10000us')
    parser.add_argument('--mode', default='gabor_boundary')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-iou', type=float, default=.98)
    parser.add_argument('--max-under-crop-fraction', type=float, default=.005)
    parser.add_argument('--max-boundary-p95', type=float, default=15.)
    args = parser.parse_args()
    if (not 0 <= args.min_iou <= 1 or not 0 <= args.max_under_crop_fraction <= 1
            or not np.isfinite(args.max_boundary_p95) or args.max_boundary_p95 < 0):
        parser.error('invalid evaluation thresholds')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('evaluation output must be new or empty')
    annotations = core.load_ground_truth(args.ground_truth, args.source, args.reference_exposure)
    if not annotations:
        parser.error('no approved GT annotations; automatic predictions are not ground truth')
    try:
        rows, summary = evaluate_report(args.report, annotations, args.mode, args.min_iou,
                                        args.max_under_crop_fraction, args.max_boundary_p95)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output / 'scene_metrics.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / 'summary.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
