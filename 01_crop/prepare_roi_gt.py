"""Create an unannotated review manifest from a completed ROI report."""

import argparse
import csv
from pathlib import Path


FIELDS = ("source", "split", "status", "height", "width", "top_points_json", "bottom_points_json")


def prepare_rows(report: Path, mode: str, validation_count: int, selection: str = 'review') -> list[dict]:
    with report.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"is_reference", "mode", "needs_review", "source", "input_height", "input_width"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Not an ROI results CSV")
        selected = [r for r in reader if r["is_reference"] == "1" and r["mode"] == mode
                    and (selection == 'all' or r["needs_review"] == "1")]
    if not selected:
        raise ValueError("No matching references for this mode/selection")
    groups, seen = {}, set()
    for row in sorted(selected, key=lambda r: r["source"]):
        key = row["source"].casefold()
        if key in seen:
            raise ValueError(f"Duplicate reference: {row['source']}")
        seen.add(key)
        if int(row["input_height"] or 0) < 2 or int(row["input_width"] or 0) < 2:
            raise ValueError(f"Missing readable reference dimensions: {row['source']}")
        # Movement variants and repeated captures of the same case stay in one split.
        group = Path(row["source"]).parts[0].split("-", 1)[0].casefold()
        groups.setdefault(group, []).append(row)
    if not 0 < validation_count < len(selected):
        raise ValueError("Validation count must leave at least one tuning reference")
    subsets = {0: set()}
    for group in sorted(groups):
        for size, keys in list(subsets.items()):
            total = size + len(groups[group])
            if total <= validation_count and total not in subsets:
                subsets[total] = keys | {group}
    validation = subsets.get(validation_count)
    if validation is None:
        raise ValueError("Cannot make the requested split without sharing a case; choose another validation count")
    return [{"source": row["source"], "split": "validation" if group in validation else "tune",
             "status": "pending", "height": row["input_height"], "width": row["input_width"],
             "top_points_json": "", "bottom_points_json": ""}
            for group, rows in groups.items() for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("std", "fft", "gabor", "fft+gabor", "auto", "gabor_boundary"), default="gabor")
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument('--selection', choices=('review', 'all'), default='review',
                        help='Use all for a GT population that can measure false passes')
    args = parser.parse_args()
    try:
        rows = prepare_rows(args.report, args.mode, args.validation_count, args.selection)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"pending={len(rows)} tune={len(rows) - args.validation_count} validation={args.validation_count}")
    print(f"output={args.output}; no automatic coordinates were copied into ground truth")


if __name__ == "__main__":
    main()
