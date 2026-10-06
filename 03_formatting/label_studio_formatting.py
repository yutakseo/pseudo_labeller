"""Build a tree-preserving, pre-annotated Label Studio segmentation dataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "__data"
SOURCE_IMAGES = DATA_ROOT / "20260918_(2)cropped" / "crops" / "auto"
SOURCE_MASKS = DATA_ROOT / "20260922_(3)labelled" / "masks"
SOURCE_LABELMAP = DATA_ROOT / "20260922_(3)labelled" / "outputs" / "labelmap.txt"
OUTPUT_ROOT = DATA_ROOT / "20260922_(4)labelstudio"

# These names must match label_config.xml and the fields used in tasks.json.
IMAGE_NAME = "image"
BRUSH_NAME = "brush"


def class_names(labelmap_path: Path = SOURCE_LABELMAP) -> list[str]:
    """Read the zero-based YOLO class map."""
    names = [line.strip() for line in labelmap_path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names:
        raise ValueError(f"Empty label map: {labelmap_path}")
    return names


def _bits_to_bytes(bits: str) -> list[int]:
    return [int(bits[index : index + 8], 2) for index in range(0, len(bits), 8)]


def encode_label_studio_rle(mask: np.ndarray) -> list[int]:
    """Encode a binary mask using Label Studio's RGBA RLE representation."""
    values = np.repeat(mask.astype(np.uint8).ravel(), 4)
    run_ends = np.flatnonzero(np.diff(values) != 0)
    run_starts = np.concatenate(([0], run_ends + 1))
    run_lengths = np.diff(np.concatenate((run_starts, [len(values)])))
    run_values = values[run_starts]

    # This is the public Label Studio converter's RLE wire format.
    header = f"{len(values):032b}" + f"{7:05b}" + "0010001101111111"
    chunks: list[str] = []
    for length, value in zip(run_lengths, run_values):
        remaining = int(length)
        integer_value = int(value)
        while remaining:
            chunk = min(remaining, 65536)
            chunks.append(
                "0" + "00" + f"{integer_value:08b}"
                if chunk == 1
                else "1" + "11" + f"{chunk - 1:016b}" + f"{integer_value:08b}"
            )
            remaining -= chunk
    body = "".join(chunks)
    padding = (-len(header + body)) % 8
    return _bits_to_bytes(header + body + "0" * padding)


def brush_result(mask_path: Path, class_name: str) -> dict[str, object]:
    """Convert one binary PNG mask to a single editable Label Studio brush mask."""
    with Image.open(mask_path).convert("L") as image:
        mask = np.where(np.asarray(image) > 128, 255, 0).astype(np.uint8)
        width, height = image.size
    return {
        "id": uuid4().hex[:10],
        "from_name": BRUSH_NAME,
        "to_name": IMAGE_NAME,
        "type": "brushlabels",
        "origin": "manual",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {"format": "rle", "rle": encode_label_studio_rle(mask), "brushlabels": [class_name]},
    }


def build_dataset(
    source_images: Path = SOURCE_IMAGES,
    source_masks: Path = SOURCE_MASKS,
    source_labelmap: Path = SOURCE_LABELMAP,
    output_root: Path = OUTPUT_ROOT,
) -> tuple[int, int]:
    """Copy tree-preserved images and masks, then create editable brush tasks."""
    names = class_names(source_labelmap)
    image_paths = sorted(path for path in source_images.rglob("*") if path.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {source_images}")

    images_root = output_root / "images"
    tasks: list[dict[str, object]] = []
    mask_count = 0
    for image_path in image_paths:
        relative_image = image_path.relative_to(source_images)
        mask_path = source_masks / relative_image
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {relative_image}: {mask_path}")

        destination = images_root / relative_image
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, destination)
        mask_destination = output_root / "masks" / relative_image
        mask_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mask_path, mask_destination)
        results = [brush_result(mask_path, names[0])]
        mask_count += 1
        tasks.append(
            {
                "data": {
                    # Set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT to OUTPUT_ROOT.
                    "image": "/data/local-files/?d=images/" + relative_image.as_posix(),
                },
                # `annotations`, rather than `predictions`, makes masks editable.
                "annotations": [{"result": results}],
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    imports_root = output_root / "imports"
    imports_root.mkdir(exist_ok=True)
    for batch_index, start in enumerate(range(0, len(tasks), 10), 1):
        (imports_root / f"tasks_{batch_index:03d}.json").write_text(
            json.dumps(tasks[start : start + 10], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (output_root / "label_config.xml").write_text(
        "<View>\n"
        '  <Image name="image" value="$image"/>\n'
        '  <BrushLabels name="brush" toName="image">\n'
        + "\n".join(f'    <Label value="{name}"/>' for name in names)
        + "\n  </BrushLabels>\n</View>\n",
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        "# Label Studio segmentation dataset\n\n"
        "- `images/` preserves the original directory tree.\n"
        "- `masks/` preserves the binary source masks using the same tree.\n"
            f"- `tasks.json` contains all {len(tasks)} editable RLE brush-mask tasks.\n"
            "- `imports/tasks_*.json` splits them into batches of 10 for browser upload.\n"
        "- Paste `label_config.xml` into the project's Labeling Interface.\n\n"
        "For local Label Studio, configure `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` "
            "to this folder, enable local-file serving, then import the files in `imports/` "
            "(or `tasks.json` if the upload limit permits). "
        "The task image URLs intentionally use `images/<original relative path>`.\n",
        encoding="utf-8",
    )
    return len(tasks), mask_count


if __name__ == "__main__":
    images, masks = build_dataset()
    print(f"Created {OUTPUT_ROOT}")
    print(f"Images: {images}; brush masks: {masks}")
