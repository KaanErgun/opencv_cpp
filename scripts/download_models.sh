#!/usr/bin/env bash
#
# download_models.sh — fetch the ONNX models + class names the apps expect into
# models/. Idempotent: already-present, checksum-matching files are skipped.
#
# v2.0 standardises on ONNX (see docs/DECISIONS.md K3); the old Darknet
# yolov3/yolov7 .weights are retired and NOT downloaded.
#
#   ./scripts/download_models.sh            # fetch everything
#   ./scripts/download_models.sh --only coco # just the class names
#
set -euo pipefail

cd "$(dirname "$0")/.."
DEST="models"
mkdir -p "$DEST"

ONLY="${2:-all}"
if [[ "${1:-}" == "--only" ]]; then ONLY="$2"; fi

sha() { shasum -a 256 "$1" | awk '{print $1}'; }

fetch() {  # url  dest  [expected_sha256]
  local url="$1" out="$2" want="${3:-}"
  if [[ -f "$out" && -n "$want" && "$(sha "$out")" == "$want" ]]; then
    echo "  ok (cached): $out"
    return
  fi
  echo "  fetching: $out"
  curl -fL --retry 3 -o "$out" "$url"
  if [[ -n "$want" && "$(sha "$out")" != "$want" ]]; then
    echo "!! checksum mismatch for $out" >&2
    echo "   expected $want" >&2
    echo "   got      $(sha "$out")" >&2
    exit 1
  fi
}

# COCO class names (for the generic yolov8n detector configs).
if [[ "$ONLY" == "all" || "$ONLY" == "coco" ]]; then
  echo "COCO class names:"
  fetch "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names" \
        "$DEST/coco.names"
fi

# Generic YOLOv8n ONNX (person/car/cow/... detection).
# Ultralytics models are AGPL-3.0; the model file is NOT committed. Export your
# own with: `yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12`
# then place the result at models/yolov8n.onnx, or set a mirror URL below.
if [[ "$ONLY" == "all" || "$ONLY" == "yolov8n" ]]; then
  echo "YOLOv8n ONNX:"
  if [[ -f "$DEST/yolov8n.onnx" ]]; then
    echo "  ok (present): $DEST/yolov8n.onnx"
  else
    echo "  !! models/yolov8n.onnx missing."
    echo "     Export it (Ultralytics, AGPL-3.0):"
    echo "       yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12"
    echo "     and place it at models/yolov8n.onnx"
  fi
fi

# Custom car+plate model (classes: Araba, Plaka). Hosted as a GitHub Release
# asset; set MODEL_BEST_URL to your release download URL.
if [[ "$ONLY" == "all" || "$ONLY" == "best" ]]; then
  echo "Custom best.onnx (car+plate):"
  BEST_SHA="5fdad91f6748a0f29b9ab4173170c730d94be321222697ddb442e41a5c8f2471"
  if [[ -f "$DEST/best.onnx" && "$(sha "$DEST/best.onnx")" == "$BEST_SHA" ]]; then
    echo "  ok (cached): $DEST/best.onnx"
  elif [[ -n "${MODEL_BEST_URL:-}" ]]; then
    fetch "$MODEL_BEST_URL" "$DEST/best.onnx" "$BEST_SHA"
  else
    echo "  !! models/best.onnx missing and MODEL_BEST_URL not set."
    echo "     Upload best.onnx as a GitHub Release asset and re-run with:"
    echo "       MODEL_BEST_URL=<url> ./scripts/download_models.sh --only best"
  fi
fi

echo "Done."
