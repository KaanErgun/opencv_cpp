# models/

Model files and class-name lists live here. **None of them are committed** —
they are `.gitignore`d (except this README) and fetched on demand.

Run:

```bash
./scripts/download_models.sh
```

## Expected layout

| File | Used by | Source |
|------|---------|--------|
| `coco.names` | generic detector configs | pjreddie/darknet (auto-downloaded) |
| `yolov8n.onnx` | `configs/{car,cow,human}_*.json` | export yourself (see below) |
| `best.onnx` | `configs/{car_plates_v8,alpr}.json` | custom car+plate model; host as a GitHub Release asset |

## yolov8n.onnx

Ultralytics models are AGPL-3.0, so the file is not committed. `download_models.sh`
exports it automatically in a throwaway `python3.12` virtualenv (needs internet).
To do it by hand instead:

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
mv yolov8n.onnx models/
```

Use `opset=12` and a static `640x640` input — OpenCV's DNN importer is happiest
with static shapes. Verified against OpenCV 5.0 DNN (bus.jpg → 4 person + 1 bus).

## best.onnx

The custom car+plate model (classes `Araba`, `Plaka`). Upload it as a GitHub
Release asset and fetch it with:

```bash
MODEL_BEST_URL=<release-download-url> ./scripts/download_models.sh --only best
```

Its expected SHA-256 is pinned in `scripts/download_models.sh`.
