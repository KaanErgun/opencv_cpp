# opencv_cpp — v2.0

A small C++/OpenCV computer-vision toolkit built around a shared core library
and a handful of thin, config-driven apps. YOLOv8/v11 ONNX object detection, a
classic HOG pedestrian backend, RTSP/webcam/file capture with automatic
reconnect, and IoU tracking for multi-camera counting.

> v2.0 replaced the previous collection of 10 copy-pasted single-file demos
> with one `core/` library + four apps. The old modules live in the `pre-v2`
> git tag. See [`V2_UPGRADE_PLAN.md`](V2_UPGRADE_PLAN.md) for the full story and
> [`docs/DECISIONS.md`](docs/DECISIONS.md) for the design decisions.

## Apps

| Target | What it does | Replaces |
|--------|--------------|----------|
| `app_detect` | Generic single-source detection (YOLO or HOG), config-driven | yolov3/v7/v8 car/cow/human demos, HOG demo |
| `app_multicam` | Multi-camera 3×3 ROI grid + IoU tracking + counting | car_detection_dual*, multi_thread_rtsp |
| `app_rtsp_record` | View or record any source; correct fps/size, clean SIGINT | simple_rtsp |
| `app_alpr` | Plate detection (ONNX) + clean crop saving | alpr_plate_detection, yolov3_plate_recognition |

Each old demo is now a JSON file in [`configs/`](configs/) — behaviour changes
without recompiling.

## Prerequisites

- **macOS:** `brew install opencv cmake ninja clang-format`
- **Linux:** `sudo apt install libopencv-dev cmake ninja-build clang-format clang-tidy`

Needs OpenCV ≥ 4.8 (for `NMSBoxesBatched`). OpenCV 5.x also works. `nlohmann_json`
and `Catch2` are fetched automatically by CMake.

> The HOG backend uses `HOGDescriptor`, which moved to the `xobjdetect` contrib
> module in OpenCV 5.x. On OpenCV 4.x it ships with core `objdetect`. CMake
> detects this automatically.

## Build

```bash
./scripts/download_models.sh                       # fetch ONNX models + coco.names
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

On an Apple-Silicon Mac with an x86_64 Homebrew OpenCV, add
`-DCMAKE_OSX_ARCHITECTURES=x86_64` and `-DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv5`.

## Run

```bash
./build/apps/app_detect  --config configs/car_webcam.json
./build/apps/app_detect  --config configs/human_hog.json
./build/apps/app_alpr    --config configs/alpr.json --save-dir plates/
./build/apps/app_multicam --config configs/multicam.json
./build/apps/app_rtsp_record --source rtsp://USER:PASS@CAMERA_IP/stream --output out.mp4 --seconds 60
```

Every app takes `--config <file.json>`; `app_detect`/`app_alpr` also accept
`--source` and `--headless`. Press **ESC** or **q** to quit a GUI window.

## Models

Model files are **never committed** (gitignored) and fetched into `models/` by
`scripts/download_models.sh`. See [`models/README.md`](models/README.md) for the
expected layout and how to export/host each one. YOLOv8 must be exported with a
static `640×640` input and `opset=12`.

## Verification

There is **no CI** (project rule). The single local quality gate is:

```bash
./scripts/check.sh    # build + unit tests + clang-format + credential scan
```

## Security

Never commit RTSP URLs, camera passwords, or API keys. Use config files with
placeholder/env-referenced secrets. The pre-v2 history contained real leaked
credentials; [`scripts/purge_history.sh`](scripts/purge_history.sh) rewrites
history to remove them (must be run manually — it force-pushes).

## License

[MIT](LICENSE). Bundled/third-party models carry their own licenses (Ultralytics
YOLO is AGPL-3.0 — its weights are not distributed here).
