# opencv_cpp

A small, modern C++/OpenCV computer-vision toolkit. One shared library, four
tiny apps, and JSON configs — no copy-paste between demos.

- **Object detection** with YOLOv8/v11 ONNX models (or a classic HOG people detector)
- **Any input**: webcam, video file, or RTSP stream — with automatic reconnect
- **Multi-camera** ROI counting with lightweight IoU tracking
- **License plates**: detect + save clean crops

## Quick start

```bash
# 1. Install dependencies (macOS). tesseract is only needed for the ALPR
#    server/client; everything else builds without it.
brew install opencv cmake ninja tesseract

# 2. Fetch models (exports YOLOv8n, downloads coco.names)
./scripts/download_models.sh

# 3. Build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 4. Run — detect people on your webcam
./build/apps/app_detect --config configs/human_yolo.json
```

Press **ESC** or **q** to quit any window.

> On Linux: `sudo apt install libopencv-dev cmake ninja-build`.
> On an Apple-Silicon Mac with an x86_64 Homebrew OpenCV, add
> `-DCMAKE_OSX_ARCHITECTURES=x86_64 -DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv5`
> to the `cmake` configure step.

## Learning path (classic CV)

Ten small, heavily-commented apps that build on each other — start at the top.
Every app prints its flags when run without arguments, quits on **ESC**/**q**,
and supports `--headless --max-frames N` for scripted runs. A source is a webcam
index (`0`), a video file (`clip.mp4`), or an RTSP URL.

| # | App | You learn | Try it |
|---|-----|-----------|--------|
| 1 | **app_image_ops** | imread/imwrite, Mat basics, resize, rotate, blur, Canny, histograms | `app_image_ops --image photo.jpg --show` |
| 2 | **app_filters** | blur family (box/gaussian/median/bilateral), morphology — with live trackbars | `app_filters --source 0` |
| 3 | **app_edges** | Sobel gradients, Canny hysteresis, threshold tuning | `app_edges --source 0` |
| 4 | **app_contours** | thresholding, findContours, areas, moments/centroids | `app_contours --source 0` |
| 5 | **app_color_track** | HSV colour space, inRange segmentation, blob tracking | `app_color_track --source 0` |
| 6 | **app_face_detect** | Haar cascades, detectMultiScale parameters, ROI search | `app_face_detect --source 0` |
| 7 | **app_motion_detect** | MOG2 background subtraction, shadow handling | `app_motion_detect --source 0` |
| 8 | **app_optical_flow** | Shi-Tomasi corners, pyramidal Lucas-Kanade flow | `app_optical_flow --source 0` |
| 9 | **app_object_track** | single-object trackers (CSRT/KCF/MIL), init/update lifecycle | `app_object_track --source 0` |
| 10 | **app_qr_scanner** | QR encode + detect + decode, perspective quads | `app_qr_scanner --encode "hi" && app_qr_scanner --image qr.png` |

## Pipeline apps (DNN / YOLO)

| App | What it does | Example |
|-----|--------------|---------|
| **app_detect** | Detect objects from one source (YOLO ONNX or HOG) | `app_detect --config configs/car_webcam.json` |
| **app_multicam** | Multiple cameras, ROI grid + IoU tracking + counting | `app_multicam --config configs/multicam.json` |
| **app_rtsp_record** | View or record any stream to a video file | `app_rtsp_record --source rtsp://... --output out.mp4` |
| **app_alpr** | Detect license plates and save cropped images (no OCR) | `app_alpr --config configs/alpr.json --save-dir plates/` |

## License-plate recognition (client / server)

A full ALPR system that **reads** plate characters (YOLO detection → Tesseract
OCR), split into a recognition server and thin capture clients. Needs Tesseract
(`brew install tesseract`); the targets are skipped if it's absent.

```bash
# 1. Start the recognition server (loads best.onnx + Tesseract, serves HTTP)
./build/apps/app_alpr_server --config configs/alpr_server.json --port 8080

# 2a. Web dashboard — open http://localhost:8080 and drop in a car photo
# 2b. C++ capture client — one image, or a live stream
./build/apps/app_alpr_client --image car.jpg --server http://localhost:8080
./build/apps/app_alpr_client --source rtsp://CAMERA --server http://host:8080 --log plates.csv
```

The server also speaks plain HTTP, so any client works:

```bash
curl -X POST --data-binary @car.jpg http://localhost:8080/recognize
# {"count":1,"ms":88,"plates":[{"text":"YIM97B","ocr_confidence":0.96,
#   "det_confidence":0.88,"box":[411,584,110,36]}]}
```

Endpoints: `POST /recognize` (image → plates JSON), `GET /events` (recognition
history, appended to `alpr_events.jsonl`), `GET /health`, `GET /` (dashboard).
OCR accuracy tracks image quality — a clean plate reads exactly; a small, blurry,
or low-light plate may read partially.

Behaviour comes from the JSON files in [`configs/`](configs/) — change the source,
model, thresholds, or class filter without recompiling. `app_detect` and
`app_alpr` also accept `--source <spec>` and `--headless` on the command line.

## Models

Model files are fetched into [`models/`](models/) and are never committed. Run
`./scripts/download_models.sh` to get them:

- **yolov8n.onnx** — general COCO detection (person, car, cow, …); exported from
  Ultralytics
- **coco.names** — class labels
- **best.onnx** — a custom car + plate model used by `app_alpr`

See [`models/README.md`](models/README.md) for details and how to provide your own.

## Configuration example

```json
{
  "source": "0",
  "detector": "yolo",
  "model": "models/yolov8n.onnx",
  "class_names_path": "models/coco.names",
  "class_filter": [2],
  "conf_threshold": 0.5
}
```

`class_filter` keeps only the given COCO class ids (here `2` = car). Leave it
empty to keep everything.

## Project layout

```
core/      two libraries: vision_core (detection, capture, tracking, drawing,
           config, cli) and vision_alpr (Tesseract OCR + recognition pipeline)
apps/      16 thin executables: 10 classic-CV lessons, 4 DNN pipeline apps,
           and the ALPR server + client
configs/   ready-made JSON configs for the DNN and ALPR apps
models/    ONNX models + class names (downloaded, not committed)
scripts/   download_models.sh, check.sh
tests/     unit tests (Catch2)
```

## Development

There's no CI — run the local quality gate before pushing:

```bash
./scripts/check.sh    # build + tests + formatting + credential scan
```

Never hardcode RTSP URLs, camera passwords, or API keys — use config files with
placeholder or env-referenced values.

## License

[MIT](LICENSE). Third-party models carry their own licenses (Ultralytics YOLO is
AGPL-3.0; its weights are not distributed here).
