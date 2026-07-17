#pragma once

#include <string>
#include <vector>

#include "vision/yolo_detector.hpp"

namespace vision {

// App-level configuration loaded from a JSON file. Secrets (rtsp urls with
// credentials) should come from the config or an env-var reference, never from
// compiled-in constants.
struct AppConfig {
    // Input
    std::string source = "0";  // SourceSpec string: "0", "clip.mp4", "rtsp://..."

    // Detector
    std::string detector = "yolo";        // "yolo" | "hog"
    std::string model;                    // onnx path (yolo)
    std::string classNamesPath;           // optional names file
    std::vector<std::string> classNames;  // inline names (used if path empty)
    int inputSize = 640;
    float confThreshold = 0.25F;
    float nmsThreshold = 0.45F;
    std::vector<int> classFilter;  // keep only these class ids
    std::string backend = "cpu";   // "cpu" | "opencl" | "cuda"

    // Output / runtime
    bool headless = false;   // no GUI; write annotated video if outputPath set
    std::string outputPath;  // annotated video sink (optional)
    int maxFrames = 0;       // 0 = unbounded; >0 stops after N frames

    // Loads from a JSON file. Missing keys keep their defaults.
    static AppConfig fromFile(const std::string& path);

    // Builds a YoloConfig from the detector-related fields.
    YoloConfig toYoloConfig() const;
    Backend parseBackend() const;
};

}  // namespace vision
