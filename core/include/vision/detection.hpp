#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace vision {

// A single detected object in full-frame pixel coordinates.
struct Detection {
    cv::Rect box;
    int classId = -1;
    float confidence = 0.0F;
};

// Common interface for every detector backend (YOLO ONNX, HOG, ...).
class IDetector {
   public:
    virtual ~IDetector() = default;

    // Runs detection on a BGR frame and returns boxes in that frame's coordinates.
    virtual std::vector<Detection> detect(const cv::Mat& frame) = 0;

    // Human-readable class names indexed by classId; empty if the backend has none.
    virtual const std::vector<std::string>& classNames() const = 0;
};

}  // namespace vision
