#pragma once

#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

#include "vision/detection.hpp"

namespace vision {

// Compute backend/target for OpenCV DNN inference.
enum class Backend { CPU, OpenCL, CUDA };

struct YoloConfig {
    std::string onnxPath;                 // path to a YOLOv8/v11 ONNX model
    std::string classNamesPath;           // newline-delimited class names file (optional)
    std::vector<std::string> classNames;  // inline names; used if classNamesPath empty
    int inputSize = 640;                  // square network input (letterboxed)
    float confThreshold = 0.25F;
    float nmsThreshold = 0.45F;
    std::vector<int> classFilter;  // keep only these class ids; empty = keep all
    Backend backend = Backend::CPU;
};

// YOLOv8/v11 ONNX detector: letterbox preprocessing, correct transposed decode
// ([1, 4+nc, 8400] -> per-row [cx,cy,w,h, class scores], no objectness column),
// class-aware NMS via cv::dnn::NMSBoxesBatched, and boxes clamped to the frame.
class YoloDetector : public IDetector {
   public:
    explicit YoloDetector(const YoloConfig& config);

    std::vector<Detection> detect(const cv::Mat& frame) override;
    const std::vector<std::string>& classNames() const override { return classNames_; }

   private:
    YoloConfig config_;
    cv::dnn::Net net_;
    std::vector<std::string> classNames_;
};

}  // namespace vision
