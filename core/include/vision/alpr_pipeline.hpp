#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "vision/plate_ocr.hpp"
#include "vision/yolo_detector.hpp"

namespace vision {

// One recognised (or merely detected) plate in a frame.
struct PlateResult {
    std::string text;  // OCR text; empty if the plate could not be read
    float ocrConfidence = 0.0F;
    float detConfidence = 0.0F;  // detector confidence for the plate box
    cv::Rect box;                // plate location in frame pixels
};

struct AlprConfig {
    YoloConfig detector;
    OcrConfig ocr;
    int plateClassId = 0;  // best.onnx emits class 0 for the plate
    // Grow the plate box by this fraction before OCR. best.onnx boxes are tight
    // but complete, so 0 is best here; raise it for detectors that clip glyphs
    // (extra background can otherwise skew the crop's Otsu threshold).
    float padding = 0.0F;
    float minOcrConfidence = 0.30F;  // below this, keep the box but drop the text
};

// Detect plates with YoloDetector, then OCR each clean crop. Thread-safe: a
// single instance can be shared across HTTP worker threads (recognition is
// serialised internally, since neither the DNN net nor Tesseract is reentrant).
class AlprPipeline {
   public:
    explicit AlprPipeline(const AlprConfig& cfg);

    std::vector<PlateResult> recognize(const cv::Mat& frame) const;

    const AlprConfig& config() const { return cfg_; }

   private:
    AlprConfig cfg_;
    mutable std::mutex mtx_;
    mutable YoloDetector detector_;  // detect() mutates the net
    PlateOcr ocr_;
};

}  // namespace vision
