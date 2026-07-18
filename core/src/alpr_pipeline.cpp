#include "vision/alpr_pipeline.hpp"

namespace vision {

AlprPipeline::AlprPipeline(const AlprConfig& cfg)
    : cfg_(cfg), detector_(cfg.detector), ocr_(cfg.ocr) {}

std::vector<PlateResult> AlprPipeline::recognize(const cv::Mat& frame) const {
    std::lock_guard<std::mutex> lock(mtx_);

    std::vector<PlateResult> results;
    if (frame.empty()) {
        return results;
    }

    const cv::Rect frameRect(0, 0, frame.cols, frame.rows);
    for (const auto& det : detector_.detect(frame)) {
        if (det.classId != cfg_.plateClassId) {
            continue;
        }

        // Grow the detected box a little before cropping — detectors clip plate
        // edges, and a few extra pixels of quiet zone help the OCR.
        const int padX = static_cast<int>(det.box.width * cfg_.padding);
        const int padY = static_cast<int>(det.box.height * cfg_.padding);
        const cv::Rect padded =
            cv::Rect(det.box.x - padX, det.box.y - padY, det.box.width + 2 * padX,
                     det.box.height + 2 * padY) &
            frameRect;
        if (padded.area() <= 0) {
            continue;
        }

        PlateResult result;
        result.box = det.box;
        result.detConfidence = det.confidence;

        const OcrResult ocr = ocr_.read(frame(padded));
        if (!ocr.text.empty() && ocr.confidence >= cfg_.minOcrConfidence) {
            result.text = ocr.text;
            result.ocrConfidence = ocr.confidence;
        }
        results.push_back(result);
    }
    return results;
}

}  // namespace vision
