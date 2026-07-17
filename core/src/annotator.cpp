#include "vision/annotator.hpp"

#include <opencv2/imgproc.hpp>

namespace vision {

void Annotator::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections,
                               const std::vector<std::string>& classNames) const {
    for (const auto& det : detections) {
        cv::rectangle(frame, det.box, boxColor, thickness);

        std::string name;
        if (det.classId >= 0 && det.classId < static_cast<int>(classNames.size())) {
            name = classNames[det.classId];
        } else {
            name = "id " + std::to_string(det.classId);
        }
        const std::string label =
            name + ": " + cv::format("%.0f%%", det.confidence * 100.0F);

        int baseLine = 0;
        const cv::Size textSize =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        const int top = std::max(det.box.y, textSize.height + 4);
        cv::rectangle(frame, cv::Point(det.box.x, top - textSize.height - 4),
                      cv::Point(det.box.x + textSize.width, top), boxColor, cv::FILLED);
        cv::putText(frame, label, cv::Point(det.box.x, top - 4), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 1);
    }
}

void Annotator::drawFps(cv::Mat& frame, double fps) const {
    const std::string label = cv::format("FPS: %.1f", fps);
    cv::putText(frame, label, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);
}

}  // namespace vision
