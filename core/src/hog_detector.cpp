#include "vision/hog_detector.hpp"

namespace vision {

HogPeopleDetector::HogPeopleDetector() {
    hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
}

std::vector<Detection> HogPeopleDetector::detect(const cv::Mat& frame) {
    if (frame.empty()) {
        return {};
    }

    std::vector<cv::Rect> found;
    std::vector<double> weights;
    hog_.detectMultiScale(frame, found, weights, 0.0, cv::Size(8, 8), cv::Size(0, 0),
                          1.05, 2.0, false);

    const cv::Rect frameRect(0, 0, frame.cols, frame.rows);
    std::vector<Detection> detections;
    detections.reserve(found.size());
    for (size_t i = 0; i < found.size(); ++i) {
        Detection det;
        det.box = found[i] & frameRect;
        det.classId = 0;
        det.confidence = static_cast<float>(weights[i]);
        if (det.box.area() > 0) {
            detections.push_back(det);
        }
    }
    return detections;
}

}  // namespace vision
