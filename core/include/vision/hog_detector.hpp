#pragma once

#include <opencv2/core/version.hpp>
// HOGDescriptor moved from objdetect (OpenCV 4.x) to xobjdetect (OpenCV 5.x).
#if CV_VERSION_MAJOR >= 5
#include <opencv2/xobjdetect.hpp>
#else
#include <opencv2/objdetect.hpp>
#endif

#include <string>
#include <vector>

#include "vision/detection.hpp"

namespace vision {

// Classic HOG + default people-detector SVM, exposed through the same IDetector
// interface so the pedestrian demo becomes just another backend.
class HogPeopleDetector : public IDetector {
   public:
    HogPeopleDetector();

    std::vector<Detection> detect(const cv::Mat& frame) override;
    const std::vector<std::string>& classNames() const override { return classNames_; }

   private:
    cv::HOGDescriptor hog_;
    std::vector<std::string> classNames_{"person"};
};

}  // namespace vision
