#pragma once

#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "vision/detection.hpp"

namespace vision {

// Centralised drawing so every app renders boxes/labels/overlays identically.
// Always draw AFTER inference (or on a copy) so overlays never feed the network.
class Annotator {
   public:
    // Draws each detection's box + "name: conf%" label. classNames may be empty,
    // in which case the numeric class id is shown.
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections,
                        const std::vector<std::string>& classNames) const;

    void drawFps(cv::Mat& frame, double fps) const;

    cv::Scalar boxColor = cv::Scalar(0, 255, 0);
    int thickness = 2;
};

}  // namespace vision
