#include "vision/video_source.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>

namespace vision {

SourceSpec SourceSpec::parse(const std::string& spec) {
    SourceSpec out;
    if (spec.rfind("rtsp://", 0) == 0 || spec.rfind("http://", 0) == 0 ||
        spec.rfind("https://", 0) == 0) {
        out.kind = Kind::Rtsp;
        out.path = spec;
        return out;
    }

    const bool allDigits =
        !spec.empty() && std::all_of(spec.begin(), spec.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        });
    if (allDigits) {
        out.kind = Kind::Webcam;
        out.cameraIndex = std::stoi(spec);
        return out;
    }

    out.kind = Kind::File;
    out.path = spec;
    return out;
}

VideoSource::VideoSource(const SourceSpec& spec) : spec_(spec) { open(); }

bool VideoSource::open() {
    if (spec_.kind == SourceSpec::Kind::Webcam) {
        cap_.open(spec_.cameraIndex);
    } else if (spec_.kind == SourceSpec::Kind::Rtsp) {
        cap_.open(spec_.path, cv::CAP_FFMPEG);
        cap_.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000);
        cap_.set(cv::CAP_PROP_READ_TIMEOUT_MSEC, 5000);
    } else {
        cap_.open(spec_.path);
    }
    return cap_.isOpened();
}

bool VideoSource::read(cv::Mat& frame) {
    if (cap_.isOpened() && cap_.read(frame) && !frame.empty()) {
        return true;
    }

    // Non-live source that failed to read is simply exhausted.
    if (!spec_.isLive()) {
        return false;
    }

    // Live source: reconnect with exponential backoff, forever.
    int delay = reconnectDelayMs_;
    while (true) {
        cap_.release();
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        if (open() && cap_.read(frame) && !frame.empty()) {
            return true;
        }
        delay = std::min(delay * 2, maxReconnectDelayMs_);
    }
}

double VideoSource::fps() const {
    const double f = cap_.get(cv::CAP_PROP_FPS);
    return (f > 1.0 && f < 240.0) ? f : 30.0;
}

cv::Size VideoSource::size() const {
    return {static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT))};
}

}  // namespace vision
