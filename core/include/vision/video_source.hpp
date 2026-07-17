#pragma once

#include <opencv2/videoio.hpp>
#include <string>

namespace vision {

// Describes an input: a webcam index ("0"), a file path ("clip.mp4"), or an
// "rtsp://" URL. Parsed from a single string so apps take one --source flag.
struct SourceSpec {
    enum class Kind { Webcam, File, Rtsp };
    Kind kind = Kind::Webcam;
    int cameraIndex = 0;
    std::string path;  // file path or rtsp url

    static SourceSpec parse(const std::string& spec);
    bool isLive() const { return kind == Kind::Webcam || kind == Kind::Rtsp; }
};

// Thin VideoCapture wrapper. For live sources a single dropped frame no longer
// kills the stream: read() transparently reopens with backoff. FFmpeg open/read
// timeouts are set for RTSP so a dead camera fails fast instead of blocking.
class VideoSource {
   public:
    explicit VideoSource(const SourceSpec& spec);

    // Reads the next frame. Returns false only when a non-live source is
    // exhausted (end of file). For live sources it keeps trying to reconnect.
    bool read(cv::Mat& frame);

    double fps() const;
    cv::Size size() const;
    bool isLive() const { return spec_.isLive(); }
    bool isOpened() const { return cap_.isOpened(); }

   private:
    bool open();

    SourceSpec spec_;
    cv::VideoCapture cap_;
    int reconnectDelayMs_ = 500;
    int maxReconnectDelayMs_ = 5000;
};

}  // namespace vision
