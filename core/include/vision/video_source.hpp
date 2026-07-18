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

    // Reads the next frame. Returns false when a non-live source is exhausted
    // (end of file), or when a live source stayed dead for the whole reconnect
    // budget (~30 s of backed-off retries) — retrying forever would freeze the
    // caller, which is typically a GUI thread.
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
    int reconnectBudgetMs_ = 30000;
};

}  // namespace vision
