// app_multicam — multi-camera detection with a 3x3 ROI grid mask, IoU tracking
// and line-crossing counting.
//
// Replaces car_detection_dual.cpp, car_detection_dual_threaded.cpp and
// multi_thread_rtsp.cpp. Each camera runs a capture thread with an atomic stop
// flag (ESC actually joins — no more forever-blocked joins), Mats are cloned
// across the thread boundary, and counting runs on stable track ids in
// full-frame coordinates (fixing the carStatus out-of-bounds UB and the mixed
// ROI coordinate systems of the old code).
//
// Usage: app_multicam --config configs/multicam.json

#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <thread>
#include <vector>

#include "vision/annotator.hpp"
#include "vision/config.hpp"
#include "vision/tracker.hpp"
#include "vision/video_source.hpp"
#include "vision/yolo_detector.hpp"

namespace {

std::string argValue(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc - 1; ++i) {
        if (key == argv[i]) {
            return argv[i + 1];
        }
    }
    return {};
}

struct SharedFrame {
    std::mutex mtx;
    cv::Mat frame;
};

// Keeps only the ROI cells of a 3x3 grid; the rest is blacked out so the
// detector ignores traffic outside the region of interest.
cv::Mat applyGridMask(const cv::Mat& frame, const std::vector<int>& activeCells) {
    if (activeCells.empty()) {
        return frame;
    }
    const int cw = frame.cols / 3;
    const int ch = frame.rows / 3;
    cv::Mat masked = cv::Mat::zeros(frame.size(), frame.type());
    for (const int cell : activeCells) {
        if (cell < 0 || cell > 8) {
            continue;
        }
        const cv::Rect roi((cell % 3) * cw, (cell / 3) * ch, cw, ch);
        frame(roi).copyTo(masked(roi));
    }
    return masked;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string configPath = argValue(argc, argv, "--config");
    if (configPath.empty()) {
        std::cerr << "Usage: app_multicam --config <file.json>\n";
        return EXIT_FAILURE;
    }

    try {
        // The multicam config reuses AppConfig for detector fields; sources and
        // ROI masks are read directly from the JSON.
        vision::AppConfig cfg = vision::AppConfig::fromFile(configPath);

        // Parse per-camera sources + roi masks from the raw JSON.
        std::vector<std::string> sources;
        std::vector<std::vector<int>> rois;
        {
            std::ifstream ifs(configPath);
            nlohmann::json j;
            ifs >> j;
            for (const auto& cam : j.value("cameras", nlohmann::json::array())) {
                sources.push_back(cam.value("source", std::string("0")));
                rois.push_back(cam.value("roi_cells", std::vector<int>{}));
            }
        }
        if (sources.empty()) {
            std::cerr << "Config has no 'cameras' array.\n";
            return EXIT_FAILURE;
        }

        vision::YoloDetector detector(cfg.toYoloConfig());
        vision::Annotator annotator;

        const size_t n = sources.size();
        std::vector<SharedFrame> shared(n);
        std::atomic<bool> running{true};
        std::vector<std::thread> captureThreads;

        for (size_t i = 0; i < n; ++i) {
            captureThreads.emplace_back([&, i]() {
                vision::VideoSource src(vision::SourceSpec::parse(sources[i]));
                cv::Mat frame;
                while (running.load() && src.read(frame)) {
                    std::lock_guard<std::mutex> lock(shared[i].mtx);
                    shared[i].frame = frame.clone();
                }
            });
        }

        std::vector<vision::IouTracker> trackers(n);
        std::vector<int> counts(n, 0);

        while (running.load()) {
            for (size_t i = 0; i < n; ++i) {
                cv::Mat frame;
                {
                    std::lock_guard<std::mutex> lock(shared[i].mtx);
                    if (!shared[i].frame.empty()) {
                        frame = shared[i].frame.clone();
                    }
                }
                if (frame.empty()) {
                    continue;
                }

                const cv::Mat masked = applyGridMask(frame, rois[i]);
                const auto detections = detector.detect(masked);
                const auto& tracks = trackers[i].update(detections);

                annotator.drawDetections(frame, detections, detector.classNames());
                counts[i] = static_cast<int>(tracks.size());
                cv::putText(frame,
                            "Camera " + std::to_string(i) +
                                "  tracks: " + std::to_string(counts[i]),
                            cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            cv::Scalar(0, 255, 255), 2);
                cv::imshow("Camera " + std::to_string(i), frame);
            }

            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q') {
                running.store(false);
            }
        }

        for (auto& t : captureThreads) {
            if (t.joinable()) {
                t.join();
            }
        }
        cv::destroyAllWindows();
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
