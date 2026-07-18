// app_alpr — license-plate detection with a YOLO ONNX model.
//
// Replaces alpr_plate_detection (OpenALPR + Haar + residents.net.au uploader)
// and yolov3_plate_recognition. OpenALPR and the web uploader are retired (see
// docs/DECISIONS.md K4/K5). This app detects plates with best.onnx and saves a
// clean crop (taken from the pre-overlay frame, clamped to bounds) for each
// plate — no temp-file JPEG round-trips, no embedded credentials.
//
// Usage: app_alpr --config configs/alpr.json [--source S] [--save-dir DIR]

#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "vision/annotator.hpp"
#include "vision/cli.hpp"
#include "vision/config.hpp"
#include "vision/video_source.hpp"
#include "vision/yolo_detector.hpp"

using vision::cli::argValue;

int main(int argc, char** argv) {
    const std::string configPath = argValue(argc, argv, "--config");
    if (configPath.empty()) {
        std::cerr << "Usage: app_alpr --config <file.json> [--source S] "
                     "[--save-dir DIR]\n";
        return EXIT_FAILURE;
    }

    try {
        vision::AppConfig cfg = vision::AppConfig::fromFile(configPath);
        if (const std::string s = argValue(argc, argv, "--source"); !s.empty()) {
            cfg.source = s;
        }
        const std::string saveDir = argValue(argc, argv, "--save-dir");
        if (!saveDir.empty()) {
            std::filesystem::create_directories(saveDir);
        }

        // The plate class id in best.onnx (classes: {"Araba"=0, "Plaka"=1}).
        const int plateClassId = cfg.classFilter.empty() ? 1 : cfg.classFilter.front();

        vision::YoloDetector detector(cfg.toYoloConfig());
        vision::Annotator annotator;
        vision::VideoSource source(vision::SourceSpec::parse(cfg.source));
        if (!source.isOpened()) {
            std::cerr << "Error: could not open source '" << cfg.source << "'\n";
            return EXIT_FAILURE;
        }

        const std::string window = "app_alpr";
        if (!cfg.headless) {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
        }

        cv::Mat frame;
        int savedCount = 0;
        int frameCount = 0;
        while (source.read(frame)) {
            const cv::Mat clean = frame.clone();  // crop source, before overlays
            const auto detections = detector.detect(frame);

            for (const auto& det : detections) {
                if (det.classId == plateClassId && !saveDir.empty()) {
                    const cv::Rect safe =
                        det.box & cv::Rect(0, 0, clean.cols, clean.rows);
                    if (safe.area() > 0) {
                        const std::string path =
                            saveDir + "/plate_" + std::to_string(savedCount++) + ".png";
                        cv::imwrite(path, clean(safe));
                    }
                }
            }

            annotator.drawDetections(frame, detections, detector.classNames());
            if (!cfg.headless) {
                cv::imshow(window, frame);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q') {
                    break;
                }
            }
            if (cfg.maxFrames > 0 && ++frameCount >= cfg.maxFrames) {
                break;
            }
        }

        if (!cfg.headless) {
            cv::destroyAllWindows();
        }
        std::cout << "Saved " << savedCount << " plate crops.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
