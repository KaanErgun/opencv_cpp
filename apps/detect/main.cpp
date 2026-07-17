// app_detect — generic single-source detection demo.
//
// Replaces yolov3_car_detection (single), yolov3_cow_detection,
// yolov7_cow_detection, yolov3_human_detection, human_detection_yolo,
// human_detection (HOG), and yolov8_car_plates_detection. Behaviour is driven
// entirely by a JSON config; each old module is now a file in configs/.
//
// Usage: app_detect --config configs/car_webcam.json [--source ...] [--headless]

#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "vision/annotator.hpp"
#include "vision/config.hpp"
#include "vision/hog_detector.hpp"
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

bool hasFlag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (key == argv[i]) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string configPath = argValue(argc, argv, "--config");
    if (configPath.empty()) {
        std::cerr << "Usage: app_detect --config <file.json> [--source S] "
                     "[--headless]\n";
        return EXIT_FAILURE;
    }

    try {
        vision::AppConfig cfg = vision::AppConfig::fromFile(configPath);
        if (const std::string s = argValue(argc, argv, "--source"); !s.empty()) {
            cfg.source = s;
        }
        if (hasFlag(argc, argv, "--headless")) {
            cfg.headless = true;
        }

        std::unique_ptr<vision::IDetector> detector;
        if (cfg.detector == "hog") {
            detector = std::make_unique<vision::HogPeopleDetector>();
        } else {
            detector = std::make_unique<vision::YoloDetector>(cfg.toYoloConfig());
        }

        vision::VideoSource source(vision::SourceSpec::parse(cfg.source));
        if (!source.isOpened()) {
            std::cerr << "Error: could not open source '" << cfg.source << "'\n";
            return EXIT_FAILURE;
        }

        cv::VideoWriter writer;
        vision::Annotator annotator;

        const std::string window = "app_detect";
        if (!cfg.headless) {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
        }

        cv::Mat frame;
        int frameCount = 0;
        auto lastTick = std::chrono::steady_clock::now();
        double fps = 0.0;

        while (source.read(frame)) {
            const auto detections = detector->detect(frame);
            annotator.drawDetections(frame, detections, detector->classNames());

            const auto now = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(now - lastTick).count();
            if (dt > 0) {
                fps = 0.9 * fps + 0.1 * (1.0 / dt);
            }
            lastTick = now;
            annotator.drawFps(frame, fps);

            if (!cfg.outputPath.empty()) {
                if (!writer.isOpened()) {
                    writer.open(cfg.outputPath,
                                cv::VideoWriter::fourcc('m', 'p', '4', 'v'), source.fps(),
                                frame.size());
                }
                if (writer.isOpened()) {
                    writer.write(frame);
                }
            }

            if (!cfg.headless) {
                cv::imshow(window, frame);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q') {  // ESC or q
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
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
