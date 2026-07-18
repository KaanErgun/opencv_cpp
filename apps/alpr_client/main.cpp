// app_alpr_client — capture-side companion to app_alpr_server.
//
// Reads from any source (image/file/webcam/RTSP), sends frames to the ALPR
// server's /recognize endpoint, and overlays the recognised plates. The client
// does NO detection or OCR itself — all the vision work lives on the server, so
// several thin clients can share one recognition service.
//
// Usage:
//   app_alpr_client --image car.png --server http://localhost:8080
//   app_alpr_client --source 0 --server http://host:8080 [--interval 1000]
//   app_alpr_client --source clip.mp4 --headless --max-frames 100 --log out.csv

#include <httplib.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "vision/cli.hpp"
#include "vision/video_source.hpp"

using json = nlohmann::json;

namespace {

// Splits "http://host:8080" into a scheme+host+port base httplib::Client wants.
// httplib::Client accepts the full URL directly in this version.
json recognise(httplib::Client& cli, const cv::Mat& frame) {
    std::vector<uchar> buf;
    cv::imencode(".jpg", frame, buf);
    const std::string body(buf.begin(), buf.end());
    auto res = cli.Post("/recognize", body, "image/jpeg");
    if (!res || res->status != 200) {
        throw std::runtime_error(
            "server request failed" +
            (res ? " (HTTP " + std::to_string(res->status) + ")" : " (no response)"));
    }
    return json::parse(res->body);
}

void drawPlates(cv::Mat& frame, const json& plates) {
    for (const auto& p : plates) {
        const auto box = p.at("box");
        const cv::Rect r(box[0], box[1], box[2], box[3]);
        cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);
        const std::string text = p.value("text", std::string());
        const std::string label =
            text.empty()
                ? "?"
                : text + cv::format(" %.0f%%", p.value("ocr_confidence", 0.0) * 100);
        const int y = std::max(r.y - 8, 16);
        cv::putText(frame, label, cv::Point(r.x, y), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string server =
        vision::cli::argOr(argc, argv, "--server", "http://localhost:8080");
    const std::string image = vision::cli::argValue(argc, argv, "--image");
    const std::string source = vision::cli::argOr(argc, argv, "--source", "0");
    const bool headless = vision::cli::hasFlag(argc, argv, "--headless");
    const int maxFrames = vision::cli::argInt(argc, argv, "--max-frames", 0);
    const int intervalMs = vision::cli::argInt(argc, argv, "--interval", 1000);
    const std::string logPath = vision::cli::argValue(argc, argv, "--log");

    try {
        httplib::Client cli(server);
        cli.set_connection_timeout(5);
        cli.set_read_timeout(10);

        std::ofstream log;
        if (!logPath.empty()) {
            log.open(logPath);
            log << "time,text,ocr_confidence,det_confidence\n";
        }

        // --- Single-image mode -------------------------------------------------
        if (!image.empty()) {
            cv::Mat frame = cv::imread(image);
            if (frame.empty()) {
                std::cerr << "Error: could not read image '" << image << "'\n";
                return EXIT_FAILURE;
            }
            const json resp = recognise(cli, frame);
            const auto& plates = resp.at("plates");
            std::cout << "Recognised " << plates.size() << " plate(s) in "
                      << resp.value("ms", 0) << " ms:\n";
            for (const auto& p : plates) {
                const std::string text = p.value("text", std::string());
                std::cout << "  " << (text.empty() ? "(unreadable)" : text)
                          << "  ocr=" << p.value("ocr_confidence", 0.0)
                          << " det=" << p.value("det_confidence", 0.0) << '\n';
            }
            if (!headless) {
                drawPlates(frame, plates);
                cv::imshow("app_alpr_client", frame);
                cv::waitKey(0);
                cv::destroyAllWindows();
            }
            return EXIT_SUCCESS;
        }

        // --- Streaming mode ----------------------------------------------------
        vision::VideoSource src(vision::SourceSpec::parse(source));
        if (!src.isOpened()) {
            std::cerr << "Error: could not open source '" << source << "'\n";
            return EXIT_FAILURE;
        }
        const std::string window = "app_alpr_client";
        if (!headless) {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
        }

        cv::Mat frame;
        json lastPlates = json::array();
        int frameCount = 0;
        int recognisedTotal = 0;
        auto lastSend =
            std::chrono::steady_clock::now() - std::chrono::milliseconds(intervalMs);

        while (src.read(frame)) {
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSend)
                    .count() >= intervalMs) {
                lastSend = now;
                try {
                    const json resp = recognise(cli, frame);
                    lastPlates = resp.value("plates", json::array());
                    for (const auto& p : lastPlates) {
                        const std::string text = p.value("text", std::string());
                        if (!text.empty()) {
                            ++recognisedTotal;
                            std::cout << "plate: " << text << "  (ocr "
                                      << p.value("ocr_confidence", 0.0) << ")\n";
                            if (log.is_open()) {
                                log << frameCount << ',' << text << ','
                                    << p.value("ocr_confidence", 0.0) << ','
                                    << p.value("det_confidence", 0.0) << '\n';
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "warn: " << e.what() << '\n';
                }
            }

            if (!headless) {
                drawPlates(frame, lastPlates);
                cv::imshow(window, frame);
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') {
                    break;
                }
                if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) {
                    break;
                }
            }
            if (maxFrames > 0 && ++frameCount >= maxFrames) {
                break;
            }
        }

        if (!headless) {
            cv::destroyAllWindows();
        }
        std::cout << "frames=" << frameCount << " recognised=" << recognisedTotal << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
