// app_rtsp_record — view or record any video source (webcam/file/RTSP).
//
// Replaces the "simple rtsp" module (rtsp_stream, rtsp_recorder). VideoWriter
// fps/size are derived from the source (fixing the silent empty-output bug), and
// SIGINT finalises the file cleanly.
//
// Usage:
//   app_rtsp_record --source rtsp://user:pass@host/stream          # view
//   app_rtsp_record --source rtsp://... --output out.mp4 --seconds 60

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#include "vision/video_source.hpp"

namespace {

std::atomic<bool> g_stop{false};
void onSignal(int) { g_stop.store(true); }

std::string argValue(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc - 1; ++i) {
        if (key == argv[i]) {
            return argv[i + 1];
        }
    }
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string source = argValue(argc, argv, "--source");
    if (source.empty()) {
        std::cerr << "Usage: app_rtsp_record --source S [--output out.mp4] "
                     "[--seconds N]\n";
        return EXIT_FAILURE;
    }
    const std::string output = argValue(argc, argv, "--output");
    const std::string secondsStr = argValue(argc, argv, "--seconds");
    const double maxSeconds = secondsStr.empty() ? 0.0 : std::stod(secondsStr);

    std::signal(SIGINT, onSignal);

    try {
        vision::VideoSource src(vision::SourceSpec::parse(source));
        if (!src.isOpened()) {
            std::cerr << "Error: could not open source '" << source << "'\n";
            return EXIT_FAILURE;
        }

        cv::VideoWriter writer;
        const bool recording = !output.empty();
        const std::string window = "app_rtsp_record";
        if (!recording) {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
        }

        cv::Mat frame;
        const auto start = std::chrono::steady_clock::now();
        while (!g_stop.load() && src.read(frame)) {
            if (recording) {
                if (!writer.isOpened()) {
                    writer.open(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                src.fps(), frame.size());
                    if (!writer.isOpened()) {
                        std::cerr << "Error: could not open writer '" << output << "'\n";
                        return EXIT_FAILURE;
                    }
                }
                writer.write(frame);
            } else {
                cv::imshow(window, frame);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q') {
                    break;
                }
            }

            if (maxSeconds > 0) {
                const double elapsed = std::chrono::duration<double>(
                                           std::chrono::steady_clock::now() - start)
                                           .count();
                if (elapsed >= maxSeconds) {
                    break;
                }
            }
        }

        if (!recording) {
            cv::destroyAllWindows();
        }
        std::cout << "Done.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
