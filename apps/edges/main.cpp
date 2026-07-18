// app_edges — gradients and edge detection.
//
// LEARN:
//   * Sobel gradients      — first derivatives of intensity in x and y; large
//                            values mean the brightness changes quickly (an edge).
//   * Gradient magnitude   — sqrt(gx^2 + gy^2) combines both directions into a
//                            single "edge strength" image.
//   * Canny hysteresis     — two thresholds: pixels above `high` are definite
//                            edges, pixels between `low` and `high` are kept only
//                            if connected to a definite edge. This keeps weak but
//                            genuine edge continuations while rejecting isolated
//                            noise responses.
//   * Threshold tuning     — rule of thumb: keep low:high in the 1:2 .. 1:3 range
//                            (e.g. 50/150). Play with the trackbars to see why.
//
// Usage:
//   app_edges                                  # webcam 0, GUI with trackbars
//   app_edges --source clip.mp4 --low 40 --high 120
//   app_edges --source 0 --headless --max-frames 90

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "vision/cli.hpp"
#include "vision/video_source.hpp"

using vision::cli::argInt;
using vision::cli::argOr;
using vision::cli::hasFlag;

namespace {

enum class ViewMode { Original, Canny, SobelMag };

const char* modeName(ViewMode m) {
    switch (m) {
        case ViewMode::Original:
            return "original";
        case ViewMode::Canny:
            return "canny";
        default:
            return "sobel magnitude";
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string source = argOr(argc, argv, "--source", "0");
    const bool headless = hasFlag(argc, argv, "--headless");
    const int maxFrames = argInt(argc, argv, "--max-frames", 0);  // 0 = unbounded
    int low = argInt(argc, argv, "--low", 50);
    int high = argInt(argc, argv, "--high", 150);

    try {
        vision::VideoSource src(vision::SourceSpec::parse(source));
        if (!src.isOpened()) {
            std::cerr << "Error: could not open source '" << source << "'\n";
            return EXIT_FAILURE;
        }

        const std::string window = "app_edges";
        if (!headless) {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
            // Trackbars drive the ints directly; we read them each frame so the
            // effect is instant. Range 0..500 covers gradients well beyond 8-bit
            // steps (Canny compares thresholds against Sobel-derived magnitudes).
            cv::createTrackbar("low", window, &low, 500);
            cv::createTrackbar("high", window, &high, 500);
        }

        ViewMode mode = ViewMode::Canny;
        cv::Mat frame, gray, blurred, edges, gx, gy, mag, magU8, display;
        long frames = 0;
        double edgeRatioSum = 0.0;  // running sum of per-frame edge fractions

        while (src.read(frame)) {
            ++frames;

            // Edge detectors are derivative filters, and derivatives amplify
            // noise: a single noisy pixel produces a strong local gradient and
            // therefore a false edge. A small Gaussian blur suppresses that
            // pixel-level noise while keeping real structure, so we ALWAYS blur
            // before differentiating. 5x5 is a good default for video.
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

            // Keep thresholds sane even if the user drags "low" above "high":
            // Canny requires low <= high (it swaps internally in some versions,
            // but being explicit teaches the intent).
            const int lo = std::min(low, high);
            const int hi = std::max(low, high);

            // Canny: Sobel gradients -> non-maximum suppression (thin the edges
            // to 1 px) -> hysteresis with (lo, hi) as explained in LEARN above.
            cv::Canny(blurred, edges, lo, hi);
            edgeRatioSum += static_cast<double>(cv::countNonZero(edges)) / edges.total();

            if (!headless) {
                if (mode == ViewMode::SobelMag) {
                    // Raw Sobel output can be negative (dark->light vs
                    // light->dark), so we compute in CV_32F, take the magnitude,
                    // then normalize to 0..255 for display as CV_8U.
                    cv::Sobel(blurred, gx, CV_32F, 1, 0);
                    cv::Sobel(blurred, gy, CV_32F, 0, 1);
                    cv::magnitude(gx, gy, mag);
                    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
                    mag.convertTo(magU8, CV_8U);
                    cv::cvtColor(magU8, display, cv::COLOR_GRAY2BGR);
                } else if (mode == ViewMode::Canny) {
                    cv::cvtColor(edges, display, cv::COLOR_GRAY2BGR);
                } else {
                    display = frame.clone();
                }

                // Overlay so the learner always knows what they are looking at.
                const std::string label =
                    std::string(modeName(mode)) + "  low=" + std::to_string(lo) +
                    " high=" + std::to_string(hi) + "  ('m' = mode, ESC/q = quit)";
                cv::putText(display, label, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                            0.6, cv::Scalar(0, 255, 0), 2);
                cv::imshow(window, display);

                // Mask to the low byte: some Linux backends set modifier bits
                // above it (no key -> -1 becomes 255, which matches nothing).
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') {
                    break;
                }
                if (key == 'm') {  // cycle original -> canny -> sobel -> ...
                    mode = (mode == ViewMode::Original) ? ViewMode::Canny
                           : (mode == ViewMode::Canny)  ? ViewMode::SobelMag
                                                        : ViewMode::Original;
                }
                // On backends whose windows have a close button (GTK/Qt),
                // imshow would otherwise resurrect a closed window forever
                // (macOS Cocoa windows can't be closed by the user).
                if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) {
                    break;
                }
            }

            if (maxFrames > 0 && frames >= maxFrames) {
                break;
            }
        }

        if (!headless) {
            cv::destroyAllWindows();
        }

        const double meanRatio = frames > 0 ? edgeRatioSum / frames : 0.0;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "frames=%ld edge_ratio=%.3f", frames, meanRatio);
        std::cout << buf << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
