// app_filters — interactive smoothing & morphology playground on live video.
//
// LEARN:
//   * Linear filtering: box blur (plain average) vs Gaussian blur (weighted
//     average) — both smear edges, Gaussian just does it more gracefully.
//   * Non-linear filtering: median blur (great against salt-and-pepper noise)
//     and the edge-preserving bilateral filter (smooths flat regions while
//     keeping strong edges sharp — but it is slow).
//   * Morphology: structuring elements (cv::getStructuringElement), erosion,
//     dilation, and their compositions opening (erode->dilate, removes small
//     bright specks) and closing (dilate->erode, fills small dark holes).
//   * GUI plumbing: trackbars driving parameters live, cv::hconcat for a
//     side-by-side original|filtered comparison view.
//
// Usage:
//   app_filters                                  # webcam 0, GUI with trackbars
//   app_filters --source clip.mp4 --mode median --ksize 7
//   app_filters --headless --mode bilateral --max-frames 60

#include <array>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "vision/cli.hpp"
#include "vision/video_source.hpp"

namespace {

// Mode order matches the "mode" trackbar (0..7) so the slider maps directly.
const std::array<std::string, 8> kModes = {"box",   "gaussian", "median", "bilateral",
                                           "erode", "dilate",   "open",   "close"};

int modeIndex(const std::string& name) {
    for (int i = 0; i < static_cast<int>(kModes.size()); ++i) {
        if (kModes[i] == name) {
            return i;
        }
    }
    return -1;
}

// Kernel sizes for blurs must be odd (a center pixel is needed); clamp to >= 1.
int forceOdd(int k) {
    if (k < 1) {
        k = 1;
    }
    return (k % 2 == 0) ? k + 1 : k;
}

// Apply the selected filter. `k` is already odd and >= 1.
cv::Mat applyFilter(const cv::Mat& src, int mode, int k) {
    cv::Mat dst;
    switch (mode) {
        case 0:  // box: every neighbour weighted equally. Fast but leaves
                 // blocky artifacts; mostly a teaching baseline.
            cv::blur(src, dst, cv::Size(k, k));
            break;
        case 1:  // gaussian: neighbours weighted by distance. The standard
                 // "denoise before edge detection" step (e.g. before Canny),
                 // because it suppresses pixel noise without ringing.
            cv::GaussianBlur(src, dst, cv::Size(k, k), 0);
            break;
        case 2:  // median: replaces each pixel by the neighbourhood median.
                 // Non-linear, so isolated outliers (salt & pepper noise)
                 // vanish entirely instead of being smeared around.
            cv::medianBlur(src, dst, k);
            break;
        case 3:  // bilateral: weights by distance AND colour similarity, so
                 // pixels across a strong edge barely contribute. Result:
                 // "cartoon" smoothing that keeps edges. O(k^2) per pixel —
                 // noticeably slower, which is why we cap the diameter.
            cv::bilateralFilter(src, dst, std::min(k, 15), 75, 75);
            break;
        default: {  // Morphology. We demonstrate it on a grayscale version:
                    // erosion/dilation are min/max filters, and their effect
                    // (shrinking/growing bright regions) is easiest to see on
                    // a single intensity channel.
            cv::Mat gray;
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
            // The structuring element defines the neighbourhood shape; an
            // ellipse gives rounder results than the default rectangle.
            const cv::Mat kernel =
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
            cv::Mat out;
            if (mode == 4) {
                cv::erode(gray, out, kernel);  // shrinks bright blobs
            } else if (mode == 5) {
                cv::dilate(gray, out, kernel);  // grows bright blobs
            } else if (mode == 6) {
                // open = erode then dilate: removes small bright specks while
                // restoring the size of surviving structures.
                cv::morphologyEx(gray, out, cv::MORPH_OPEN, kernel);
            } else {
                // close = dilate then erode: fills small dark holes/gaps.
                cv::morphologyEx(gray, out, cv::MORPH_CLOSE, kernel);
            }
            // Convert back to BGR so hconcat with the colour original works
            // (hconcat requires identical type across inputs).
            cv::cvtColor(out, dst, cv::COLOR_GRAY2BGR);
            break;
        }
    }
    return dst;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace vision::cli;

    const std::string source = argOr(argc, argv, "--source", "0");
    const bool headless = hasFlag(argc, argv, "--headless");
    const int maxFrames = argInt(argc, argv, "--max-frames", 0);
    const std::string modeName = argOr(argc, argv, "--mode", "gaussian");
    int ksize = forceOdd(argInt(argc, argv, "--ksize", 5));

    int mode = modeIndex(modeName);
    if (mode < 0) {
        std::cerr << "Error: unknown --mode '" << modeName
                  << "' (box|gaussian|median|bilateral|erode|dilate|open|close)\n";
        return EXIT_FAILURE;
    }

    try {
        vision::VideoSource src(vision::SourceSpec::parse(source));
        if (!src.isOpened()) {
            std::cerr << "Error: could not open source '" << source << "'\n";
            return EXIT_FAILURE;
        }

        const std::string window = "app_filters";
        if (!headless) {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
            // Trackbars write straight into `mode` / `ksize`; we re-clamp
            // ksize each frame because the slider can land on even values.
            cv::createTrackbar("mode", window, &mode,
                               static_cast<int>(kModes.size()) - 1);
            cv::createTrackbar("ksize", window, &ksize, 31);
        }

        cv::Mat frame;
        long frames = 0;
        while (src.read(frame)) {
            const int k = forceOdd(ksize);
            const cv::Mat filtered = applyFilter(frame, mode, k);
            ++frames;

            if (!headless) {
                // Side-by-side comparison: unfiltered left, filtered right,
                // so the effect of each filter is immediately visible.
                cv::Mat panel;
                cv::hconcat(frame, filtered, panel);
                const std::string label = kModes[mode] + "  k=" + std::to_string(k);
                cv::putText(panel, label, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                            {0, 255, 0}, 2);
                cv::imshow(window, panel);
                // Mask to the low byte: some Linux backends set modifier bits
                // above the low byte. (waitKey returns -1 -> 255, harmless.)
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') {
                    break;
                }
                // On backends whose windows have a close button (GTK/Qt),
                // imshow would otherwise resurrect a closed window forever
                // (macOS Cocoa windows can't be closed).
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
        if (headless) {
            // Machine-checkable summary for smoke tests.
            std::cout << "frames=" << frames << " mode=" << kModes[mode] << "\n";
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
