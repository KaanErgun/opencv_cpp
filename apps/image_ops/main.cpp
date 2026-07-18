// app_image_ops — image-processing basics: one input image -> a gallery of PNGs.
//
// LEARN:
//   * imread / imwrite            — decoding and encoding images (imgcodecs)
//   * cv::Mat basics              — size, channels, depth, clone vs. shared data
//   * cvtColor                    — BGR -> GRAY and BGR -> HSV colour spaces
//   * resize + interpolation      — why INTER_AREA is the right choice for shrinking
//   * cv::rotate                  — lossless 90-degree rotations
//   * GaussianBlur                — low-pass filtering to suppress pixel noise
//   * Canny                       — gradient-based edge detection with hysteresis
//   * convertTo (alpha/beta)      — linear brightness/contrast adjustment
//   * equalizeHist                — spreading the grayscale histogram for contrast
//
// Usage:
//   app_image_ops --image photo.jpg
//   app_image_ops --image photo.jpg --outdir gallery --show
//
// Every stage is written as NN_<name>.png into --outdir and its path is printed.
// Without --show the app is fully batch/headless; with --show each stage is
// displayed and any key advances (ESC or 'q' quits early, files still saved).

#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "vision/cli.hpp"

using vision::cli::argOr;
using vision::cli::argValue;
using vision::cli::hasFlag;

namespace {

// Human-readable name for a Mat's element depth. depth() describes ONE channel
// (8-bit unsigned, 32-bit float, ...) while channels() says how many there are.
std::string depthName(int depth) {
    switch (depth) {
        case CV_8U:
            return "CV_8U";
        case CV_8S:
            return "CV_8S";
        case CV_16U:
            return "CV_16U";
        case CV_16S:
            return "CV_16S";
        case CV_32S:
            return "CV_32S";
        case CV_32F:
            return "CV_32F";
        case CV_64F:
            return "CV_64F";
        default:
            return "unknown";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string imagePath = argValue(argc, argv, "--image");
        if (imagePath.empty()) {
            std::cerr << "Usage: app_image_ops --image photo.jpg "
                         "[--outdir image_ops_out] [--show]\n";
            return EXIT_FAILURE;
        }
        const std::string outdir = argOr(argc, argv, "--outdir", "image_ops_out");
        const bool show = hasFlag(argc, argv, "--show");

        // imread decodes the file into a Mat. IMREAD_COLOR guarantees a 3-channel
        // BGR image (OpenCV's native channel order — NOT RGB!) regardless of
        // whether the file was grayscale or had an alpha channel.
        const cv::Mat original = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (original.empty()) {
            std::cerr << "Fatal: could not read image '" << imagePath << "'\n";
            return EXIT_FAILURE;
        }

        // Basic Mat properties: geometry, channel count, and per-channel depth.
        std::cout << "input " << imagePath << " " << original.cols << "x" << original.rows
                  << " channels=" << original.channels()
                  << " depth=" << depthName(original.depth()) << "\n";

        std::filesystem::create_directories(outdir);

        int savedCount = 0;
        bool keepShowing = show;
        // Small helper: write the stage to disk, print it, optionally display it.
        auto saveStage = [&](const std::string& name, const cv::Mat& img) {
            const std::string path = outdir + "/" + name + ".png";
            if (!cv::imwrite(path, img)) {
                throw std::runtime_error("could not write " + path);
            }
            ++savedCount;
            std::cout << "saved " << path << "\n";
            if (keepShowing) {
                cv::imshow("app_image_ops", img);
                const int key = cv::waitKey(0);  // any key advances to next stage
                if (key == 27 || key == 'q') {
                    keepShowing = false;  // stop displaying, keep saving files
                }
            }
        };

        // 01 — the untouched input, re-encoded as PNG for a fair visual baseline.
        saveStage("01_original", original);

        // 02 — grayscale. Many algorithms (edges, features, histograms) only need
        // intensity, so we collapse 3 channels into 1 with a weighted sum.
        cv::Mat gray;
        cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
        saveStage("02_grayscale", gray);

        // 03 — half-size resize. INTER_AREA averages the source pixels that map
        // onto each destination pixel, which avoids the aliasing/moire artifacts
        // that INTER_LINEAR or INTER_NEAREST produce when SHRINKING an image.
        // (For enlarging you would prefer INTER_LINEAR or INTER_CUBIC instead.)
        cv::Mat half;
        cv::resize(original, half, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        saveStage("03_half_size", half);

        // 04 — rotate 90 degrees clockwise. cv::rotate just remaps pixels, so
        // unlike warpAffine there is no interpolation and no quality loss.
        cv::Mat rotated;
        cv::rotate(original, rotated, cv::ROTATE_90_CLOCKWISE);
        saveStage("04_rotate90_cw", rotated);

        // 05 — Gaussian blur with a 9x9 kernel. A Gaussian is a low-pass filter:
        // it removes fine high-frequency detail (sensor noise) while preserving
        // the large structures we usually care about.
        cv::Mat blurred;
        cv::GaussianBlur(original, blurred, cv::Size(9, 9), 0);
        saveStage("05_gaussian_blur", blurred);

        // 06 — Canny edges on a BLURRED grayscale image. Canny differentiates the
        // image, and differentiation amplifies noise — without the blur, random
        // pixel noise would show up as thousands of tiny false edges. Thresholds
        // 50/150 follow the recommended ~1:3 low:high hysteresis ratio.
        cv::Mat blurredGray, edges;
        cv::GaussianBlur(gray, blurredGray, cv::Size(9, 9), 0);
        cv::Canny(blurredGray, edges, 50.0, 150.0);
        saveStage("06_canny_edges", edges);

        // 07 — HSV hue channel. HSV separates colour (hue) from brightness, which
        // makes colour-based segmentation robust to lighting changes. Shown as a
        // grayscale image: each pixel's brightness encodes its hue angle (0-179).
        cv::Mat hsv;
        cv::cvtColor(original, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvChannels;
        cv::split(hsv, hsvChannels);
        saveStage("07_hue_channel", hsvChannels[0]);

        // 08 — brightness/contrast via convertTo: out = alpha * in + beta.
        // alpha > 1 stretches contrast, beta > 0 lifts brightness. Results are
        // saturated (clamped) to [0,255] automatically for 8-bit images.
        cv::Mat brighter;
        original.convertTo(brighter, -1, 1.3, 20);  // -1 keeps the same depth
        saveStage("08_bright_contrast", brighter);

        // 09 — histogram equalization. Spreads the grayscale intensity histogram
        // over the full [0,255] range, revealing detail in murky low-contrast
        // images. Works on single-channel images only, hence the gray input.
        cv::Mat equalized;
        cv::equalizeHist(gray, equalized);
        saveStage("09_equalized_gray", equalized);

        if (show) {
            cv::destroyAllWindows();
        }
        std::cout << "saved=" << savedCount << " dir=" << outdir << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
