// app_qr_scanner — QR codes both directions: encode text to a PNG, decode from a
// still image or a live video source.
//
// LEARN:
//   * cv::QRCodeEncoder  — turning a text payload into a QR module matrix.
//   * cv::QRCodeDetector — the two-stage pipeline: DETECT (find the code's four
//     corners in the image) vs DECODE (rectify the quad and read the modules).
//   * Perspective quads  — the detector returns 4 corner points, not a bounding
//     box, because a QR code seen at an angle is a general quadrilateral.
//   * cv::INTER_NEAREST  — the only sane interpolation for upscaling binary
//     module grids (any smoothing filter would blur module edges to grey).
//
// Usage:
//   app_qr_scanner --encode "hello world" --out qr.png     # write a QR PNG
//   app_qr_scanner --image qr.png                          # decode one still
//   app_qr_scanner --source 0                              # live webcam decode
//   app_qr_scanner --source clip.mp4 --headless --max-frames 100

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

#include "vision/cli.hpp"
#include "vision/video_source.hpp"

using vision::cli::argInt;
using vision::cli::argOr;
using vision::cli::argValue;
using vision::cli::hasFlag;

namespace {

// Runs detect+decode on one frame and draws the result in place.
// Returns the decoded payload ("" when nothing was found or decode failed).
std::string decodeAndAnnotate(cv::QRCodeDetector& detector, cv::Mat& frame,
                              bool annotate) {
    // detectAndDecode does both stages: first it LOCATES the code (using the
    // three big finder squares), then it warps the quad flat and DECODES the
    // module grid. `points` receives the 4 corners of the located code.
    std::vector<cv::Point2f> points;
    std::string text = detector.detectAndDecode(frame, points);

    if (points.size() == 4 && annotate) {
        // The four points capture PERSPECTIVE: a QR code held at an angle
        // projects to a general quadrilateral, so we draw a closed polygon
        // through the corners instead of an axis-aligned rectangle.
        std::vector<cv::Point> quad;
        for (const auto& p : points) {
            quad.emplace_back(cvRound(p.x), cvRound(p.y));
        }
        // Green when the payload decoded, red when the code was DETECTED but
        // the decode came back empty — typically the code is too small in the
        // frame or motion-blurred, so the module grid can't be read reliably.
        const cv::Scalar color =
            text.empty() ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::polylines(frame, quad, /*isClosed=*/true, color, 2);
        const std::string label = text.empty() ? "(detected, decode failed)" : text;
        cv::putText(frame, label, quad[0] + cv::Point(0, -10), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2);
    }
    return text;
}

// Mode 1: encode `text` into a QR image and save it as a PNG.
int runEncode(const std::string& text, const std::string& outPath) {
    // The encoder emits a tiny CV_8UC1 matrix with ONE PIXEL PER MODULE
    // (e.g. 21x21 for the smallest version), values 0/255.
    cv::Ptr<cv::QRCodeEncoder> encoder = cv::QRCodeEncoder::create();
    cv::Mat modules;
    encoder->encode(text, modules);
    if (modules.empty()) {
        std::cerr << "Error: encoding failed (payload too long?)\n";
        return EXIT_FAILURE;
    }

    // Upscale to ~400px with INTER_NEAREST: nearest-neighbour just replicates
    // pixels, so each module stays a crisp black/white square. Linear or cubic
    // interpolation would produce grey gradients at module borders, which both
    // looks wrong and hurts scannability.
    const int scale = std::max(1, 400 / modules.cols);
    cv::Mat big;
    cv::resize(modules, big, {}, scale, scale, cv::INTER_NEAREST);

    // A quiet zone (white border) around the code is part of the QR spec and
    // helps real-world scanners find the finder patterns.
    cv::copyMakeBorder(big, big, 4 * scale, 4 * scale, 4 * scale, 4 * scale,
                       cv::BORDER_CONSTANT, cv::Scalar(255));

    if (!cv::imwrite(outPath, big)) {
        std::cerr << "Error: could not write '" << outPath << "'\n";
        return EXIT_FAILURE;
    }
    std::cout << "encoded=" << outPath << "\n";
    return EXIT_SUCCESS;
}

// Mode 2: decode a single still image.
int runImage(const std::string& imagePath, bool headless) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: could not read image '" << imagePath << "'\n";
        return EXIT_FAILURE;
    }
    cv::QRCodeDetector detector;
    const std::string text = decodeAndAnnotate(detector, image, /*annotate=*/true);
    std::cout << "decoded=" << text << "\n";
    if (!headless) {
        cv::imshow("app_qr_scanner", image);
        cv::waitKey(0);  // any key closes a still image
        cv::destroyAllWindows();
    }
    return EXIT_SUCCESS;
}

// Mode 3 (default): live decode from a video source.
int runVideo(const std::string& source, bool headless, int maxFrames) {
    vision::VideoSource src(vision::SourceSpec::parse(source));
    if (!src.isOpened()) {
        std::cerr << "Error: could not open source '" << source << "'\n";
        return EXIT_FAILURE;
    }

    const std::string window = "app_qr_scanner";
    if (!headless) {
        cv::namedWindow(window, cv::WINDOW_NORMAL);
    }

    cv::QRCodeDetector detector;
    cv::Mat frame;
    std::string lastPayload;  // suppress consecutive duplicate prints
    // A code at marginal distance (or under motion blur) flickers between
    // decode success and failure frame-to-frame. Clearing lastPayload on the
    // FIRST empty frame would reprint the payload at ~15 Hz for such a code,
    // so we only forget it after a run of consecutive misses (~0.5 s at 30
    // fps) — long enough that a genuinely removed-and-reshown code still
    // reprints.
    constexpr int kMissLimit = 15;
    int missStreak = 0;
    int frames = 0;
    int decoded = 0;

    while (src.read(frame)) {
        ++frames;
        const std::string text = decodeAndAnnotate(detector, frame, !headless);
        if (!text.empty()) {
            ++decoded;
            missStreak = 0;
            // Print each NEW payload once. A code held in front of the camera
            // decodes every frame; without this check we would spam stdout
            // with the same string 30 times a second.
            if (text != lastPayload) {
                std::cout << text << "\n";
                lastPayload = text;
            }
        } else if (++missStreak >= kMissLimit) {
            lastPayload.clear();  // allow re-print if the same code reappears
        }

        if (!headless) {
            cv::imshow(window, frame);
            // Mask to the low byte: some Linux backends set modifier bits
            // above it. (waitKey returns -1 -> 255, harmless.)
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') {
                break;
            }
            // On backends whose windows have a close button (GTK/Qt), imshow
            // would otherwise resurrect a closed window forever (macOS Cocoa
            // windows can't be closed).
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
        std::cout << "frames=" << frames << " decoded_frames=" << decoded << "\n";
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool headless = hasFlag(argc, argv, "--headless");
        const int maxFrames = argInt(argc, argv, "--max-frames", 0);

        const std::string encodeText = argValue(argc, argv, "--encode");
        if (!encodeText.empty()) {
            return runEncode(encodeText, argOr(argc, argv, "--out", "qr.png"));
        }

        const std::string imagePath = argValue(argc, argv, "--image");
        if (!imagePath.empty()) {
            return runImage(imagePath, headless);
        }

        return runVideo(argOr(argc, argv, "--source", "0"), headless, maxFrames);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
