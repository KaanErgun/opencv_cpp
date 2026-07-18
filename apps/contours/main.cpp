// app_contours — from a binary mask to shape measurements.
//
// LEARN:
//   * Global binary threshold (cv::threshold) vs adaptive threshold
//     (cv::adaptiveThreshold) and when each one wins.
//   * findContours retrieval modes (RETR_EXTERNAL) and point-compression
//     modes (CHAIN_APPROX_SIMPLE).
//   * Filtering noise contours by contourArea.
//   * Image moments: computing a centroid from m10/m00 and m01/m00.
//   * boundingRect for a quick axis-aligned box around a shape.
//
// Usage:
//   app_contours                                  # webcam 0, GUI with trackbar
//   app_contours --source clip.mp4 --thresh 100   # video file, custom threshold
//   app_contours --source 0 --headless --max-frames 90 --min-area 300
//   app_contours --invert                          # dark parts on a lightbox
//
// GUI keys: ESC or 'q' quits, 'a' toggles global vs adaptive threshold,
// 'i' toggles threshold polarity (normal vs inverted).

#include <iostream>
// OpenCV 5 note: contourArea/boundingRect/moments moved from imgproc into the
// new `geometry` module, so we include it (and link opencv_geometry) as well.
#include <opencv2/geometry.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "vision/cli.hpp"
#include "vision/video_source.hpp"

using vision::cli::argInt;
using vision::cli::argOr;
using vision::cli::hasFlag;

int main(int argc, char** argv) {
    try {
        const std::string source = argOr(argc, argv, "--source", "0");
        const bool headless = hasFlag(argc, argv, "--headless");
        const int maxFrames = argInt(argc, argv, "--max-frames", 0);  // 0 = unbounded
        int thresh = argInt(argc, argv, "--thresh", 128);
        const int minArea = argInt(argc, argv, "--min-area", 500);
        // --invert starts with THRESH_BINARY_INV (dark objects on a bright
        // background); the 'i' key flips it at runtime in the GUI.
        bool invert = hasFlag(argc, argv, "--invert");

        vision::VideoSource src(vision::SourceSpec::parse(source));
        if (!src.isOpened()) {
            std::cerr << "Error: could not open source '" << source << "'\n";
            return EXIT_FAILURE;
        }

        const std::string window = "app_contours";
        if (!headless) {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
            // Trackbar edits `thresh` in place; we read it every frame so the
            // effect is immediate. (Only meaningful for the global threshold.)
            cv::createTrackbar("thresh", window, &thresh, 255);
        }

        bool useAdaptive = false;  // 'a' key flips between global and adaptive
        long frames = 0;
        long contoursTotal = 0;

        cv::Mat frame, gray, blurred, mask;
        while (src.read(frame)) {
            ++frames;

            // Contours are defined on a single-channel binary image, so we
            // first collapse the colour frame to grayscale.
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // Blur before thresholding: sensor noise creates lone bright/dark
            // pixels that would each become a tiny (junk) contour. A small
            // Gaussian smooths them away, giving far cleaner blob outlines.
            cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

            // Polarity matters: plain THRESH_BINARY marks BRIGHT pixels as
            // foreground, so it suits bright objects on a dark background.
            // For dark parts on a bright lightbox use THRESH_BINARY_INV —
            // otherwise the bright background becomes one huge foreground
            // blob and RETR_EXTERNAL returns a single frame-sized contour.
            const int threshType = invert ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
            if (useAdaptive) {
                // Adaptive threshold picks a *local* cutoff per pixel from the
                // mean of its 11x11 neighbourhood (minus C=2). This survives
                // uneven lighting (shadows, vignetting) where one global value
                // cannot separate foreground everywhere at once.
                cv::adaptiveThreshold(blurred, mask, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                                      threshType, 11, 2);
            } else {
                // Global threshold: one cutoff for the whole image. Simple and
                // fast, ideal when lighting is even.
                cv::threshold(blurred, mask, thresh, 255, threshType);
            }

            // RETR_EXTERNAL keeps only the outermost contours (no holes /
            // nesting hierarchy) — right for counting distinct objects.
            // CHAIN_APPROX_SIMPLE compresses straight runs of boundary pixels
            // into their endpoints, storing far fewer points per contour.
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            int kept = 0;
            for (const auto& contour : contours) {
                const double area = cv::contourArea(contour);
                // Area filter: thresholding always leaves speckles; anything
                // smaller than --min-area px^2 is treated as noise and skipped.
                if (area < minArea) {
                    continue;
                }
                ++kept;

                if (!headless) {
                    // Outline the shape itself (index -1 would draw all; here
                    // we draw one at a time so each can get its own labels).
                    cv::drawContours(frame, std::vector<std::vector<cv::Point>>{contour},
                                     -1, cv::Scalar(0, 255, 0), 2);

                    // Axis-aligned bounding box: the cheapest "where is it".
                    const cv::Rect box = cv::boundingRect(contour);
                    cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 1);

                    // Moments give us the centroid. m00 is the area (sum of
                    // pixels), m10 the sum of x coordinates and m01 the sum of
                    // y coordinates — so cx = m10/m00, cy = m01/m00 is the
                    // average position of the shape's mass, i.e. its centroid.
                    const cv::Moments m = cv::moments(contour);
                    if (m.m00 > 0) {
                        const cv::Point centroid(static_cast<int>(m.m10 / m.m00),
                                                 static_cast<int>(m.m01 / m.m00));
                        cv::circle(frame, centroid, 4, cv::Scalar(0, 0, 255), -1);
                    }

                    // Label the shape with its area, just above the box.
                    cv::putText(frame, "area=" + std::to_string(static_cast<int>(area)),
                                {box.x, std::max(box.y - 6, 12)},
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255),
                                1);
                }
            }
            contoursTotal += kept;

            if (!headless) {
                // HUD: which threshold mode is active plus the kept count, so
                // the effect of 'a' and the trackbar is visible at a glance.
                const std::string hud = std::string(useAdaptive ? "adaptive" : "global") +
                                        " thresh | " + (invert ? "inverted" : "normal") +
                                        " | contours=" + std::to_string(kept);
                cv::putText(frame, hud, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            cv::Scalar(0, 255, 0), 2);
                cv::imshow(window, frame);

                // Mask to the low byte: some Linux backends set modifier bits
                // above it. (waitKey returns -1 -> 255, harmless.)
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') {
                    break;
                }
                if (key == 'a') {
                    useAdaptive = !useAdaptive;
                }
                if (key == 'i') {
                    invert = !invert;
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
        // Machine-checkable summary for smoke tests.
        std::cout << "frames=" << frames << " contours_total=" << contoursTotal << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
