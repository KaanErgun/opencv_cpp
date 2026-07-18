// app_color_track — track an object by its colour in HSV space.
//
// LEARN:
//   * HSV colour space: why hue/saturation/value beats BGR for colour matching.
//   * cv::inRange — turning a colour band into a binary segmentation mask.
//   * Morphological cleanup (MORPH_OPEN + dilate) to remove speckle noise.
//   * cv::findContours + picking the largest blob as the tracked object.
//   * Image moments for the centroid, and drawing a motion trail (polyline).
//
// Usage:
//   app_color_track                              # webcam 0, track green things
//   app_color_track --source clip.mp4 --hmin 100 --hmax 130   # track blue
//   app_color_track --source clip.mp4 --headless --max-frames 90

#include <deque>
#include <iostream>
// OpenCV 5 split shape analysis (contourArea, boundingRect, moments) into the
// new `geometry` module; segmentation (inRange/findContours) stays in imgproc.
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
        const int maxFrames = argInt(argc, argv, "--max-frames", 0);

        // Default band 35..85 covers typical greens. OpenCV stores hue as
        // 0..179 (half the usual 0..359) so it fits in 8 bits.
        int hmin = argInt(argc, argv, "--hmin", 35);
        int hmax = argInt(argc, argv, "--hmax", 85);
        int smin = argInt(argc, argv, "--smin", 60);
        int vmin = argInt(argc, argv, "--vmin", 60);
        const int minArea = argInt(argc, argv, "--min-area", 400);

        vision::VideoSource src(vision::SourceSpec::parse(source));
        if (!src.isOpened()) {
            std::cerr << "Error: could not open source '" << source << "'\n";
            return EXIT_FAILURE;
        }

        const std::string window = "app_color_track";
        if (!headless) {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
            // Trackbars write straight into our ints, so the segmentation
            // band can be tuned live while the video plays.
            cv::createTrackbar("hmin", window, &hmin, 179);
            cv::createTrackbar("hmax", window, &hmax, 179);
            cv::createTrackbar("smin", window, &smin, 255);
            cv::createTrackbar("vmin", window, &vmin, 255);
        }

        // The trail remembers the last 30 centroids; drawing them as a
        // polyline visualises the object's recent path.
        std::deque<cv::Point> trail;
        const size_t kTrailLen = 30;

        // A small elliptical kernel matches roundish noise blobs better than
        // a square one and erodes object corners less aggressively.
        const cv::Mat kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

        cv::Mat frame, hsv, mask;
        long frames = 0, hitFrames = 0;
        while (src.read(frame)) {
            ++frames;

            // WHY HSV: in BGR a "green" pixel changes all three channels when
            // the lighting dims. HSV separates chromaticity (H) from purity
            // (S) and brightness (V), so a single hue band survives shadows
            // and highlights that would break a BGR threshold.
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

            // inRange: 1 where every channel sits inside [lo, hi], else 0.
            // This is the whole segmentation step — one comparison per pixel.
            // Hue is CIRCULAR: red straddles the 179->0 seam, so a band like
            // "--hmin 170 --hmax 10" is the natural red range. A single
            // inRange with hmin > hmax would match nothing, so we split the
            // wrapped band into [hmin..179] OR [0..hmax].
            if (hmin <= hmax) {
                cv::inRange(hsv, cv::Scalar(hmin, smin, vmin), cv::Scalar(hmax, 255, 255),
                            mask);
            } else {
                cv::Mat maskLo, maskHi;
                cv::inRange(hsv, cv::Scalar(hmin, smin, vmin), cv::Scalar(179, 255, 255),
                            maskHi);
                cv::inRange(hsv, cv::Scalar(0, smin, vmin), cv::Scalar(hmax, 255, 255),
                            maskLo);
                cv::bitwise_or(maskLo, maskHi, mask);
            }

            // OPEN (erode then dilate) deletes isolated speckles smaller than
            // the kernel; the follow-up DILATE grows the surviving blob back
            // and fuses small gaps so findContours sees one solid region.
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel);

            // EXTERNAL: only outermost outlines — holes inside the blob are
            // irrelevant for locating it. SIMPLE compresses straight runs.
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Track the LARGEST blob: with a decent colour band the object of
            // interest dominates, and leftover noise blobs stay small.
            int best = -1;
            double bestArea = 0.0;
            for (size_t i = 0; i < contours.size(); ++i) {
                const double a = cv::contourArea(contours[i]);
                if (a > bestArea) {
                    bestArea = a;
                    best = static_cast<int>(i);
                }
            }

            bool hit = false;
            if (best >= 0 && bestArea >= minArea) {
                hit = true;
                ++hitFrames;

                const cv::Rect box = cv::boundingRect(contours[best]);
                // Moments give the area-weighted centre of the blob — more
                // stable than the box centre when the outline is ragged.
                const cv::Moments m = cv::moments(contours[best]);
                const cv::Point centroid(static_cast<int>(m.m10 / m.m00),
                                         static_cast<int>(m.m01 / m.m00));

                trail.push_back(centroid);
                if (trail.size() > kTrailLen) {
                    trail.pop_front();
                }

                if (!headless) {
                    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
                    cv::circle(frame, centroid, 4, cv::Scalar(0, 0, 255), -1);
                    cv::putText(frame, "area=" + std::to_string((int)bestArea),
                                box.tl() + cv::Point(0, -6), cv::FONT_HERSHEY_SIMPLEX,
                                0.5, cv::Scalar(0, 255, 0), 1);
                }
            } else if (!trail.empty()) {
                // Lost the object: age the trail out so a stale path does not
                // linger on screen forever.
                trail.pop_front();
            }

            if (!headless) {
                // Draw the trail as connected segments, fading nothing —
                // simple and cheap; the deque bound keeps it short.
                for (size_t i = 1; i < trail.size(); ++i) {
                    cv::line(frame, trail[i - 1], trail[i], cv::Scalar(255, 0, 255), 2);
                }

                // Show the frame and the mask side by side so cause (mask)
                // and effect (tracking) can be compared while tuning.
                cv::Mat maskBgr, panel;
                cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);
                cv::hconcat(frame, maskBgr, panel);
                cv::imshow(window, panel);

                // Mask to the low byte: some Linux backends set modifier bits
                // above it (waitKey returns -1 -> 255, harmless).
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
            } else {
                (void)hit;  // summary counters are the headless output
            }

            if (maxFrames > 0 && frames >= maxFrames) {
                break;
            }
        }

        if (!headless) {
            cv::destroyAllWindows();
        }
        std::cout << "frames=" << frames << " hit_frames=" << hitFrames << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
