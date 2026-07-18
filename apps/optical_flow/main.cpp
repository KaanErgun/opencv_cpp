// app_optical_flow — sparse Lucas-Kanade optical flow with point trails.
//
// LEARN:
//   * Optical flow assumptions: brightness constancy (a pixel keeps its
//     intensity as it moves) and small motion between frames. When either
//     breaks, tracking fails — which is why we manage track lifecycles.
//   * Shi-Tomasi corner detection (cv::goodFeaturesToTrack): picking points
//     that are actually trackable.
//   * Pyramidal Lucas-Kanade (cv::calcOpticalFlowPyrLK): coarse-to-fine
//     tracking so large motions still satisfy the "small motion" assumption.
//   * Track lifecycle management: dropping lost points, reseeding when too
//     few survive (drift and occlusion slowly kill every track).
//
// Usage:
//   app_optical_flow                              # webcam 0, GUI
//   app_optical_flow --source clip.mp4 --points 300
//   app_optical_flow --source clip.mp4 --headless --max-frames 200
//   Keys: 'r' reseeds features, ESC or 'q' quits.

#include <cmath>
#include <iostream>
// NOTE: on OpenCV 5.x goodFeaturesToTrack moved from imgproc to the new
// `features` module (the successor of features2d). On 4.x that header does
// not exist and imgproc (included below) already provides the function.
#if __has_include(<opencv2/features.hpp>)
#include <opencv2/features.hpp>
#endif
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>  // calcOpticalFlowPyrLK lives in the video module
#include <string>
#include <vector>

#include "vision/cli.hpp"
#include "vision/video_source.hpp"

namespace {

// Detect fresh Shi-Tomasi corners. Corners are the only points optical flow
// can localise in BOTH x and y: on an edge the point can slide along the edge
// (aperture problem), and in a flat region there is no gradient at all. A
// corner has strong gradients in two directions, so its motion is fully
// determined — that is exactly what goodFeaturesToTrack scores.
std::vector<cv::Point2f> seedFeatures(const cv::Mat& gray, int maxPoints) {
    std::vector<cv::Point2f> pts;
    // qualityLevel 0.01: keep corners at least 1% as strong as the best one.
    // minDistance 8: spread points out so trails cover the whole scene.
    cv::goodFeaturesToTrack(gray, pts, maxPoints, 0.01, 8.0);
    return pts;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace vision::cli;
    const std::string source = argOr(argc, argv, "--source", "0");
    const bool headless = hasFlag(argc, argv, "--headless");
    const int maxFrames = argInt(argc, argv, "--max-frames", 0);  // 0 = unbounded
    const int maxPoints = argInt(argc, argv, "--points", 200);

    try {
        vision::VideoSource src(vision::SourceSpec::parse(source));
        if (!src.isOpened()) {
            std::cerr << "Error: could not open source '" << source << "'\n";
            return EXIT_FAILURE;
        }

        const std::string window = "app_optical_flow";
        if (!headless) {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
        }

        cv::Mat frame, gray, prevGray;
        cv::Mat trails;  // persistent overlay: lines accumulate here, never cleared
        std::vector<cv::Point2f> prevPts;
        int seededCount = 0;  // how many points the last reseed produced

        long frames = 0;
        double dispSum = 0.0;  // sum of per-frame mean displacements (headless stat)
        long dispFrames = 0;   // frames that contributed to dispSum

        while (src.read(frame)) {
            // LK works on intensity, not colour: brightness constancy is an
            // assumption about grayscale values, so we convert once per frame.
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            if (trails.empty()) {
                trails = cv::Mat::zeros(frame.size(), frame.type());
            }

            // A live-source reconnect can deliver frames at a different
            // resolution. LK needs prev/next images of the same size, and
            // add() needs trails to match the frame — so drop all tracking
            // state and start over at the new size.
            if (!prevGray.empty() && gray.size() != prevGray.size()) {
                prevGray.release();
                prevPts.clear();
                trails = cv::Mat::zeros(frame.size(), frame.type());
            }

            // Whether this iteration only (re)seeded instead of tracking.
            // IMPORTANT: seeding must NOT skip the display/key/max-frames
            // section below — on featureless input (no corners found) a
            // `continue` here would starve the key pump and the frame
            // counter, making the app unquittable.
            bool seeded = false;
            std::vector<cv::Point2f> keptPrev, keptNext;

            if (prevGray.empty() || prevPts.empty()) {
                // First frame (or everything lost): seed the initial tracks.
                // Flow needs two frames, so this iteration just displays.
                prevPts = seedFeatures(gray, maxPoints);
                seededCount = static_cast<int>(prevPts.size());
                gray.copyTo(prevGray);
                seeded = true;
            } else {
                // Pyramidal LK: build image pyramids and track from the coarsest
                // level down. A 32 px motion at full resolution is only 4 px three
                // pyramid levels up — small enough for the LK linearisation to
                // hold — and each finer level refines the estimate.
                std::vector<cv::Point2f> nextPts;
                std::vector<uchar> status;
                std::vector<float> err;
                cv::calcOpticalFlowPyrLK(prevGray, gray, prevPts, nextPts, status, err,
                                         cv::Size(21, 21), 3);

                // Keep only points LK tracked successfully (status == 1). Lost
                // points are gone for good — resurrecting them would just track
                // whatever now occupies that pixel.
                for (size_t i = 0; i < status.size(); ++i) {
                    if (status[i]) {
                        keptPrev.push_back(prevPts[i]);
                        keptNext.push_back(nextPts[i]);
                    }
                }

                // Mean displacement of survivors — a scalar "how much did the
                // scene move" signal, also our headless summary metric.
                if (!keptNext.empty()) {
                    double sum = 0.0;
                    for (size_t i = 0; i < keptNext.size(); ++i) {
                        sum += cv::norm(keptNext[i] - keptPrev[i]);
                    }
                    dispSum += sum / static_cast<double>(keptNext.size());
                    ++dispFrames;
                }

                // Draw motion: a line segment per point on the persistent trail
                // layer, plus a dot at the current position on the live frame.
                for (size_t i = 0; i < keptNext.size(); ++i) {
                    cv::line(trails, keptPrev[i], keptNext[i], cv::Scalar(0, 255, 0), 1,
                             cv::LINE_AA);
                    cv::circle(frame, keptNext[i], 3, cv::Scalar(0, 0, 255), cv::FILLED);
                }
            }

            bool reseed = false;
            // Auto-reseed: tracks die over time — drift (small errors compound
            // frame to frame), occlusion, and points leaving the image. Once
            // fewer than 25% of the seeded points survive, the remaining set
            // is too sparse to describe the scene, so detect fresh corners.
            if (!seeded && seededCount > 0 &&
                static_cast<int>(keptNext.size()) < seededCount / 4) {
                reseed = true;
            }

            if (!headless) {
                // Overlay the accumulated trails on the current frame. add()
                // works because trails is black where nothing was drawn.
                cv::Mat display;
                cv::add(frame, trails, display);
                const std::string label =
                    seeded ? "seeking features...  ('r' reseeds, ESC/q quits)"
                           : "tracks: " + std::to_string(keptNext.size()) +
                                 "  ('r' reseeds, ESC/q quits)";
                cv::putText(display, label, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                cv::imshow(window, display);
                // Mask to the low byte: some Linux backends set modifier bits
                // above it. (waitKey's -1 "no key" becomes 255 — harmless.)
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') {
                    break;
                }
                if (key == 'r') {
                    reseed = true;
                }
                // On backends whose windows have a close button (GTK/Qt),
                // imshow would otherwise resurrect a closed window forever
                // (macOS Cocoa windows can't be closed).
                if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) {
                    break;
                }
            }

            if (!seeded) {
                if (reseed) {
                    prevPts = seedFeatures(gray, maxPoints);
                    seededCount = static_cast<int>(prevPts.size());
                    trails.setTo(cv::Scalar::all(0));  // old trails no longer apply
                } else {
                    prevPts = std::move(keptNext);
                }
                gray.copyTo(prevGray);
            }

            ++frames;
            if (maxFrames > 0 && frames >= maxFrames) {
                break;
            }
        }

        if (!headless) {
            cv::destroyAllWindows();
        }
        const double meanDisp = dispFrames > 0 ? dispSum / dispFrames : 0.0;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f", meanDisp);
        std::cout << "frames=" << frames << " mean_disp=" << buf << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
