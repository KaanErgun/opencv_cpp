// app_object_track — single-object tracking: initialise once, follow every frame.
//
// LEARN:
//   * Detection vs tracking: a DETECTOR answers "where are objects of class X?"
//     on every frame independently (expensive, needs a model). A TRACKER is told
//     ONCE where one object is, then follows that patch frame-to-frame using
//     appearance/correlation cues (cheap, class-agnostic, but it never re-detects:
//     once lost, it stays lost until you re-initialise it).
//   * Tracker families and the accuracy-vs-speed tradeoff:
//       - CSRT (contrib): discriminative correlation filter with channel and
//         spatial reliability. Most accurate of the three, handles scale change
//         and non-rectangular objects better — but the slowest (often ~25 fps).
//       - KCF (contrib): kernelized correlation filter. Very fast (100+ fps),
//         but a fixed-scale model; it drifts when the object grows/shrinks.
//       - MIL (core video module): Multiple Instance Learning, an older
//         boosting-based baseline. Kept in OpenCV core mostly for teaching and
//         comparison; slower AND less accurate than CSRT today.
//   * Tracker lifecycle: create() -> init(frame, roi) -> update(frame, box)
//     once per frame. update() returns false when the internal confidence
//     collapses — typical causes are DRIFT (the model slowly locks onto
//     background texture) and OCCLUSION (the object passes behind something).
//   * cv::selectROI for interactive box selection, cv::getTickCount for FPS.
//
// Usage:
//   app_object_track --source 0                          # webcam, drag a box
//   app_object_track --source clip.mp4 --tracker kcf     # pick speed over accuracy
//   app_object_track --source clip.mp4 --headless --roi 100,80,60,60 --max-frames 200

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>        // contrib: TrackerCSRT, TrackerKCF
#include <opencv2/video/tracking.hpp>  // core video: TrackerMIL
#include <sstream>
#include <string>

#include "vision/cli.hpp"
#include "vision/video_source.hpp"

namespace {

// Parses "x,y,w,h" into a Rect. Returns false on any malformed input; we parse
// strictly (four comma-separated ints) so a typo fails loudly instead of
// initialising the tracker on a garbage box.
bool parseRoi(const std::string& text, cv::Rect& roi) {
    std::istringstream in(text);
    int v[4];
    char sep = ',';
    for (int i = 0; i < 4; ++i) {
        if (i > 0 && (!(in >> sep) || sep != ',')) {
            return false;
        }
        if (!(in >> v[i])) {
            return false;
        }
    }
    roi = cv::Rect(v[0], v[1], v[2], v[3]);
    return roi.width > 0 && roi.height > 0;
}

// Factory: all trackers share the cv::Tracker interface, so the per-frame loop
// below is identical regardless of algorithm — only creation differs.
cv::Ptr<cv::Tracker> makeTracker(const std::string& name) {
    if (name == "csrt") {
        return cv::TrackerCSRT::create();  // accurate, slow
    }
    if (name == "kcf") {
        return cv::TrackerKCF::create();  // fast, fixed scale
    }
    if (name == "mil") {
        return cv::TrackerMIL::create();  // legacy baseline
    }
    return nullptr;
}

}  // namespace

using vision::cli::argInt;
using vision::cli::argOr;
using vision::cli::argValue;
using vision::cli::hasFlag;

int main(int argc, char** argv) {
    try {
        const std::string source = argOr(argc, argv, "--source", "0");
        const bool headless = hasFlag(argc, argv, "--headless");
        const int maxFrames = argInt(argc, argv, "--max-frames", 0);
        const std::string trackerName = argOr(argc, argv, "--tracker", "csrt");
        const std::string roiText = argValue(argc, argv, "--roi");

        cv::Ptr<cv::Tracker> tracker = makeTracker(trackerName);
        if (!tracker) {
            std::cerr << "Error: unknown tracker '" << trackerName
                      << "' (use csrt|kcf|mil)\n";
            return EXIT_FAILURE;
        }

        vision::VideoSource src(vision::SourceSpec::parse(source));
        if (!src.isOpened()) {
            std::cerr << "Error: could not open source '" << source << "'\n";
            return EXIT_FAILURE;
        }

        // The tracker needs a first frame plus a box to learn what to follow.
        cv::Mat frame;
        if (!src.read(frame)) {
            std::cerr << "Error: could not read the first frame\n";
            return EXIT_FAILURE;
        }

        const std::string window = "app_object_track [" + trackerName + "]";
        cv::Rect roi;
        if (!roiText.empty()) {
            if (!parseRoi(roiText, roi)) {
                std::cerr << "Error: --roi expects \"x,y,w,h\" with w,h > 0\n";
                return EXIT_FAILURE;
            }
        } else if (headless) {
            // No GUI means no interactive selection, so the box is mandatory.
            std::cerr << "Error: --roi x,y,w,h is required with --headless\n";
            return EXIT_FAILURE;
        } else {
            // Interactive: user drags a rectangle, ENTER/SPACE confirms.
            roi = cv::selectROI(window, frame, /*showCrosshair=*/true);
            if (roi.width <= 0 || roi.height <= 0) {
                std::cerr << "Error: empty selection, nothing to track\n";
                return EXIT_FAILURE;
            }
        }

        // LEARN: trackers trust the box they are given — an out-of-bounds ROI
        // crashes CSRT with a cryptic cv::Exception and silently corrupts the
        // others. Intersecting with the frame rect (operator& on cv::Rect) is
        // the idiomatic OpenCV way to clip a box to valid pixels.
        const cv::Rect frameRect(0, 0, frame.cols, frame.rows);
        const cv::Rect clipped = roi & frameRect;
        if (clipped.width <= 0 || clipped.height <= 0) {
            std::cerr << "Error: --roi " << roi.x << ',' << roi.y << ',' << roi.width
                      << ',' << roi.height << " lies outside the frame (" << frame.cols
                      << 'x' << frame.rows << ")\n";
            return EXIT_FAILURE;
        }
        if (clipped != roi) {
            std::cerr << "Notice: --roi clipped to frame bounds: " << clipped.x << ','
                      << clipped.y << ',' << clipped.width << ',' << clipped.height
                      << '\n';
            roi = clipped;
        }

        // init() learns the appearance model from this single box. Everything
        // afterwards is update() — the tracker never sees the ROI flag again.
        tracker->init(frame, roi);

        int frames = 1;  // the init frame counts as processed
        int tracked = 1;
        cv::Rect lastBox = roi;

        while (src.read(frame)) {
            // Measure per-frame latency around update() only: that is where the
            // csrt/kcf/mil speed difference actually shows up.
            const int64 t0 = cv::getTickCount();
            cv::Rect box;
            const bool ok = tracker->update(frame, box);
            const double ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
            ++frames;

            if (ok) {
                ++tracked;
                lastBox = box;
            }
            // else: failure modes are drift (the model slid onto background
            // texture) or occlusion (the object is hidden). A plain tracker
            // cannot recover by itself — tracking does not re-detect; a real
            // system pairs it with a detector for re-acquisition.

            if (!headless) {
                if (ok) {
                    cv::rectangle(frame, box, {0, 255, 0}, 2);
                } else {
                    cv::putText(frame, "LOST", {20, 80}, cv::FONT_HERSHEY_SIMPLEX, 1.5,
                                {0, 0, 255}, 3);
                }
                std::ostringstream hud;
                hud.precision(1);
                hud << std::fixed << trackerName << "  " << (ms > 0 ? 1000.0 / ms : 0.0)
                    << " fps (update only)";
                cv::putText(frame, hud.str(), {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            {255, 255, 0}, 2);
                cv::imshow(window, frame);
                // Mask to the low byte: some Linux backends set modifier bits
                // above it. (waitKey returns -1 -> 255, harmless.)
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
        std::cout << "frames=" << frames << " tracked=" << tracked
                  << " last_box=" << lastBox.x << ',' << lastBox.y << ',' << lastBox.width
                  << ',' << lastBox.height << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
