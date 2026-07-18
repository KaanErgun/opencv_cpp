// app_motion_detect — security-camera style motion detection via background subtraction.
//
// LEARN:
//   * Background modelling: instead of diffing consecutive frames, MOG2 learns a
//     per-pixel Gaussian-mixture model of "what the scene normally looks like",
//     so slow lighting changes are absorbed while real movers stand out.
//   * MOG2 parameters: history (how many frames the model remembers),
//     varThreshold (how far a pixel must deviate before it counts as foreground)
//     and detectShadows (classify shadows separately instead of as motion).
//   * Shadow handling: MOG2 marks shadow pixels with value 127 in the mask; a
//     simple threshold at 200 keeps only "true" foreground (255).
//   * The classic morphology-then-contours pattern: clean the binary mask with
//     OPEN (erode+dilate) to kill speckle noise, dilate to merge fragments, then
//     findContours + boundingRect + an area filter to get stable motion boxes.
//
// Usage:
//   app_motion_detect --source 0                          # webcam, GUI windows
//   app_motion_detect --source clip.mp4 --min-area 1200   # ignore small movers
//   app_motion_detect --source clip.mp4 --output annotated.mp4
//   app_motion_detect --source clip.mp4 --headless --max-frames 200

#include <atomic>
#include <csignal>
#include <iostream>
#include <opencv2/geometry.hpp>  // OpenCV 5: contourArea/boundingRect live here
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>  // createBackgroundSubtractorMOG2 lives here
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

#include "vision/cli.hpp"
#include "vision/video_source.hpp"

namespace {

// SIGINT (Ctrl+C) must not hard-kill us mid-recording: VideoWriter needs a
// clean release() to write the MP4 index, otherwise the file is unplayable.
std::atomic<bool> g_stop{false};
void onSignal(int) { g_stop.store(true); }

}  // namespace

using vision::cli::argInt;
using vision::cli::argOr;
using vision::cli::argValue;
using vision::cli::hasFlag;

int main(int argc, char** argv) {
    try {
        const std::string source = argOr(argc, argv, "--source", "0");
        const bool headless = hasFlag(argc, argv, "--headless");
        const int maxFrames = argInt(argc, argv, "--max-frames", 0);  // 0 = unbounded
        const int minArea = argInt(argc, argv, "--min-area", 800);
        const std::string output = argValue(argc, argv, "--output");

        std::signal(SIGINT, onSignal);

        vision::VideoSource src(vision::SourceSpec::parse(source));
        if (!src.isOpened()) {
            std::cerr << "Error: could not open source '" << source << "'\n";
            return EXIT_FAILURE;
        }

        // MOG2 background subtractor:
        //   history = 500      -> the model adapts over roughly the last 500
        //                         frames; larger = more stable background but
        //                         slower to absorb scene changes (a parked car
        //                         takes longer to "become background").
        //   varThreshold = 16  -> squared Mahalanobis distance a pixel must
        //                         exceed to be foreground; raise it on noisy
        //                         cameras to reduce false motion.
        //   detectShadows=true -> shadows are detected and labelled 127 in the
        //                         mask instead of 255, so we can discard them
        //                         (a shadow is a lighting change, not motion).
        cv::Ptr<cv::BackgroundSubtractor> subtractor =
            cv::createBackgroundSubtractorMOG2(500, 16, true);

        cv::VideoWriter writer;  // opened lazily once we know the frame size
        cv::Size writerSize;     // size the writer was opened with

        const std::string winFrame = "app_motion_detect";
        const std::string winMask = "fgmask";
        if (!headless) {
            cv::namedWindow(winFrame, cv::WINDOW_NORMAL);
            cv::namedWindow(winMask, cv::WINDOW_NORMAL);
        }

        // Kernels: a small ellipse for OPEN (kills isolated noise pixels without
        // eating thin limbs) and a larger one for dilate (merges the fragments of
        // one mover into a single blob so it yields ONE contour, not ten).
        const cv::Mat openKernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        const cv::Mat dilateKernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));

        cv::Mat frame, fgmask;
        long frames = 0;
        long motionFrames = 0;

        while (!g_stop.load() && src.read(frame)) {
            ++frames;

            // 1) Update the background model AND get the foreground mask.
            //    Pixels: 255 = foreground, 127 = shadow, 0 = background.
            subtractor->apply(frame, fgmask);

            // 2) Threshold at 200: keeps only 255-valued (true foreground)
            //    pixels and drops the 127 shadow label — otherwise every sunny
            //    day would be one long "motion" event.
            cv::threshold(fgmask, fgmask, 200, 255, cv::THRESH_BINARY);

            // 3) Morphological cleanup. OPEN first (erode then dilate) removes
            //    salt noise from camera grain; the extra dilate afterwards grows
            //    and fuses the surviving blobs so contours are contiguous.
            cv::morphologyEx(fgmask, fgmask, cv::MORPH_OPEN, openKernel);
            cv::dilate(fgmask, fgmask, dilateKernel);

            // 4) Contours on the cleaned mask. RETR_EXTERNAL: only outer
            //    outlines, holes inside a mover are irrelevant here.
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(fgmask, contours, cv::RETR_EXTERNAL,
                             cv::CHAIN_APPROX_SIMPLE);

            // 5) Area filter: tiny contours are leaves/noise/compression
            //    artefacts; only blobs >= --min-area become motion boxes.
            bool motion = false;
            for (const auto& contour : contours) {
                if (cv::contourArea(contour) < minArea) {
                    continue;
                }
                motion = true;
                cv::rectangle(frame, cv::boundingRect(contour), cv::Scalar(0, 255, 0), 2);
            }
            if (motion) {
                ++motionFrames;
                // Red banner, security-camera style, when anything qualifies.
                cv::putText(frame, "MOTION", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX,
                            1.2, cv::Scalar(0, 0, 255), 3);
            }

            // Overlay how "busy" the mask is — a quick health indicator: near
            // 100% usually means the model is still warming up or the camera moved.
            const double pct =
                100.0 * cv::countNonZero(fgmask) / (fgmask.rows * fgmask.cols);
            cv::putText(frame, cv::format("fg: %.1f%%", pct),
                        cv::Point(10, frame.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 255, 0), 2);

            // Optional annotated recording; fps/size come from the source so the
            // output plays back at the right speed.
            if (!output.empty()) {
                if (!writer.isOpened()) {
                    writer.open(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                src.fps(), frame.size());
                    if (!writer.isOpened()) {
                        std::cerr << "Error: could not open writer '" << output << "'\n";
                        return EXIT_FAILURE;
                    }
                    writerSize = frame.size();
                }
                // A live-source reconnect can change the resolution; VideoWriter
                // silently drops frames whose size differs from what it was
                // opened with, so resize back to the original size.
                if (frame.size() != writerSize) {
                    cv::resize(frame, frame, writerSize);
                }
                writer.write(frame);
            }

            if (!headless) {
                cv::imshow(winFrame, frame);
                cv::imshow(winMask, fgmask);
                // Mask to the low byte: some Linux backends set modifier bits
                // above it. (waitKey returns -1 -> 255, harmless.)
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') {
                    break;
                }
                // On backends whose windows have a close button (GTK/Qt),
                // imshow would otherwise resurrect a closed window forever
                // (macOS Cocoa windows can't be closed).
                if (cv::getWindowProperty(winFrame, cv::WND_PROP_VISIBLE) < 1) {
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
        // Explicit release writes the MP4 index; a hard kill corrupts the file.
        if (writer.isOpened()) {
            writer.release();
        }
        if (headless) {
            // Machine-checkable summary for smoke tests.
            std::cout << "frames=" << frames << " motion_frames=" << motionFrames << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
