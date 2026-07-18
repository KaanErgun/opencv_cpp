// app_face_detect — classic Haar-cascade face & eye detection (pre-deep-learning CV).
//
// LEARN:
//   * Haar features: sums/differences of pixel intensities over rectangular
//     regions — cheap to evaluate via integral images, so thousands can be
//     checked per window in real time.
//   * Cascade classifiers: a chain of increasingly strict stages; most windows
//     are rejected by the first few cheap stages, which is why Viola-Jones
//     detection was the first face detector fast enough for live video (2001).
//   * detectMultiScale parameters: scaleFactor / minNeighbors / minSize and
//     the speed-vs-recall trade-offs they control.
//   * ROI-restricted secondary detection: searching for eyes only inside the
//     upper half of each detected face.
//
// Usage:
//   app_face_detect                                   # webcam 0, GUI window
//   app_face_detect --source clip.mp4                 # video file
//   app_face_detect --image portrait.jpg              # one image, prints faces=N
//   app_face_detect --image portrait.jpg --show       # ... and displays result
//   app_face_detect --cascade-dir /path/to/haarcascades
//   app_face_detect --headless --max-frames 90        # smoke-test mode

#include <cstdlib>
#include <iostream>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

// OpenCV 5 moved cv::CascadeClassifier from objdetect into the contrib module
// xobjdetect (the class and API are unchanged). Pick the right header per
// version so this example builds against both OpenCV 4 and 5.
#if CV_VERSION_MAJOR >= 5
#include <opencv2/xobjdetect.hpp>
#else
#include <opencv2/objdetect.hpp>
#endif
#include <vector>

#include "vision/cli.hpp"
#include "vision/video_source.hpp"

namespace {

// Locates the directory holding the Haar cascade XML files. Resolution order:
//   1. --cascade-dir flag
//   2. OPENCV_HAAR_DIR environment variable
//   3. A short list of common install locations (Homebrew / Linux packages).
std::string resolveCascadeDir(const std::string& flagDir) {
    auto hasFaceXml = [](const std::string& dir) {
        // A directory "works" if the face cascade actually loads from it.
        cv::CascadeClassifier probe;
        return probe.load(dir + "/haarcascade_frontalface_default.xml");
    };
    if (!flagDir.empty()) {
        if (hasFaceXml(flagDir)) return flagDir;
        throw std::runtime_error(
            "no haarcascade_frontalface_default.xml in --cascade-dir '" + flagDir + "'");
    }
    if (const char* env = std::getenv("OPENCV_HAAR_DIR")) {
        if (hasFaceXml(env)) return env;
    }
    const std::vector<std::string> candidates = {
        "/usr/local/opt/opencv/share/opencv5/haarcascades",
        "/opt/homebrew/opt/opencv/share/opencv5/haarcascades",
        "/usr/local/share/opencv4/haarcascades",
        "/usr/share/opencv4/haarcascades",
    };
    for (const auto& dir : candidates) {
        if (hasFaceXml(dir)) return dir;
    }
    throw std::runtime_error(
        "could not find Haar cascade files; pass --cascade-dir <dir> pointing at a "
        "directory containing haarcascade_frontalface_default.xml and "
        "haarcascade_eye.xml");
}

// Runs face + eye detection on one BGR frame; draws results in place.
// Returns the number of faces found.
int detectAndDraw(const cv::Mat& frameBgr, cv::Mat& drawOn,
                  cv::CascadeClassifier& faceCascade, cv::CascadeClassifier& eyeCascade) {
    // Haar features are computed on intensity only, so we work in grayscale.
    cv::Mat gray;
    cv::cvtColor(frameBgr, gray, cv::COLOR_BGR2GRAY);

    // Lighting normalisation: Haar features compare raw intensity sums, so a
    // dim or washed-out image shifts every feature value. Histogram
    // equalisation spreads intensities over the full range, making the
    // classifier far less sensitive to exposure and lighting conditions.
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    // detectMultiScale slides a fixed-size window over an image pyramid:
    //   scaleFactor 1.1  — shrink the image by 10% per pyramid level. Smaller
    //                      steps find more sizes (better recall) but cost more
    //                      levels (slower). 1.1 is the classic balance.
    //   minNeighbors 5   — a real face fires at many overlapping windows;
    //                      require >=5 overlapping hits before accepting.
    //                      Higher = fewer false positives, may miss weak faces.
    //   minSize 60x60    — skip pyramid levels that would detect tiny faces;
    //                      big speed win and tiny detections are unreliable.
    faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(60, 60));

    for (const auto& face : faces) {
        cv::rectangle(drawOn, face, cv::Scalar(0, 255, 0), 2);

        // Eyes are anatomically in the upper half of a face. Restricting the
        // eye search to that ROI both removes false positives (nostrils and
        // mouth corners look eye-like to Haar features) and makes it faster —
        // the classifier scans a fraction of the pixels.
        const cv::Rect upperHalf(face.x, face.y, face.width, face.height / 2);
        cv::Mat eyeRegion = gray(upperHalf);
        std::vector<cv::Rect> eyes;
        eyeCascade.detectMultiScale(eyeRegion, eyes, 1.1, 5, 0, cv::Size(15, 15));
        for (const auto& eye : eyes) {
            // Eye rects are relative to the ROI: offset back to frame coords.
            const cv::Point center(upperHalf.x + eye.x + eye.width / 2,
                                   upperHalf.y + eye.y + eye.height / 2);
            const int radius = (eye.width + eye.height) / 4;
            cv::circle(drawOn, center, radius, cv::Scalar(255, 0, 0), 2);
        }
    }
    return static_cast<int>(faces.size());
}

}  // namespace

int main(int argc, char** argv) {
    using namespace vision::cli;
    try {
        const std::string imagePath = argValue(argc, argv, "--image");
        const std::string cascadeDir =
            resolveCascadeDir(argValue(argc, argv, "--cascade-dir"));
        const bool headless = hasFlag(argc, argv, "--headless");
        const int maxFrames = argInt(argc, argv, "--max-frames", 0);

        cv::CascadeClassifier faceCascade, eyeCascade;
        if (!faceCascade.load(cascadeDir + "/haarcascade_frontalface_default.xml") ||
            !eyeCascade.load(cascadeDir + "/haarcascade_eye.xml")) {
            std::cerr << "Error: failed to load cascades from '" << cascadeDir << "'\n";
            return EXIT_FAILURE;
        }

        const std::string window = "app_face_detect";

        // ---- Single-image mode -------------------------------------------
        if (!imagePath.empty()) {
            cv::Mat image = cv::imread(imagePath);
            if (image.empty()) {
                std::cerr << "Error: could not read image '" << imagePath << "'\n";
                return EXIT_FAILURE;
            }
            const int faces = detectAndDraw(image, image, faceCascade, eyeCascade);
            std::cout << "faces=" << faces << "\n";
            if (hasFlag(argc, argv, "--show") && !headless) {
                cv::imshow(window, image);
                cv::waitKey(0);  // any key closes
                cv::destroyAllWindows();
            }
            return EXIT_SUCCESS;
        }

        // ---- Video mode ---------------------------------------------------
        vision::VideoSource src(
            vision::SourceSpec::parse(argOr(argc, argv, "--source", "0")));
        if (!src.isOpened()) {
            std::cerr << "Error: could not open source\n";
            return EXIT_FAILURE;
        }
        if (!headless) {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
        }

        cv::Mat frame;
        int frames = 0;
        long facesTotal = 0;
        while (src.read(frame)) {
            facesTotal += detectAndDraw(frame, frame, faceCascade, eyeCascade);
            ++frames;
            if (!headless) {
                cv::imshow(window, frame);
                // Mask to the low byte: some Linux backends set modifier bits
                // above it (waitKey's -1 "no key" becomes 255, harmless).
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') break;
                // On backends whose windows have a close button (GTK/Qt),
                // imshow would otherwise resurrect a closed window forever
                // (macOS Cocoa windows can't be closed).
                if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) {
                    break;
                }
            }
            if (maxFrames > 0 && frames >= maxFrames) break;
        }

        if (!headless) {
            cv::destroyAllWindows();
        }
        std::cout << "frames=" << frames << " faces_total=" << facesTotal << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
