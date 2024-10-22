#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0); // Use USB camera, usually index 0
    if (!cap.isOpened()) {
        std::cerr << "USB camera could not be opened." << std::endl;
        return -1;
    }

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    cv::namedWindow("Human Detection", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame;
        cap >> frame; // Capture frame from USB camera
        if (frame.empty()) break; // If no frame is captured, exit the loop

        std::vector<cv::Rect> detections;
        std::vector<double> foundWeights;

        // Detect people in the frame
        hog.detectMultiScale(frame, detections, foundWeights);

        for (size_t i = 0; i < detections.size(); i++) {
            cv::Rect &d = detections[i];
            // Draw green rectangle around detected person
            cv::rectangle(frame, d, cv::Scalar(0, 255, 0), 3);
            std::cout << "Detected person, weight: " << foundWeights[i] << std::endl;
        }

        cv::imshow("Human Detection", frame);

        // Exit loop if any key is pressed
        if (cv::waitKey(30) >= 0) break;
    }

    cap.release(); // Release the camera resource
    cv::destroyAllWindows();
    return 0;
}
