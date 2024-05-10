#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap("rtsp://192.168.0.21/live/1"); // Enter your RTSP URL here
    if (!cap.isOpened()) {
        std::cerr << "Video could not be opened." << std::endl;
        return -1;
    }

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    cv::namedWindow("Human Detection", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame;
        cap >> frame; // Get the frame
        if (frame.empty()) break; // If no frame is captured, exit the loop

        std::vector<cv::Rect> detections;
        std::vector<double> foundWeights;

        hog.detectMultiScale(frame, detections, foundWeights);

        for (size_t i = 0; i < detections.size(); i++) {
            cv::Rect &d = detections[i];
            cv::rectangle(frame, d, cv::Scalar(0, 255, 0), 3); // Draw a green rectangle
            std::cout << "Detected person, weight: " << foundWeights[i] << std::endl;
        }

        cv::imshow("Human Detection", frame);

        if (cv::waitKey(30) >= 0) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
