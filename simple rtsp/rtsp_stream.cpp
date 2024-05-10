#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap("rtsp://192.168.0.21/live/1"); // RTSP URL

    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    cv::namedWindow("RTSP Stream", cv::WINDOW_NORMAL); // Create window

    while (true) {
        cv::Mat frame;
        cap >> frame; // Read a frame from the video stream

        if (frame.empty()) {
            std::cerr << "Received empty frame" << std::endl;
            break;
        }

        cv::imshow("RTSP Stream", frame); // Display the frame

        // Exit the loop when 'ESC' is pressed
        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    cap.release(); // Release the resource
    cv::destroyAllWindows(); // Close all windows
    return 0;
}
