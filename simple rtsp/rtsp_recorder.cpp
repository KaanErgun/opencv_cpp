#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    // RTSP URL
    std::string rtsp_url = "rtsp://admin:alpDADE2@10.54.41.88:554";

    // VideoCapture object to capture the RTSP stream
    cv::VideoCapture cap(rtsp_url);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open RTSP stream." << std::endl;
        return -1;
    }

    // Set the frame size to 480p
    int frame_width = 640;  // 480p width
    int frame_height = 480; // 480p height

    // Define the codec and create VideoWriter object to save the video
    cv::VideoWriter video("output_480p.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height));

    if (!video.isOpened()) {
        std::cerr << "Error: Could not open the output video file for write." << std::endl;
        return -1;
    }

    // Record start time
    auto start_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::minutes(10);  // 10 minutes duration

    // Loop to read frames from the RTSP stream and write them to the video file
    cv::Mat frame;
    while (true) {
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty()) {
            std::cerr << "Error: Frame is empty." << std::endl;
            break;
        }

        // Resize the frame to 480p
        cv::resize(frame, frame, cv::Size(frame_width, frame_height));

        // Write the frame into the file
        video.write(frame);

        // Display the resulting frame (optional)
        cv::imshow("Frame", frame);

        // Check if 10 minutes have passed
        auto current_time = std::chrono::steady_clock::now();
        if (current_time - start_time >= duration) {
            std::cout << "10 minutes have passed. Stopping the recording." << std::endl;
            break;
        }

        // Press 'q' on the keyboard to exit the loop early
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the VideoCapture and VideoWriter objects
    cap.release();
    video.release();

    // Close all OpenCV windows
    cv::destroyAllWindows();

    return 0;
}
