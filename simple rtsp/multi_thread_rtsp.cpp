// g++ -std=c++11 -o multi_thread_rtsp.out multi_thread_rtsp.cpp `pkg-config --cflags --libs opencv4` -lpthread

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

// Global frames and mutexes
cv::Mat frame1, frame2;
std::mutex frame1_mutex, frame2_mutex;
std::condition_variable cv_frame1, cv_frame2;
bool ready1 = false, ready2 = false;

// Function to capture video from a camera
void captureVideo(const std::string& url, cv::Mat& frame, std::mutex& frame_mutex, std::condition_variable& cv_frame, bool& ready) {
    cv::VideoCapture cap(url);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream from " << url << std::endl;
        return;
    }

    while (true) {
        cv::Mat local_frame;
        cap >> local_frame;
        if (local_frame.empty()) {
            std::cerr << "Error: Could not retrieve frame from " << url << std::endl;
            break;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            local_frame.copyTo(frame);
            ready = true;
        }
        cv_frame.notify_one();

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    cap.release();
}

int main() {
    // Camera URLs
    std::string cam1_url = "rtsp://admin:Password.123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0";
    std::string cam2_url = "rtsp://admin:Password.1234@192.168.1.64:554/Streaming/Channels/101";

    // Create threads for each camera
    std::thread t1(captureVideo, cam1_url, std::ref(frame1), std::ref(frame1_mutex), std::ref(cv_frame1), std::ref(ready1));
    std::thread t2(captureVideo, cam2_url, std::ref(frame2), std::ref(frame2_mutex), std::ref(cv_frame2), std::ref(ready2));

    while (true) {
        {
            std::unique_lock<std::mutex> lock1(frame1_mutex);
            cv_frame1.wait(lock1, []{ return ready1; });
            ready1 = false;
            cv::imshow("Camera 1", frame1);
        }

        {
            std::unique_lock<std::mutex> lock2(frame2_mutex);
            cv_frame2.wait(lock2, []{ return ready2; });
            ready2 = false;
            cv::imshow("Camera 2", frame2);
        }

        if (cv::waitKey(30) == 27) { // Exit on ESC key press
            break;
        }
    }

    // Wait for both threads to finish
    t1.join();
    t2.join();

    cv::destroyAllWindows();
    return 0;
}
