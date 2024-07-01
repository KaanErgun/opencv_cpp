// g++ -std=c++11 -I/usr/local/include/openalpr -o alpr_haar_cascade.out alpr_haar_cascade.cpp `pkg-config --cflags --libs opencv4` -L/usr/local/lib -lopenalpr -Wl,-rpath,/usr/local/lib

#include <alpr.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <thread>
#include <atomic>
#include <iostream>
#include <vector>
#include <limits>
#include <condition_variable>
#include <chrono>
#include <ctime>

std::atomic<bool> stop_thread(false);
std::mutex frame_mutex[2];
std::condition_variable frame_cv[2];
cv::Mat frames[2];
cv::Mat raw_frames[2];
cv::Mat masked_frames[2];
bool new_frame[2] = {false, false};

cv::Mat preprocessFrame(const cv::Mat& frame, int cameraIndex) {
    int width = frame.cols;
    int height = frame.rows;

    cv::Mat mask = cv::Mat::zeros(height, width, frame.type());
    if (cameraIndex == 0) { // 88 ile biten kamera
        cv::rectangle(mask, cv::Rect(width / 4, 0, 3 * width / 4, height), cv::Scalar(255, 255, 255), cv::FILLED);
    } else if (cameraIndex == 1) { // 89 ile biten kamera
        cv::rectangle(mask, cv::Rect(5 * width / 8, 0, 3 * width / 8, height), cv::Scalar(255, 255, 255), cv::FILLED);
    }

    cv::Mat maskedFrame;
    frame.copyTo(maskedFrame, mask);

    return maskedFrame;
}

cv::Mat extractPlateRegion(const cv::Mat& frame, cv::CascadeClassifier& plate_cascade) {
    std::vector<cv::Rect> plates;
    cv::Mat gray;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    plate_cascade.detectMultiScale(gray, plates);

    if (plates.empty()) {
        return cv::Mat();
    }

    cv::Rect largestPlate = *std::max_element(plates.begin(), plates.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.area() < b.area();
    });

    return frame(largestPlate);
}

void haarCascadeThread(const std::string& rtsp_url, int cameraIndex, cv::CascadeClassifier& plate_cascade) {
    cv::VideoCapture cap(rtsp_url);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream: " << rtsp_url << std::endl;
        return;
    }

    while (!stop_thread) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        {
            std::lock_guard<std::mutex> lock(frame_mutex[cameraIndex]);
            raw_frames[cameraIndex] = frame.clone();
            masked_frames[cameraIndex] = preprocessFrame(frame, cameraIndex);
        }

        cv::Mat processedFrame = preprocessFrame(frame, cameraIndex);
        cv::Mat plateRegion = extractPlateRegion(processedFrame, plate_cascade);

        if (!plateRegion.empty()) {
            std::string filename = "cam" + std::to_string(cameraIndex) + ".jpg";
            cv::imwrite(filename, plateRegion);

            {
                std::lock_guard<std::mutex> lock(frame_mutex[cameraIndex]);
                frames[cameraIndex] = plateRegion;
                new_frame[cameraIndex] = true;
            }
            frame_cv[cameraIndex].notify_one();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    cap.release();
}

void saveFramesPeriodically() {
    while (!stop_thread) {
        std::this_thread::sleep_for(std::chrono::minutes(1));
        for (int i = 0; i < 2; ++i) {
            std::lock_guard<std::mutex> lock(frame_mutex[i]);
            if (!raw_frames[i].empty() && !masked_frames[i].empty()) {
                std::string raw_filepath = "raw_cam" + std::to_string(i) + ".jpg";
                std::string masked_filepath = "masked_cam" + std::to_string(i) + ".jpg";

                cv::imwrite(raw_filepath, raw_frames[i]);
                cv::imwrite(masked_filepath, masked_frames[i]);
            }
        }
    }
}

void openALPRThread(int cameraIndex, const std::string& country, const std::string& configFile, const std::string& runtimeDataDir) {
    alpr::Alpr openalpr(country, configFile, runtimeDataDir);
    if (!openalpr.isLoaded()) {
        std::cerr << "Error loading OpenALPR" << std::endl;
        return;
    }

    double max_confidence = -std::numeric_limits<double>::infinity();
    std::string best_plate;

    while (!stop_thread) {
        std::unique_lock<std::mutex> lock(frame_mutex[cameraIndex]);
        frame_cv[cameraIndex].wait(lock, [&]{ return new_frame[cameraIndex]; });

        cv::Mat plateRegion = frames[cameraIndex];
        new_frame[cameraIndex] = false;
        lock.unlock();

        std::vector<unsigned char> buffer;
        cv::imencode(".jpg", plateRegion, buffer);
        std::vector<char> image_data(buffer.begin(), buffer.end());

        alpr::AlprResults results = openalpr.recognize(image_data);

        for (auto& plate : results.plates) {
            if (plate.bestPlate.overall_confidence > max_confidence) {
                max_confidence = plate.bestPlate.overall_confidence;
                best_plate = plate.bestPlate.characters;

                std::time_t now = std::time(nullptr);
                std::tm* local_time = std::localtime(&now);

                char timestamp[20];
                std::strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", local_time);

                std::string raw_filepath = "detected_raw_cam" + std::to_string(cameraIndex) + "_" + timestamp + ".jpg";
                std::string masked_filepath = "detected_masked_cam" + std::to_string(cameraIndex) + "_" + timestamp + ".jpg";

                cv::imwrite(raw_filepath, raw_frames[cameraIndex]);
                cv::imwrite(masked_filepath, masked_frames[cameraIndex]);
            }
        }

        if (max_confidence > 0) {
            std::cout << "Camera " << cameraIndex << " Plate: " << best_plate 
                      << " Confidence: " << max_confidence << std::endl;

            // Uyutma işlemi
            std::this_thread::sleep_for(std::chrono::seconds(30));
            max_confidence = -std::numeric_limits<double>::infinity();
        }
    }
}

int main() {
    std::string country = "au"; // Avustralya
    std::string configFile = "openalpr.conf";
    std::string runtimeDataDir = "/usr/local/share/openalpr/runtime_data";
    std::string cascadePath = "plate.xml";

    cv::CascadeClassifier plate_cascade;
    if (!plate_cascade.load(cascadePath)) {
        std::cerr << "Error loading Haar Cascade file" << std::endl;
        return -1;
    }

    std::string rtsp_url1 = "rtsp://admin:alpDADE2@10.54.41.88:554";
    std::string rtsp_url2 = "rtsp://admin:alpDADE2@10.54.41.89:554";

    std::thread haarThreads[2];
    std::thread alprThreads[2];

    haarThreads[0] = std::thread(haarCascadeThread, rtsp_url1, 0, std::ref(plate_cascade));
    haarThreads[1] = std::thread(haarCascadeThread, rtsp_url2, 1, std::ref(plate_cascade));
    alprThreads[0] = std::thread(openALPRThread, 0, country, configFile, runtimeDataDir);
    alprThreads[1] = std::thread(openALPRThread, 1, country, configFile, runtimeDataDir);

    std::thread saveThread(saveFramesPeriodically);

    for (int i = 0; i < 2; ++i) {
        haarThreads[i].join();
        alprThreads[i].join();
    }

    saveThread.join();
    cv::destroyAllWindows();
    return 0;
}
