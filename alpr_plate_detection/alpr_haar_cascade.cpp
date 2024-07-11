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
#include <iomanip>

std::atomic<bool> stop_thread(false);
std::atomic<bool> enable_saving(false);
std::mutex frame_mutex[2];
std::condition_variable frame_cv[2];
cv::Mat frames[2];
cv::Mat raw_frames[2];
cv::Mat masked_frames[2];
bool new_frame[2] = {false, false};

cv::Mat preprocessFrame(const cv::Mat& frame, int cameraIndex) {
    int width = frame.cols;
    int height = frame.rows;

    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
    if (cameraIndex == 0) {
        cv::rectangle(mask, cv::Rect(width / 4, 0, width / 2, height), cv::Scalar(255), cv::FILLED);
    } else if (cameraIndex == 1) {
        cv::rectangle(mask, cv::Rect(5 * width / 8, 0, 3 * width / 8, height), cv::Scalar(255), cv::FILLED);
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

    plate_cascade.detectMultiScale(gray, plates, 1.1, 3, 0, cv::Size(30, 30));

    if (plates.empty()) {
        return cv::Mat();
    }

    cv::Rect largestPlate = *std::max_element(plates.begin(), plates.end(), 
        [](const cv::Rect& a, const cv::Rect& b) { return a.area() < b.area(); });

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
        if (!cap.read(frame)) {
            std::cerr << "Error reading frame from camera " << cameraIndex << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex[cameraIndex]);
            raw_frames[cameraIndex] = frame.clone();
            masked_frames[cameraIndex] = preprocessFrame(frame, cameraIndex);
        }

        cv::Mat processedFrame = preprocessFrame(frame, cameraIndex);
        cv::Mat plateRegion = extractPlateRegion(processedFrame, plate_cascade);

        if (!plateRegion.empty()) {
            {
                std::lock_guard<std::mutex> lock(frame_mutex[cameraIndex]);
                frames[cameraIndex] = plateRegion;
                new_frame[cameraIndex] = true;
            }
            frame_cv[cameraIndex].notify_one();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void saveFramesPeriodically(int cameraIndex) {
    while (!stop_thread) {
        std::this_thread::sleep_for(std::chrono::minutes(1));
        if (enable_saving) {
            std::lock_guard<std::mutex> lock(frame_mutex[cameraIndex]);
            if (!raw_frames[cameraIndex].empty() && !masked_frames[cameraIndex].empty()) {
                std::time_t now = std::time(nullptr);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&now), "%Y%m%d%H%M%S");
                std::string timestamp = ss.str();

                std::string raw_filepath = "raw_cam" + std::to_string(cameraIndex) + "_" + timestamp + ".jpg";
                std::string masked_filepath = "masked_cam" + std::to_string(cameraIndex) + "_" + timestamp + ".jpg";

                cv::imwrite(raw_filepath, raw_frames[cameraIndex]);
                cv::imwrite(masked_filepath, masked_frames[cameraIndex]);
            }
        }
    }
}

void saveDetectedFrames(int cameraIndex, const std::string& raw_image, const std::string& masked_image) {
    std::unique_lock<std::mutex> lock(frame_mutex[cameraIndex]);
    if (!frame_cv[cameraIndex].wait_for(lock, std::chrono::seconds(5), [&]{ return new_frame[cameraIndex]; })) {
        return;
    }

    if (enable_saving) {
        cv::Mat raw_frame = raw_frames[cameraIndex];
        cv::Mat masked_frame = masked_frames[cameraIndex];
        new_frame[cameraIndex] = false;
        lock.unlock();

        std::time_t now = std::time(nullptr);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now), "%Y%m%d%H%M%S");
        std::string timestamp = ss.str();

        std::string raw_filepath = raw_image + "_" + timestamp + ".jpg";
        std::string masked_filepath = masked_image + "_" + timestamp + ".jpg";

        cv::imwrite(raw_filepath, raw_frame);
        cv::imwrite(masked_filepath, masked_frame);
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
        if (!frame_cv[cameraIndex].wait_for(lock, std::chrono::seconds(5), [&]{ return new_frame[cameraIndex]; })) {
            continue;
        }

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

                std::string raw_filepath = "detected_raw_cam" + std::to_string(cameraIndex);
                std::string masked_filepath = "detected_masked_cam" + std::to_string(cameraIndex);

                std::thread(saveDetectedFrames, cameraIndex, raw_filepath, masked_filepath).detach();
            }
        }

        if (max_confidence > 0) {
            std::cout << "Camera " << cameraIndex << " Plate: " << best_plate 
                      << " Confidence: " << max_confidence << std::endl;

            std::this_thread::sleep_for(std::chrono::seconds(30));
            max_confidence = -std::numeric_limits<double>::infinity();
        }
    }
}

void handleUserInput() {
    std::string input;
    while (!stop_thread) {
        std::getline(std::cin, input);
        if (input == "e") {
            enable_saving = true;
            std::cout << "Saving enabled." << std::endl;
        } else if (input == "d") {
            enable_saving = false;
            std::cout << "Saving disabled." << std::endl;
        } else if (input == "q") {
            stop_thread = true;
            std::cout << "Stopping all threads..." << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <e|d>" << std::endl;
        return -1;
    }

    enable_saving = (argv[1][0] == 'e');

    std::string country = "au";
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
    std::thread saveThreads[2];

    haarThreads[0] = std::thread(haarCascadeThread, rtsp_url1, 0, std::ref(plate_cascade));
    haarThreads[1] = std::thread(haarCascadeThread, rtsp_url2, 1, std::ref(plate_cascade));
    alprThreads[0] = std::thread(openALPRThread, 0, country, configFile, runtimeDataDir);
    alprThreads[1] = std::thread(openALPRThread, 1, country, configFile, runtimeDataDir);

    saveThreads[0] = std::thread(saveFramesPeriodically, 0);
    saveThreads[1] = std::thread(saveFramesPeriodically, 1);

    std::thread userInputThread(handleUserInput);

    for (int i = 0; i < 2; ++i) {
        haarThreads[i].join();
        alprThreads[i].join();
        saveThreads[i].join();
    }

    userInputThread.join();
    return 0;
}