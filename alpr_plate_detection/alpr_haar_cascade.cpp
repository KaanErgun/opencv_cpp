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
#include "./helper/Helper.h"
#include "./webapi/webapi.h"
#include "./thirdparty/json/json/json.h"

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

void cameraThread(int cameraIndex, cv::CascadeClassifier& plate_cascade) {
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream from camera " << cameraIndex << std::endl;
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

void sendToWebAPI(const std::string& plateStr, const cv::Mat& orgImage, const cv::Mat& plateImg, int camera_id) {
    std::string auth_token;
    if (WebAPI::getAuthToken(auth_token)) {
        std::string vehicleImageBase64, numberPlateImageBase64;
        
        // Encode vehicle image
        {
            std::vector<uchar> buf;
            cv::imencode(".jpg", orgImage, buf);
            auto base64_png = reinterpret_cast<const unsigned char*>(buf.data());
            vehicleImageBase64 = std::move("image/jpeg;base64," + Helper::base64_encode(base64_png, buf.size()));
        }

        // Encode number plate image
        {
            std::vector<uchar> buf;
            cv::imencode(".jpg", plateImg, buf);
            auto base64_png = reinterpret_cast<const unsigned char*>(buf.data());
            numberPlateImageBase64 = std::move("image/jpeg;base64," + Helper::base64_encode(base64_png, buf.size()));
        }

        auto dt = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());

        std::string station_id = "";
        uint16_t logtype = 2;
        
        if (camera_id == 1) {
            logtype = 1;
            station_id = "28038"; // giriş id
        } else if (camera_id == 2) {
            station_id = "28039"; // çıkış id
            logtype = 2;
        } else {
            logtype = 2; // default
        }

        Json::Value jReq;
        jReq["carAccessDateTime"] = std::to_string((int)(dt.time_since_epoch().count() / 1000) - 3600);
        jReq["systemTime"] = std::to_string((int)(dt.time_since_epoch().count() / 1000) - 3600);
        jReq["alprsLogType"] = unsigned(logtype);
        jReq["vehicleImageFileName"] = "img7.jpeg";
        jReq["numberPlateImageFileName"] = "img7-plate.jpeg";
        jReq["carRegistrationNumber"] = plateStr;
        jReq["numberPlateCharacters"] = plateStr;
        jReq["carDetailId"] = unsigned(1);
        jReq["alprsSystemId"] = "";
        jReq["alprsStationId"] = station_id;
        jReq["alprsSystemSerialNumber"] = "";
        jReq["username"] = "alpdade";
        jReq["password"] = "alpdade";
        jReq["geographicalAreaNameCode"] = "AU";
        jReq["vehicleMake"] = ""; //"Porsche"
        jReq["vehicleModel"] = ""; //"cayenne";
        jReq["alprsContrastValue"] = "60";
        jReq["numberPlateRecognitionScore"] = "90";
        jReq["vehicleImageFile"] = vehicleImageBase64;
        jReq["numberPlateImageFile"] = numberPlateImageBase64;
        jReq["alprsFK"] = "string";
        jReq["bookingId"] = unsigned(1);

        bool success = WebAPI::uploadPlateLog(auth_token, std::move(jReq.toStyledString()));

        if (success) {
            std::cout << std::endl << "\033[33m  ** The plate has been sent to the server. **    \033[0m" << std::endl;
        } else {
            std::cerr << std::endl << "\033[33m  ** Error: The plate hasn't been sent to the server! **    \033[0m" << std::endl;
        }
    } else {
        std::cerr << "Authentication failed.\n";
        std::cerr << "WebAPI: The Auth Token hasn't been gotten from the server.\n";
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
        cv::Mat originalFrame = raw_frames[cameraIndex];
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
                
                sendToWebAPI(best_plate, originalFrame, plateRegion, cameraIndex);
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

    std::string country = "au"; // Specify the country code for your plates
    std::string configFile = "openalpr.conf";
    std::string runtimeDataDir = "/usr/local/share/openalpr/runtime_data";
    std::string cascadePath = "plate.xml";

    cv::CascadeClassifier plate_cascade;
    if (!plate_cascade.load(cascadePath)) {
        std::cerr << "Error loading Haar Cascade file" << std::endl;
        return -1;
    }

    std::thread cameraThreads[2];
    std::thread alprThreads[2];
    std::thread saveThreads[2];

    // Assuming you have two cameras. If you have only one, you can adjust accordingly.
    cameraThreads[0] = std::thread(cameraThread, 0, std::ref(plate_cascade));
    cameraThreads[1] = std::thread(cameraThread, 1, std::ref(plate_cascade));
    alprThreads[0] = std::thread(openALPRThread, 0, country, configFile, runtimeDataDir);
    alprThreads[1] = std::thread(openALPRThread, 1, country, configFile, runtimeDataDir);

    saveThreads[0] = std::thread(saveFramesPeriodically, 0);
    saveThreads[1] = std::thread(saveFramesPeriodically, 1);

    std::thread userInputThread(handleUserInput);

    for (int i = 0; i < 2; ++i) {
        cameraThreads[i].join();
        alprThreads[i].join();
        saveThreads[i].join();
    }

    userInputThread.join();
    return 0;
}
