// g++ -std=c++11 -I/usr/local/include/openalpr -o alpr.out alpr.cpp `pkg-config --cflags --libs opencv4` -L/usr/local/lib -lopenalpr -Wl,-rpath,/usr/local/lib

#include <alpr.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <map>

std::atomic<bool> stop_thread(false);
std::queue<std::pair<int, cv::Mat>> frameQueue;
std::mutex queueMutex;
std::condition_variable queueCondVar;

int vehicleCountCamera0 = 0;
int vehicleCountCamera1 = 0;

std::map<std::string, cv::Point> previousPositions; // Plakaların önceki pozisyonlarını tutan harita

cv::Mat preprocessFrame(const cv::Mat& frame) {
    int width = frame.cols;
    int height = frame.rows;

    int cropSize = std::min(width, height);
    int xOffset = (width - cropSize) / 2;
    int yOffset = (height - cropSize) / 2;

    cv::Rect cropRegion(xOffset, yOffset, cropSize, cropSize);
    cv::Mat croppedFrame = frame(cropRegion);

    cv::Mat resizedFrame;
    cv::resize(croppedFrame, resizedFrame, cv::Size(800, 800));

    return resizedFrame;
}

cv::Mat extractPlateRegion(const cv::Mat& frame, cv::Rect& plateRect) {
    cv::Mat gray, blurred, edged;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edged, 50, 200);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edged, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> possiblePlates;

    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        double aspectRatio = (double)rect.width / rect.height;
        if (aspectRatio > 2 && aspectRatio < 6 && rect.width > 100 && rect.height > 20) {
            possiblePlates.push_back(rect);
        }
    }

    if (possiblePlates.empty()) {
        return cv::Mat();
    }

    plateRect = *std::max_element(possiblePlates.begin(), possiblePlates.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.area() < b.area();
    });

    return frame(plateRect);
}

void preProcessStream(const std::string& rtsp_url, int index) {
    cv::VideoCapture cap(rtsp_url);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream: " << rtsp_url << std::endl;
        return;
    }

    int frameCounter = 0;
    int fps = 10;
    int frameSkip = 30 / fps;

    while (!stop_thread) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        if (frameCounter % frameSkip == 0) {
            cv::Mat processedFrame = preprocessFrame(frame);

            std::unique_lock<std::mutex> lock(queueMutex);
            frameQueue.push({index, processedFrame});
            queueCondVar.notify_one();
        }

        frameCounter++;
    }

    cap.release();
}

void alprProcess(const std::string& country, const std::string& configFile, const std::string& runtimeDataDir, std::vector<alpr::AlprResults>& resultsVec, std::vector<cv::Mat>& framesVec) {
    alpr::Alpr openalpr(country, configFile, runtimeDataDir);
    if (!openalpr.isLoaded()) {
        std::cerr << "Error loading OpenALPR" << std::endl;
        return;
    }

    while (!stop_thread) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCondVar.wait(lock, [] { return !frameQueue.empty() || stop_thread; });

        if (stop_thread && frameQueue.empty()) break;

        auto item = frameQueue.front();
        frameQueue.pop();
        lock.unlock();

        int index = item.first;
        cv::Mat frame = item.second;

        cv::Rect plateRect;
        cv::Mat plateRegion = extractPlateRegion(frame, plateRect);

        if (!plateRegion.empty()) {
            std::vector<unsigned char> buffer;
            cv::imencode(".jpg", plateRegion, buffer);
            std::vector<char> image_data(buffer.begin(), buffer.end());

            alpr::AlprResults results = openalpr.recognize(image_data);

            std::unique_lock<std::mutex> lockResults(queueMutex);
            framesVec[index] = frame;
            resultsVec[index] = results;
            lockResults.unlock();

            if (results.plates.size() > 0) {
                for (int i = 0; i < results.plates.size(); i++) {
                    alpr::AlprPlateResult plate = results.plates[i];
                    std::cout << "Camera " << index << " Plate: " << plate.bestPlate.characters 
                              << " Confidence: " << plate.bestPlate.overall_confidence << std::endl;

                    cv::rectangle(frame, plateRect, cv::Scalar(0, 255, 0), 2);
                    cv::putText(frame, plate.bestPlate.characters, 
                                cv::Point(plateRect.x, plateRect.y - 10), 
                                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);

                    // Araç sayımı için çizgi kontrolü
                    std::string plateNumber = plate.bestPlate.characters;
                    cv::Point currentPosition(plateRect.x + plateRect.width / 2, plateRect.y + plateRect.height / 2);

                    if (previousPositions.find(plateNumber) != previousPositions.end()) {
                        cv::Point previousPosition = previousPositions[plateNumber];

                        if (index == 0 && currentPosition.y < frame.rows / 2 && previousPosition.y >= frame.rows / 2) {
                            vehicleCountCamera0++;
                        } else if (index == 1 && currentPosition.y > frame.rows / 2 && previousPosition.y <= frame.rows / 2) {
                            vehicleCountCamera1++;
                        }
                    }

                    previousPositions[plateNumber] = currentPosition;
                }
            }
        }
    }
}

int main() {
    std::string country = "au"; // Avustralya
    std::string configFile = "/usr/local/share/openalpr/config/openalpr.conf";
    std::string runtimeDataDir = "/usr/local/share/openalpr/runtime_data";

    std::vector<alpr::AlprResults> resultsVec(2);
    std::vector<cv::Mat> framesVec(2);

    std::string rtsp_url1 = "rtsp://admin:alpDADE2@10.54.41.88:554";
    std::string rtsp_url2 = "rtsp://admin:alpDADE2@10.54.41.89:554";

    std::thread preProcessThread1(preProcessStream, rtsp_url1, 0); // RTSP URL 1
    std::thread preProcessThread2(preProcessStream, rtsp_url2, 1); // RTSP URL 2
    std::thread alprThread(alprProcess, country, configFile, runtimeDataDir, std::ref(resultsVec), std::ref(framesVec));

    while (!stop_thread) {
        for (int cameraIndex = 0; cameraIndex < 2; ++cameraIndex) {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (!framesVec[cameraIndex].empty()) {
                cv::putText(framesVec[cameraIndex], "Count: " + std::to_string(cameraIndex == 0 ? vehicleCountCamera0 : vehicleCountCamera1), 
                            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
                cv::imshow("Camera " + std::to_string(cameraIndex), framesVec[cameraIndex]);
            }
        }

        if (cv::waitKey(10) == 27) {
            stop_thread = true;
            queueCondVar.notify_all();
            break;
        }
    }

    preProcessThread1.join();
    preProcessThread2.join();
    alprThread.join();

    cv::destroyAllWindows();
    return 0;
}
