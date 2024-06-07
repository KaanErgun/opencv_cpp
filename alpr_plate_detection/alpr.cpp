// g++ -std=c++11 -I/usr/local/include/openalpr -o alpr.out alpr.cpp `pkg-config --cflags --libs opencv4` -L/usr/local/lib -lopenalpr -Wl,-rpath,/usr/local/lib

#include <alpr.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <iostream>
#include <vector>

std::atomic<bool> stop_thread(false);

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

cv::Mat extractPlateRegion(const cv::Mat& frame) {
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

    cv::Rect largestPlate = *std::max_element(possiblePlates.begin(), possiblePlates.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.area() < b.area();
    });

    return frame(largestPlate);
}

void processStream(const std::string& rtsp_url, const std::string& country, const std::string& configFile, const std::string& runtimeDataDir, std::vector<alpr::AlprResults>& resultsVec, std::vector<cv::Mat>& framesVec, int index) {
    cv::VideoCapture cap(rtsp_url);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream: " << rtsp_url << std::endl;
        return;
    }

    alpr::Alpr openalpr(country, configFile, runtimeDataDir);
    if (!openalpr.isLoaded()) {
        std::cerr << "Error loading OpenALPR" << std::endl;
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
            cv::Mat plateRegion = extractPlateRegion(processedFrame);

            if (!plateRegion.empty()) {
                std::vector<unsigned char> buffer;
                cv::imencode(".jpg", plateRegion, buffer);
                std::vector<char> image_data(buffer.begin(), buffer.end());

                alpr::AlprResults results = openalpr.recognize(image_data);

                framesVec[index] = plateRegion;
                resultsVec[index] = results;
            }
        }

        frameCounter++;
    }

    cap.release();
}

int main() {
    std::string country = "au"; // Avustralya
    std::string configFile = "/openalpr.conf";
    std::string runtimeDataDir = "/usr/local/share/openalpr/runtime_data";

    std::vector<alpr::AlprResults> resultsVec(2);
    std::vector<cv::Mat> framesVec(2);

    std::string rtsp_url1 = "rtsp://admin:alpDADE2@10.54.41.88:554";
    std::string rtsp_url2 = "rtsp://admin:alpDADE2@10.54.41.89:554";

    std::thread thread1(processStream, rtsp_url1, country, configFile, runtimeDataDir, std::ref(resultsVec), std::ref(framesVec), 0); // RTSP URL 1
    std::thread thread2(processStream, rtsp_url2, country, configFile, runtimeDataDir, std::ref(resultsVec), std::ref(framesVec), 1); // RTSP URL 2

    while (!stop_thread) {
        for (int cameraIndex = 0; cameraIndex < 2; ++cameraIndex) {
            if (!framesVec[cameraIndex].empty()) {
                cv::Mat frame = framesVec[cameraIndex];
                alpr::AlprResults results = resultsVec[cameraIndex];

                if (results.plates.size() > 0) {
                    for (int i = 0; i < results.plates.size(); i++) {
                        alpr::AlprPlateResult plate = results.plates[i];
                        std::cout << "Camera " << cameraIndex << " Plate: " << plate.bestPlate.characters 
                                  << " Confidence: " << plate.bestPlate.overall_confidence << std::endl;

                        cv::rectangle(frame, 
                                      cv::Rect(plate.plate_points[0].x, plate.plate_points[0].y, 
                                               plate.plate_points[2].x - plate.plate_points[0].x, 
                                               plate.plate_points[2].y - plate.plate_points[0].y), 
                                      cv::Scalar(0, 255, 0), 2);

                        cv::putText(frame, plate.bestPlate.characters, 
                                    cv::Point(plate.plate_points[0].x, plate.plate_points[0].y - 10), 
                                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
                    }
                } else {
                    //std::cout << "Camera " << cameraIndex << " No plates detected." << std::endl;
                }

                cv::imshow("Camera " + std::to_string(cameraIndex), frame);
            }
        }

        if (cv::waitKey(10) == 27) {
            stop_thread = true;
            break;
        }
    }

    thread1.join();
    thread2.join();

    cv::destroyAllWindows();
    return 0;
}
