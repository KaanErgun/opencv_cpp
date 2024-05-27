// g++ -std=c++11 -o car_detection_dual.out car_detection_dual.cpp `pkg-config --cflags --libs opencv4` -L/usr/local/lib

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <iomanip>
#include <chrono>

float iou(cv::Rect box1, cv::Rect box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    int intersection = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = box1.area() + box2.area() - intersection;
    return (float)intersection / unionArea;
}

std::mutex frameMutex;
std::mutex fileMutex;
cv::Mat frame1, frame2;
cv::Mat mask1, mask2;
int imageCounter = 0;
std::chrono::time_point<std::chrono::steady_clock> lastSavedTime = std::chrono::steady_clock::now();
int carCount1 = 0, carCount2 = 0;

// ROI indices for each camera (0-8, default to center)
int roiIndex1 = 4;
int roiIndex2 = 4;

void processCamera(int cameraIndex, cv::dnn::Net& net, cv::Mat& frame, int& carCount, int roiIndex) {
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera " << cameraIndex << std::endl;
        return;
    }

    std::vector<cv::Ptr<cv::Tracker>> trackers;
    std::vector<cv::Rect> trackedBoxes;
    std::vector<int> directions;

    while (true) {
        cv::Mat localFrame;
        cap >> localFrame;
        if (localFrame.empty()) break;

        // Calculate grid size
        int cellWidth = localFrame.cols / 3;
        int cellHeight = localFrame.rows / 3;

        // Determine the ROI based on roiIndex
        int roiRow = roiIndex / 3;
        int roiCol = roiIndex % 3;
        cv::Rect roi(roiCol * cellWidth, roiRow * cellHeight, cellWidth, cellHeight);

        // Draw the grid and ROI
        for (int i = 0; i < 4; ++i) {
            cv::line(localFrame, cv::Point(i * cellWidth, 0), cv::Point(i * cellWidth, localFrame.rows), cv::Scalar(0, 0, 255), 2);
            cv::line(localFrame, cv::Point(0, i * cellHeight), cv::Point(localFrame.cols, i * cellHeight), cv::Scalar(0, 0, 255), 2);
        }
        cv::rectangle(localFrame, roi, cv::Scalar(0, 255, 0), 2);

        // Extract the ROI from the frame
        cv::Mat roiFrame = localFrame(roi);

        cv::Mat blob;
        cv::dnn::blobFromImage(roiFrame, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        for (auto& output : outs) {
            auto* data = (float*)output.data;
            for (int i = 0; i < output.rows; ++i, data += output.cols) {
                cv::Mat scores = output.row(i).colRange(5, output.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > 0.5 && classIdPoint.x == 2) { // Only for cars
                    int centerX = (int)(data[0] * roiFrame.cols);
                    int centerY = (int)(data[1] * roiFrame.rows);
                    int width = (int)(data[2] * roiFrame.cols);
                    int height = (int)(data[3] * roiFrame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    cv::Rect box(left, top, width, height);
                    float nmsThreshold = 0.4;
                    bool keep = true;
                    for (int j = 0; j < boxes.size(); ++j) {
                        if (iou(box, boxes[j]) > nmsThreshold) {
                            keep = false;
                            break;
                        }
                    }
                    if (keep) {
                        boxes.push_back(box);
                        confidences.push_back(confidence);
                    }
                }
            }
        }

        // Update trackers
        for (size_t i = 0; i < trackers.size(); ++i) {
            trackers[i]->update(roiFrame, trackedBoxes[i]);
        }

        // Add new trackers for detected boxes
        for (const auto& box : boxes) {
            bool found = false;
            for (const auto& trackedBox : trackedBoxes) {
                if (iou(box, trackedBox) > 0.5) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
                tracker->init(roiFrame, box);
                trackers.push_back(tracker);
                trackedBoxes.push_back(box);
                directions.push_back(0);
            }
        }

        // Check for cars crossing the frame vertically
        for (size_t i = 0; i < trackedBoxes.size(); ++i) {
            if (directions[i] == 0) {
                if (trackedBoxes[i].y > roiFrame.rows / 2) {
                    directions[i] = 1; // Going down
                } else if (trackedBoxes[i].y < roiFrame.rows / 2) {
                    directions[i] = -1; // Going up
                }
            } else {
                if ((directions[i] == 1 && trackedBoxes[i].y > roiFrame.rows) ||
                    (directions[i] == -1 && trackedBoxes[i].y + trackedBoxes[i].height < 0)) {
                    carCount++;
                    trackers.erase(trackers.begin() + i);
                    trackedBoxes.erase(trackedBoxes.begin() + i);
                    directions.erase(directions.begin() + i);
                    i--;
                }
            }
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastSavedTime).count() >= 5 && !boxes.empty()) {
            std::lock_guard<std::mutex> fileLock(fileMutex);
            std::ostringstream fileName;
            fileName << "frame_" << cameraIndex << "_" << imageCounter++ << ".jpg";
            cv::imwrite(fileName.str(), localFrame);
            lastSavedTime = now;
        }

        for (size_t i = 0; i < boxes.size(); ++i) {
            std::ostringstream ss;
            ss << "Car: " << std::fixed << std::setprecision(2) << confidences[i];
            std::string label = ss.str();
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(boxes[i].y, labelSize.height);
            rectangle(roiFrame, cv::Point(boxes[i].x, top - labelSize.height - 10),
                      cv::Point(boxes[i].x + labelSize.width, top), cv::Scalar(0, 255, 0), cv::FILLED);
            putText(roiFrame, label, cv::Point(boxes[i].x, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // Display car count on frame
        std::ostringstream countLabel;
        countLabel << "Cars: " << carCount;
        putText(localFrame, countLabel.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        {
            std::lock_guard<std::mutex> lock(frameMutex);
            frame = localFrame.clone();
        }
    }

    cap.release();
}

int main() {
    std::string modelConfiguration = "../yolov3/yolov3.cfg";
    std::string modelWeights = "../yolov3/yolov3.weights"; 
    std::string classesFile = "../yolov3/coco.names";

    cv::dnn::Net net1 = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    cv::dnn::Net net2 = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net1.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net1.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    net2.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net2.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Define ROI indices for each camera (0-8)
    roiIndex1 = 4; // Default to center grid for camera 1
    roiIndex2 = 4; // Default to center grid for camera 2

    std::thread cam1Thread(processCamera, 0, std::ref(net1), std::ref(frame1), std::ref(carCount1), roiIndex1);
    std::thread cam2Thread(processCamera, 1, std::ref(net2), std::ref(frame2), std::ref(carCount2), roiIndex2);

    while (true) {
        cv::Mat displayFrame1, displayFrame2;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (!frame1.empty()) displayFrame1 = frame1.clone();
            if (!frame2.empty()) displayFrame2 = frame2.clone();
        }

        if (!displayFrame1.empty()) {
            cv::imshow("Camera 1", displayFrame1);
        }
        if (!displayFrame2.empty()) {
            cv::imshow("Camera 2", displayFrame2);
        }

        if (cv::waitKey(1) == 27) break; // Exit on ESC
    }

    cam1Thread.join();
    cam2Thread.join();

    cv::destroyAllWindows();
    return 0;
}
