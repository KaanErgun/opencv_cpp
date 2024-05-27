// g++ -std=c++11 -o car_detection_dual.out car_detection_dual.cpp `pkg-config --cflags --libs opencv4` -L/usr/local/lib

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>

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
cv::Mat frame1, frame2;
cv::Mat mask1, mask2;
std::vector<cv::Point> roiPoints1, roiPoints2;

void processCamera(int cameraIndex, cv::dnn::Net& net, cv::Mat& frame, cv::Mat& mask) {
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera " << cameraIndex << std::endl;
        return;
    }

    while (true) {
        cv::Mat localFrame;
        cap >> localFrame;
        if (localFrame.empty()) break;

        // Apply mask
        cv::Mat maskedFrame;
        localFrame.copyTo(maskedFrame, mask);

        cv::Mat blob;
        cv::dnn::blobFromImage(maskedFrame, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
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
                    int centerX = (int)(data[0] * localFrame.cols);
                    int centerY = (int)(data[1] * localFrame.rows);
                    int width = (int)(data[2] * localFrame.cols);
                    int height = (int)(data[3] * localFrame.rows);
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

        for (size_t i = 0; i < boxes.size(); ++i) {
            rectangle(localFrame, boxes[i], cv::Scalar(0, 255, 0), 3);
            std::ostringstream ss;
            ss << "Car: " << std::fixed << std::setprecision(2) << confidences[i];
            std::string label = ss.str();
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(boxes[i].y, labelSize.height);
            rectangle(localFrame, cv::Point(boxes[i].x, top - labelSize.height - 10),
                      cv::Point(boxes[i].x + labelSize.width, top), cv::Scalar(0, 255, 0), cv::FILLED);
            putText(localFrame, label, cv::Point(boxes[i].x, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        {
            std::lock_guard<std::mutex> lock(frameMutex);
            frame = localFrame.clone();
        }
    }

    cap.release();
}

void createMask(cv::Mat& mask, const std::vector<cv::Point>& points) {
    mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(points);
    cv::fillPoly(mask, contours, cv::Scalar(255));
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

    // Define the regions of interest for each camera
    cv::Mat dummyFrame;
    cv::VideoCapture cap1(0), cap2(1);
    cap1 >> dummyFrame;
    roiPoints1 = {cv::Point(100, 100), cv::Point(100, 400), cv::Point(500, 400), cv::Point(500, 100)};
    cap2 >> dummyFrame;
    roiPoints2 = {cv::Point(50, 50), cv::Point(50, 450), cv::Point(550, 450), cv::Point(550, 50)};
    cap1.release();
    cap2.release();

    mask1 = cv::Mat(dummyFrame.size(), CV_8UC1);
    mask2 = cv::Mat(dummyFrame.size(), CV_8UC1);
    createMask(mask1, roiPoints1);
    createMask(mask2, roiPoints2);

    std::thread cam1Thread(processCamera, 0, std::ref(net1), std::ref(frame1), std::ref(mask1));
    std::thread cam2Thread(processCamera, 1, std::ref(net2), std::ref(frame2), std::ref(mask2));

    while (true) {
        cv::Mat displayFrame1, displayFrame2;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (!frame1.empty()) displayFrame1 = frame1.clone();
            if (!frame2.empty()) displayFrame2 = frame2.clone();
        }

        if (!displayFrame1.empty()) {
            cv::polylines(displayFrame1, roiPoints1, true, cv::Scalar(0, 0, 255), 2); // Draw ROI for camera 1
            cv::imshow("Camera 1", displayFrame1);
        }
        if (!displayFrame2.empty()) {
            cv::polylines(displayFrame2, roiPoints2, true, cv::Scalar(0, 0, 255), 2); // Draw ROI for camera 2
            cv::imshow("Camera 2", displayFrame2);
        }

        if (cv::waitKey(1) == 27) break; // Exit on ESC
    }

    cam1Thread.join();
    cam2Thread.join();

    cv::destroyAllWindows();
    return 0;
}
