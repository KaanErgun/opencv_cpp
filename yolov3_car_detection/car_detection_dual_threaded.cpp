#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>
#include <fstream>

// Constants
const int FRAME_SIZE = 500;
const int FPS = 1;
const int YOLO_SIZE = 416;
const float CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;

// Global variables
std::mutex frameMutex;
cv::Mat frame1, frame2;
bool running = true;

// ROI binary arrays for each camera
bool roiBinaryArray1[9] = {false, false, false, false, true, false, false, false, false}; // Center region for camera 1
bool roiBinaryArray2[9] = {false, false, false, true, true, true, false, false, false};  // Center and middle row for camera 2

// Load class names from coco.names file
std::vector<std::string> loadClassNames(const std::string& file) {
    std::vector<std::string> classNames;
    std::ifstream ifs(file.c_str());
    std::string line;
    while (getline(ifs, line)) {
        classNames.push_back(line);
    }
    return classNames;
}

// Function to draw grid and highlight ROI
void drawGridAndROI(cv::Mat& frame, const bool roiBinaryArray[9]) {
    int cellWidth = frame.cols / 3;
    int cellHeight = frame.rows / 3;

    for (int i = 0; i < 4; ++i) {
        cv::line(frame, cv::Point(i * cellWidth, 0), cv::Point(i * cellWidth, frame.rows), cv::Scalar(0, 0, 255), 2);
        cv::line(frame, cv::Point(0, i * cellHeight), cv::Point(frame.cols, i * cellHeight), cv::Scalar(0, 0, 255), 2);
    }

    for (int i = 0; i < 9; ++i) {
        if (roiBinaryArray[i]) {
            int roiRow = i / 3;
            int roiCol = i % 3;
            cv::Rect roi(roiCol * cellWidth, roiRow * cellHeight, cellWidth, cellHeight);
            cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
        }
    }
}

// Function to mask non-ROI areas
cv::Mat getROIMaskedFrame(const cv::Mat& frame, const bool roiBinaryArray[9]) {
    int cellWidth = frame.cols / 3;
    int cellHeight = frame.rows / 3;
    cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());

    for (int i = 0; i < 9; ++i) {
        if (roiBinaryArray[i]) {
            int roiRow = i / 3;
            int roiCol = i % 3;
            cv::Rect roi(roiCol * cellWidth, roiRow * cellHeight, cellWidth, cellHeight);
            frame(roi).copyTo(mask(roi));
        }
    }

    return mask;
}

// Function to capture and process frames from a camera
void captureFrames(int cameraIndex, cv::VideoCapture& cap) {
    while (running) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Crop the frame to a 1:1 aspect ratio
        int cropSize = std::min(frame.cols, frame.rows);
        cv::Rect cropRegion((frame.cols - cropSize) / 2, (frame.rows - cropSize) / 2, cropSize, cropSize);
        cv::Mat croppedFrame = frame(cropRegion);

        // Resize the frame to 500x500
        cv::resize(croppedFrame, croppedFrame, cv::Size(FRAME_SIZE, FRAME_SIZE));

        // Draw grid and highlight ROI
        if (cameraIndex == 0) {
            drawGridAndROI(croppedFrame, roiBinaryArray1);
        } else if (cameraIndex == 1) {
            drawGridAndROI(croppedFrame, roiBinaryArray2);
        }

        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (cameraIndex == 0) {
                frame1 = croppedFrame.clone();
            } else if (cameraIndex == 1) {
                frame2 = croppedFrame.clone();
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / FPS));
    }
}

// Function to process frames using YOLO v3
void processFrames(cv::dnn::Net& net, const std::vector<std::string>& outputLayers, cv::Mat& frame, const std::vector<std::string>& classNames, const bool roiBinaryArray[9]) {
    cv::Mat roiMaskedFrame = getROIMaskedFrame(frame, roiBinaryArray);

    cv::Mat blob;
    cv::dnn::blobFromImage(roiMaskedFrame, blob, 1 / 255.0, cv::Size(YOLO_SIZE, YOLO_SIZE), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, outputLayers);

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& output : outputs) {
        for (int i = 0; i < output.rows; i++) {
            const auto* data = reinterpret_cast<const float*>(output.data) + i * output.cols;
            float confidence = data[4];
            if (confidence >= CONFIDENCE_THRESHOLD) {
                cv::Mat scores = output.row(i).colRange(5, output.cols);
                cv::Point classIdPoint;
                double maxClassScore;
                cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
                if (maxClassScore > CONFIDENCE_THRESHOLD && classIdPoint.x == 2) { // Only for cars (classIdPoint.x == 2)
                    int centerX = static_cast<int>(data[0] * frame.cols);
                    int centerY = static_cast<int>(data[1] * frame.rows);
                    int width = static_cast<int>(data[2] * frame.cols);
                    int height = static_cast<int>(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back(static_cast<float>(maxClassScore));
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        std::string label = classNames[classIds[idx]] + ": " + cv::format("%.2f", confidences[idx]);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int topAdjusted = std::max(box.y, labelSize.height);
        cv::rectangle(frame, cv::Point(box.x, topAdjusted - labelSize.height),
                      cv::Point(box.x + labelSize.width, topAdjusted + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(frame, label, cv::Point(box.x, topAdjusted), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// Function to display frames
void displayFrames(cv::dnn::Net& net, const std::vector<std::string>& outputLayers, const std::vector<std::string>& classNames) {
    while (running) {
        cv::Mat displayFrame1, displayFrame2;

        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (!frame1.empty()) {
                displayFrame1 = frame1.clone();
            }
            if (!frame2.empty()) {
                displayFrame2 = frame2.clone();
            }
        }

        if (!displayFrame1.empty()) {
            processFrames(net, outputLayers, displayFrame1, classNames, roiBinaryArray1);
            cv::imshow("Camera 0", displayFrame1);
        }
        if (!displayFrame2.empty()) {
            processFrames(net, outputLayers, displayFrame2, classNames, roiBinaryArray2);
            cv::imshow("Camera 1", displayFrame2);
        }

        if (cv::waitKey(1000 / FPS) == 27) {
            running = false;
            break; // Exit on ESC
        }
    }
}

int main() {
    try {
        cv::VideoCapture cap1(0);
        cv::VideoCapture cap2(1);
        if (!cap1.isOpened() || !cap2.isOpened()) {
            std::cerr << "Error opening video streams." << std::endl;
            return -1;
        }

        // Load YOLO v3 model
        std::string modelConfiguration = "../yolov3/yolov3.cfg";
        std::string modelWeights = "../yolov3/yolov3.weights";
        std::string classNamesFile = "../yolov3/coco.names";
        cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        std::vector<std::string> classNames = loadClassNames(classNamesFile);
        std::vector<std::string> outputLayers = net.getUnconnectedOutLayersNames();

        std::thread captureThread1(captureFrames, 0, std::ref(cap1));
        std::thread captureThread2(captureFrames, 1, std::ref(cap2));

        // Display frames on the main thread
        displayFrames(net, outputLayers, classNames);

        captureThread1.join();
        captureThread2.join();

        cv::destroyAllWindows();
    } catch (const cv::Exception& e) {
        std::cerr << "Error in main: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception in main: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error in main." << std::endl;
    }

    return 0;
}
