// g++ -std=c++11 -o car_detection_dual.out car_detection_dual.cpp `pkg-config --cflags --libs opencv4` -L/usr/local/lib

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <vector>
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
int imageCounter = 0;
std::chrono::time_point<std::chrono::steady_clock> lastSavedTime = std::chrono::steady_clock::now();
int carCount1 = 0, carCount2 = 0;

// ROI binary array (0-8) for each camera
bool roiBinaryArray1[9] = {false, false, false, false, true, false, false, false, false}; // Center region for camera 1
bool roiBinaryArray2[9] = {false, false, false, true, true, true, false, false, false};  // Center and middle row for camera 2

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

void updateTrackers(std::vector<cv::Ptr<cv::Tracker>>& trackers, std::vector<cv::Rect>& trackedBoxes, const cv::Mat& roiFrame) {
    try {
        for (size_t i = 0; i < trackers.size(); ++i) {
            if (!trackers[i]->update(roiFrame, trackedBoxes[i])) {
                trackers.erase(trackers.begin() + i);
                trackedBoxes.erase(trackedBoxes.begin() + i);
                --i;
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error in updateTrackers: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception in updateTrackers: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error in updateTrackers." << std::endl;
    }
}

void addNewTrackers(std::vector<cv::Ptr<cv::Tracker>>& trackers, std::vector<cv::Rect>& trackedBoxes, const std::vector<cv::Rect>& boxes, const cv::Mat& roiFrame) {
    try {
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
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error in addNewTrackers: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception in addNewTrackers: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error in addNewTrackers." << std::endl;
    }
}

void processDetections(const std::vector<cv::Mat>& outs, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, const cv::Mat& roiFrame) {
    try {
        for (const auto& output : outs) {
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
                    for (const auto& existingBox : boxes) {
                        if (iou(box, existingBox) > nmsThreshold) {
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
    } catch (const cv::Exception& e) {
        std::cerr << "Error in processDetections: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception in processDetections: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error in processDetections." << std::endl;
    }
}

void checkCarStatus(std::vector<cv::Ptr<cv::Tracker>>& trackers, std::vector<cv::Rect>& trackedBoxes, std::vector<int>& carStatus, int& carCount, const cv::Rect& roi) {
    try {
        for (size_t i = 0; i < trackedBoxes.size(); ++i) {
            cv::Rect absoluteBox = trackedBoxes[i] + cv::Point(roi.x, roi.y); // Convert relative ROI coordinates to absolute coordinates

            if (roi.contains(absoluteBox.tl()) && roi.contains(absoluteBox.br())) {
                if (carStatus[i] == 0) {
                    carStatus[i] = 1; // Car entered ROI
                }
            } else {
                if (carStatus[i] == 1) {
                    carStatus[i] = 2; // Car exited ROI
                    carCount++;
                    trackers.erase(trackers.begin() + i);
                    trackedBoxes.erase(trackedBoxes.begin() + i);
                    carStatus.erase(carStatus.begin() + i);
                    i--;
                }
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error in checkCarStatus: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception in checkCarStatus: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error in checkCarStatus." << std::endl;
    }
}

void processCamera(int cameraIndex, cv::dnn::Net& net, cv::Mat& frame, int& carCount, bool roiBinaryArray[9]) {
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera " << cameraIndex << std::endl;
        return;
    }

    std::vector<cv::Ptr<cv::Tracker>> trackers;
    std::vector<cv::Rect> trackedBoxes;
    std::vector<int> carStatus; // 0: not in ROI, 1: in ROI, 2: exited ROI

    while (true) {
        try {
            cv::Mat localFrame;
            cap >> localFrame;
            if (localFrame.empty()) break;

            // Calculate grid size
            int cellWidth = localFrame.cols / 3;
            int cellHeight = localFrame.rows / 3;

            drawGridAndROI(localFrame, roiBinaryArray);

            bool detectedCar = false;

            for (int i = 0; i < 9; ++i) {
                if (!roiBinaryArray[i]) continue;

                int roiRow = i / 3;
                int roiCol = i % 3;
                cv::Rect roi(roiCol * cellWidth, roiRow * cellHeight, cellWidth, cellHeight);

                // Extract the ROI from the frame
                cv::Mat roiFrame = localFrame(roi);

                cv::Mat blob;
                cv::dnn::blobFromImage(roiFrame, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
                net.setInput(blob);

                std::vector<cv::Mat> outs;
                net.forward(outs, net.getUnconnectedOutLayersNames());

                std::vector<cv::Rect> boxes;
                std::vector<float> confidences;
                processDetections(outs, boxes, confidences, roiFrame);
                updateTrackers(trackers, trackedBoxes, roiFrame);
                addNewTrackers(trackers, trackedBoxes, boxes, roiFrame);
                checkCarStatus(trackers, trackedBoxes, carStatus, carCount, roi);

                if (!boxes.empty()) {
                    detectedCar = true;
                }
            }

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - lastSavedTime).count() >= 5 && detectedCar) {
                std::lock_guard<std::mutex> fileLock(fileMutex);
                std::ostringstream fileName;
                fileName << "frame_" << cameraIndex << "_" << imageCounter++ << ".jpg";
                cv::imwrite(fileName.str(), localFrame);
                lastSavedTime = now;
            }

            // Display car count on frame
            std::ostringstream countLabel;
            countLabel << "Cars: " << carCount;
            putText(localFrame, countLabel.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            {
                std::lock_guard<std::mutex> lock(frameMutex);
                frame = localFrame.clone();
            }
        } catch (const cv::Exception& e) {
            std::cerr << "Error in processCamera: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Standard exception in processCamera: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown error in processCamera." << std::endl;
        }
    }

    cap.release();
}

int main() {
    try {
        std::string modelConfiguration = "../yolov3/yolov3.cfg";
        std::string modelWeights = "../yolov3/yolov3.weights"; 
        std::string classesFile = "../yolov3/coco.names";

        cv::dnn::Net net1 = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
        cv::dnn::Net net2 = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
        net1.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net1.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        net2.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net2.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        std::thread cam1Thread(processCamera, 0, std::ref(net1), std::ref(frame1), std::ref(carCount1), roiBinaryArray1);
        std::thread cam2Thread(processCamera, 1, std::ref(net2), std::ref(frame2), std::ref(carCount2), roiBinaryArray2);

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
    } catch (const cv::Exception& e) {
        std::cerr << "Error in main: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception in main: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error in main." << std::endl;
    }

    return 0;
}
