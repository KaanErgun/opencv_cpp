// g++ -std=c++17 cow_detection_file.cpp -o cow_detection_file.out -I/usr/include/opencv4 -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_dnn

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

float iou(cv::Rect box1, cv::Rect box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    int intersection = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = box1.area() + box2.area() - intersection;
    return (float)intersection / unionArea;
}

int main() {
    std::string modelConfiguration = "../yolov7/yolov7-tiny.cfg";
    std::string modelWeights = "../yolov7/yolov7-tiny.weights";
    std::string classesFile = "../yolov7/coco.names";

    std::vector<std::string> classes;
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);

    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::VideoCapture cap("01.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video stream." << std::endl;
        return -1;
    }

    cv::namedWindow("YOLO Detection", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        for (auto& output : outs) {
            auto* data = (float*)output.data;
            for (int i = 0; i < output.rows; ++i, data += output.cols) {
                cv::Mat scores = output.row(i).colRange(5, output.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > 0.3) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
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
                        classIds.push_back(classIdPoint.x);
                    }
                }
            }
        }

        for (size_t i = 0; i < boxes.size(); ++i) {
            rectangle(frame, boxes[i], cv::Scalar(10, 255, 0), 3);
            std::ostringstream ss;
            ss << classes[classIds[i]] << ": " << std::fixed << std::setprecision(2) << confidences[i];
            std::string label = ss.str();
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(boxes[i].y, labelSize.height);
            rectangle(frame, cv::Point(boxes[i].x, top - labelSize.height - 10),
                      cv::Point(boxes[i].x + labelSize.width, top), cv::Scalar(10, 255, 0), cv::FILLED);
            putText(frame, label, cv::Point(boxes[i].x, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        cv::imshow("YOLO Detection", frame);
        if (cv::waitKey(30) >= 0) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}