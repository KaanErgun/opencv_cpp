#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream> // Dosya işlemleri için gereken başlık

// Function to get the output layer names in the architecture
std::vector<std::string> getOutputsNames(const cv::dnn::Net& net) {
    static std::vector<std::string> names;
    if (names.empty()) {
        // Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        // get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();
        // Get the names of the output layers in the architecture
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

// Post-processing function to handle YOLO output (Only detecting humans - class ID 0)
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, const std::vector<std::string>& classNames, float confThreshold = 0.5, float nmsThreshold = 0.4) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                // Only consider humans (class ID 0)
                if (classIdPoint.x == 0) {
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    }

    // Non-maxima suppression to remove overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

        // Optionally: Display "Person" label on the box
        std::string label = cv::format("Person: %.2f", confidences[idx]);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        // 'top' değişkeni burada tanımlanmalı
        int top = std::max(box.y, labelSize.height);

        cv::rectangle(frame, cv::Point(box.x, top - labelSize.height),
            cv::Point(box.x + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
        cv::putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

int main() {
    // Load YOLOv3 model
    std::string configPath = "/Users/kaanergun/opencv_cpp/yolov3/yolov3.cfg"; // Tam dosya yolu
    std::string modelPath = "/Users/kaanergun/opencv_cpp/yolov3/yolov3.weights"; // Tam dosya yolu
    std::string classesFile = "/Users/kaanergun/opencv_cpp/yolov3/coco.names"; // Tam dosya yolu

    // Load class names
    std::vector<std::string> classNames;
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) classNames.push_back(line);

    // Load the network
    cv::dnn::Net net = cv::dnn::readNet(modelPath, configPath);

    // Set preferable backend and target
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Open video capture (camera)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video capture." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cv::waitKey(1) < 0) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Empty frame captured." << std::endl;
            break;
        }

        // Prepare the frame for YOLO
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // Run forward pass to get outputs
        std::vector<cv::Mat> outs;
        net.forward(outs, getOutputsNames(net));

        // Post-process the output to get human detections
        postprocess(frame, outs, classNames);

        // Show the frame
        cv::imshow("YOLOv3 Human Detection", frame);
    }

    return 0;
}
