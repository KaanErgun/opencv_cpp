#include "vision/yolo_detector.hpp"

#include <algorithm>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace vision {
namespace {

std::vector<std::string> loadNames(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open class names file: " + path);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        names.push_back(line);
    }
    return names;
}

// Letterbox: resize keeping aspect ratio, pad to a square with a constant border.
// Returns the padded image and fills scale + padding so boxes can be un-mapped.
cv::Mat letterbox(const cv::Mat& src, int size, float& scale, int& padX, int& padY) {
    scale = std::min(static_cast<float>(size) / src.cols,
                     static_cast<float>(size) / src.rows);
    const int resizedW = static_cast<int>(std::round(src.cols * scale));
    const int resizedH = static_cast<int>(std::round(src.rows * scale));
    padX = (size - resizedW) / 2;
    padY = (size - resizedH) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resizedW, resizedH));

    cv::Mat out(size, size, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(padX, padY, resizedW, resizedH)));
    return out;
}

}  // namespace

YoloDetector::YoloDetector(const YoloConfig& config) : config_(config) {
    if (!config_.classNamesPath.empty()) {
        classNames_ = loadNames(config_.classNamesPath);
    } else {
        classNames_ = config_.classNames;
    }

    try {
        net_ = cv::dnn::readNetFromONNX(config_.onnxPath);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Failed to load ONNX model '" + config_.onnxPath +
                                 "': " + e.what());
    }
    if (net_.empty()) {
        throw std::runtime_error("Loaded an empty ONNX network: " + config_.onnxPath);
    }

    switch (config_.backend) {
        case Backend::CUDA:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            break;
        case Backend::OpenCL:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
            break;
        case Backend::CPU:
        default:
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            break;
    }
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {
    if (frame.empty()) {
        return {};
    }

    float scale = 1.0F;
    int padX = 0;
    int padY = 0;
    const cv::Mat input = letterbox(frame, config_.inputSize, scale, padX, padY);

    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.0 / 255.0,
                           cv::Size(config_.inputSize, config_.inputSize), cv::Scalar(),
                           true, false);
    net_.setInput(blob);

    cv::Mat output = net_.forward();  // YOLOv8/v11: shape [1, 4+nc, 8400]

    // Collapse the batch dim and transpose to [8400, 4+nc] so each row is one box.
    const int dimensions = output.size[1];
    const int anchors = output.size[2];
    output = output.reshape(1, dimensions);  // [4+nc, 8400]
    cv::transpose(output, output);           // [8400, 4+nc]

    const int numClasses = dimensions - 4;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    boxes.reserve(anchors);
    confidences.reserve(anchors);
    classIds.reserve(anchors);

    for (int i = 0; i < anchors; ++i) {
        const float* row = output.ptr<float>(i);
        const cv::Mat scores(1, numClasses, CV_32F, const_cast<float*>(row + 4));
        cv::Point classId;
        double maxScore = 0.0;
        cv::minMaxLoc(scores, nullptr, &maxScore, nullptr, &classId);

        if (maxScore < config_.confThreshold) {
            continue;
        }
        if (!config_.classFilter.empty() &&
            std::find(config_.classFilter.begin(), config_.classFilter.end(),
                      classId.x) == config_.classFilter.end()) {
            continue;
        }

        // Un-map from letterboxed 640x640 space back to original frame pixels.
        const float cx = (row[0] - padX) / scale;
        const float cy = (row[1] - padY) / scale;
        const float w = row[2] / scale;
        const float h = row[3] / scale;

        const int left = static_cast<int>(std::round(cx - w / 2.0F));
        const int top = static_cast<int>(std::round(cy - h / 2.0F));
        boxes.emplace_back(left, top, static_cast<int>(std::round(w)),
                           static_cast<int>(std::round(h)));
        confidences.push_back(static_cast<float>(maxScore));
        classIds.push_back(classId.x);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxesBatched(boxes, confidences, classIds, config_.confThreshold,
                             config_.nmsThreshold, keep);

    const cv::Rect frameRect(0, 0, frame.cols, frame.rows);
    std::vector<Detection> detections;
    detections.reserve(keep.size());
    for (const int idx : keep) {
        Detection det;
        det.box = boxes[idx] & frameRect;  // clamp to frame bounds
        det.classId = classIds[idx];
        det.confidence = confidences[idx];
        if (det.box.area() > 0) {
            detections.push_back(det);
        }
    }
    return detections;
}

}  // namespace vision
