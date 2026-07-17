#include "vision/config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace vision {

AppConfig AppConfig::fromFile(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    nlohmann::json j;
    ifs >> j;

    AppConfig c;
    c.source = j.value("source", c.source);
    c.detector = j.value("detector", c.detector);
    c.model = j.value("model", c.model);
    c.classNamesPath = j.value("class_names_path", c.classNamesPath);
    c.classNames = j.value("class_names", c.classNames);
    c.inputSize = j.value("input_size", c.inputSize);
    c.confThreshold = j.value("conf_threshold", c.confThreshold);
    c.nmsThreshold = j.value("nms_threshold", c.nmsThreshold);
    c.classFilter = j.value("class_filter", c.classFilter);
    c.backend = j.value("backend", c.backend);
    c.headless = j.value("headless", c.headless);
    c.outputPath = j.value("output_path", c.outputPath);
    c.maxFrames = j.value("max_frames", c.maxFrames);
    return c;
}

Backend AppConfig::parseBackend() const {
    if (backend == "cuda") {
        return Backend::CUDA;
    }
    if (backend == "opencl") {
        return Backend::OpenCL;
    }
    return Backend::CPU;
}

YoloConfig AppConfig::toYoloConfig() const {
    YoloConfig y;
    y.onnxPath = model;
    y.classNamesPath = classNamesPath;
    y.classNames = classNames;
    y.inputSize = inputSize;
    y.confThreshold = confThreshold;
    y.nmsThreshold = nmsThreshold;
    y.classFilter = classFilter;
    y.backend = parseBackend();
    return y;
}

}  // namespace vision
