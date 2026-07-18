// app_alpr_server — HTTP license-plate recognition service.
//
// Loads best.onnx + Tesseract once, then serves:
//   GET  /health          -> {status, detector, ocr, uptime_s}
//   POST /recognize        -> body = image bytes (or multipart "image");
//                             returns {plates:[{text,ocr_confidence,det_confidence,box}],
//                             count, ms}
//   GET  /events?limit=N   -> last N recognised plates from the event log
//   GET  /                 -> web dashboard (upload an image, see results)
//
// Every read with text is appended to a JSON-lines event log so history
// survives restarts. See app_alpr_client for the capture-side counterpart.
//
// Usage: app_alpr_server --config configs/alpr_server.json [--port 8080]

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

#include "dashboard.hpp"
#include "vision/alpr_pipeline.hpp"
#include "vision/cli.hpp"

using json = nlohmann::json;

namespace {

std::string nowIso() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

// Turns one recognition into a JSON object (also used for the event log).
json plateToJson(const vision::PlateResult& p) {
    return json{{"text", p.text},
                {"ocr_confidence", p.ocrConfidence},
                {"det_confidence", p.detConfidence},
                {"box", {p.box.x, p.box.y, p.box.width, p.box.height}}};
}

vision::AlprConfig loadAlprConfig(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }
    json j;
    ifs >> j;

    vision::AlprConfig cfg;
    cfg.detector.onnxPath = j.value("model", std::string("models/best.onnx"));
    cfg.detector.classNames =
        j.value("class_names", std::vector<std::string>{"Plaka", "Araba"});
    cfg.detector.classFilter = j.value("class_filter", std::vector<int>{0});
    cfg.detector.confThreshold = j.value("conf_threshold", 0.3F);
    cfg.detector.inputSize = j.value("input_size", 640);
    cfg.plateClassId = j.value("plate_class_id", 0);
    cfg.minOcrConfidence = j.value("min_ocr_confidence", 0.3F);

    if (j.contains("ocr")) {
        const auto& o = j.at("ocr");
        cfg.ocr.whitelist = o.value("whitelist", cfg.ocr.whitelist);
        cfg.ocr.psm = o.value("psm", cfg.ocr.psm);
        cfg.ocr.upscaleHeight = o.value("upscale_height", cfg.ocr.upscaleHeight);
        cfg.ocr.minChars = o.value("min_chars", cfg.ocr.minChars);
        cfg.ocr.tessdataPath = o.value("tessdata_path", cfg.ocr.tessdataPath);
    }
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string configPath =
        vision::cli::argOr(argc, argv, "--config", "configs/alpr_server.json");

    try {
        std::ifstream cfgFile(configPath);
        json rawCfg;
        if (cfgFile.is_open()) {
            cfgFile >> rawCfg;
        }
        const int port =
            vision::cli::argInt(argc, argv, "--port", rawCfg.value("port", 8080));
        const std::string host =
            vision::cli::argOr(argc, argv, "--host", rawCfg.value("host", "0.0.0.0"));
        const std::string eventsPath =
            rawCfg.value("events_path", std::string("alpr_events.jsonl"));

        vision::AlprPipeline pipeline(loadAlprConfig(configPath));
        std::mutex eventsMtx;
        const auto startTime = std::chrono::steady_clock::now();

        httplib::Server svr;

        svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(vision::kDashboardHtml, "text/html");
        });

        svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
            const auto up = std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::steady_clock::now() - startTime)
                                .count();
            res.set_content(json{{"status", "ok"},
                                 {"detector", pipeline.config().detector.onnxPath},
                                 {"ocr", "tesseract"},
                                 {"uptime_s", up}}
                                .dump(),
                            "application/json");
        });

        svr.Post("/recognize", [&](const httplib::Request& req, httplib::Response& res) {
            // Accept a raw image body or a multipart "image" file field.
            std::string bytes = req.body;
            if (req.has_file("image")) {
                bytes = req.get_file_value("image").content;
            }
            if (bytes.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "empty body"}}.dump(), "application/json");
                return;
            }

            const std::vector<uchar> buf(bytes.begin(), bytes.end());
            const cv::Mat frame = cv::imdecode(buf, cv::IMREAD_COLOR);
            if (frame.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "could not decode image"}}.dump(),
                                "application/json");
                return;
            }

            const auto t0 = std::chrono::steady_clock::now();
            const auto plates = pipeline.recognize(frame);
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();

            json arr = json::array();
            for (const auto& p : plates) {
                arr.push_back(plateToJson(p));
            }

            // Log only successful reads (non-empty text) to the event history.
            {
                std::lock_guard<std::mutex> lock(eventsMtx);
                std::ofstream log(eventsPath, std::ios::app);
                for (const auto& p : plates) {
                    if (!p.text.empty()) {
                        json e = plateToJson(p);
                        e["time"] = nowIso();
                        log << e.dump() << '\n';
                    }
                }
            }

            res.set_content(
                json{{"plates", arr}, {"count", plates.size()}, {"ms", ms}}.dump(),
                "application/json");
        });

        svr.Get("/events", [&](const httplib::Request& req, httplib::Response& res) {
            int limit = 50;
            if (req.has_param("limit")) {
                limit = std::max(1, std::atoi(req.get_param_value("limit").c_str()));
            }
            std::vector<std::string> lines;
            {
                std::lock_guard<std::mutex> lock(eventsMtx);
                std::ifstream log(eventsPath);
                std::string line;
                while (std::getline(log, line)) {
                    if (!line.empty()) {
                        lines.push_back(line);
                    }
                }
            }
            json arr = json::array();
            const size_t start =
                lines.size() > static_cast<size_t>(limit) ? lines.size() - limit : 0;
            for (size_t i = start; i < lines.size(); ++i) {
                try {
                    arr.push_back(json::parse(lines[i]));
                } catch (...) {
                }
            }
            res.set_content(json{{"events", arr}, {"count", arr.size()}}.dump(),
                            "application/json");
        });

        std::cout << "ALPR server listening on http://" << host << ":" << port
                  << "  (dashboard at /, POST images to /recognize)\n";
        if (!svr.listen(host, port)) {
            std::cerr << "Fatal: could not bind " << host << ":" << port << '\n';
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
