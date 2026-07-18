#include "vision/plate_ocr.hpp"

#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>

#include <cctype>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace vision {
namespace {

// Picks a tessdata directory: explicit config wins, then a documented search
// list (covers Homebrew intel/arm and apt), then let Tesseract use its default
// (TESSDATA_PREFIX / compiled path) by returning empty.
std::string resolveTessdata(const std::string& configured) {
    if (!configured.empty()) {
        return configured;
    }
    const char* candidates[] = {
        "/usr/local/share/tessdata",
        "/usr/local/Cellar/tesseract/5.5.2/share/tessdata",
        "/opt/homebrew/share/tessdata",
        "/usr/share/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
    };
    for (const char* dir : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(std::string(dir) + "/eng.traineddata", ec)) {
            return dir;
        }
    }
    return {};  // fall back to Tesseract's own resolution
}

}  // namespace

struct PlateOcr::Impl {
    tesseract::TessBaseAPI api;
};

PlateOcr::PlateOcr(const OcrConfig& cfg) : impl_(std::make_unique<Impl>()), cfg_(cfg) {
    const std::string tessdata = resolveTessdata(cfg_.tessdataPath);
    const char* dataPath = tessdata.empty() ? nullptr : tessdata.c_str();
    if (impl_->api.Init(dataPath, cfg_.language.c_str()) != 0) {
        throw std::runtime_error(
            "Failed to initialise Tesseract (language '" + cfg_.language +
            "'). Install the language data or set OcrConfig.tessdataPath / the "
            "TESSDATA_PREFIX environment variable.");
    }
    impl_->api.SetPageSegMode(static_cast<tesseract::PageSegMode>(cfg_.psm));
    if (!cfg_.whitelist.empty()) {
        impl_->api.SetVariable("tessedit_char_whitelist", cfg_.whitelist.c_str());
    }
}

PlateOcr::~PlateOcr() {
    if (impl_) {
        impl_->api.End();
    }
}

PlateOcr::PlateOcr(PlateOcr&&) noexcept = default;
PlateOcr& PlateOcr::operator=(PlateOcr&&) noexcept = default;

cv::Mat PlateOcr::preprocess(const cv::Mat& plateCrop) const {
    if (plateCrop.empty()) {
        return {};
    }

    cv::Mat gray;
    if (plateCrop.channels() == 3) {
        cv::cvtColor(plateCrop, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = plateCrop.clone();
    }

    // Upscale so characters are big enough for Tesseract (it prefers ~30+ px
    // tall glyphs). Cubic keeps edges reasonably sharp.
    if (gray.rows > 0 && gray.rows < cfg_.upscaleHeight) {
        const double scale = static_cast<double>(cfg_.upscaleHeight) / gray.rows;
        cv::resize(gray, gray, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }

    // Local contrast boost (CLAHE) rescues characters under uneven lighting /
    // glare before thresholding — this measurably improves reads on dim plates.
    cv::createCLAHE(2.0, cv::Size(8, 8))->apply(gray, gray);

    // Light denoise, then Otsu binarisation (adapts the threshold to the crop).
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Tesseract wants dark text on a light background. If the crop came out
    // mostly black (light text on dark plate), invert so polarity is correct.
    if (static_cast<size_t>(cv::countNonZero(bin)) < bin.total() / 2) {
        cv::bitwise_not(bin, bin);
    }

    // A quiet margin helps the recogniser near the plate edges.
    cv::copyMakeBorder(bin, bin, 12, 12, 12, 12, cv::BORDER_CONSTANT, cv::Scalar(255));
    return bin;
}

OcrResult PlateOcr::read(const cv::Mat& plateCrop) const {
    const cv::Mat bin = preprocess(plateCrop);
    if (bin.empty()) {
        return {};
    }

    impl_->api.SetImage(bin.data, bin.cols, bin.rows, 1, static_cast<int>(bin.step));
    impl_->api.SetSourceResolution(300);
    impl_->api.Recognize(nullptr);

    std::unique_ptr<char[], void (*)(char*)> raw(impl_->api.GetUTF8Text(),
                                                 [](char* p) { delete[] p; });

    OcrResult result;
    if (raw) {
        for (const char* p = raw.get(); *p != '\0'; ++p) {
            const unsigned char c = static_cast<unsigned char>(*p);
            if (std::isalnum(c) != 0) {
                result.text.push_back(static_cast<char>(std::toupper(c)));
            }
        }
    }

    // Average the per-symbol confidences from the result iterator. This is more
    // reliable than MeanTextConf(), which returns 0 in some whitelist setups.
    double confSum = 0.0;
    int confCount = 0;
    if (tesseract::ResultIterator* it = impl_->api.GetIterator()) {
        do {
            const float c = it->Confidence(tesseract::RIL_SYMBOL);
            if (c >= 0.0F) {
                confSum += c;
                ++confCount;
            }
        } while (it->Next(tesseract::RIL_SYMBOL));
    }
    result.confidence =
        confCount > 0 ? static_cast<float>(confSum / confCount / 100.0) : 0.0F;

    if (static_cast<int>(result.text.size()) < cfg_.minChars) {
        return {};  // too short to be a real plate read
    }
    return result;
}

}  // namespace vision
