#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>

namespace vision {

struct OcrResult {
    std::string text;         // recognised characters (uppercased, whitelisted)
    float confidence = 0.0F;  // mean per-character confidence, 0..1
};

struct OcrConfig {
    // Directory holding eng.traineddata. Empty => auto-detect (a documented
    // search list, then TESSDATA_PREFIX, then Tesseract's compiled default).
    std::string tessdataPath;
    std::string language = "eng";
    std::string whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    int psm = 7;              // page segmentation mode: 7 = single text line
    int upscaleHeight = 140;  // upscale the crop to this height before OCR
    int minChars = 4;         // reject reads shorter than this
};

// Tesseract-backed OCR for license-plate crops. Preprocesses each crop
// (grayscale, upscale, denoise, Otsu binarisation with polarity correction,
// white border) before recognition. The Tesseract dependency is hidden behind
// a pImpl so consumers don't need its headers. Not thread-safe: guard a shared
// instance, or give each worker its own (see AlprPipeline).
class PlateOcr {
   public:
    explicit PlateOcr(const OcrConfig& cfg);
    ~PlateOcr();
    PlateOcr(PlateOcr&&) noexcept;
    PlateOcr& operator=(PlateOcr&&) noexcept;
    PlateOcr(const PlateOcr&) = delete;
    PlateOcr& operator=(const PlateOcr&) = delete;

    // Reads characters from a plate crop (BGR or single-channel). Returns an
    // empty text on failure or when the result is below minChars.
    OcrResult read(const cv::Mat& plateCrop) const;

    // Exposes the preprocessed (binarised) image for debugging/visualisation.
    cv::Mat preprocess(const cv::Mat& plateCrop) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    OcrConfig cfg_;
};

}  // namespace vision
