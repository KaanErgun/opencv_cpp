#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "vision/detection.hpp"

namespace vision {

struct Track {
    int id = -1;
    cv::Rect box;
    int classId = -1;
    int age = 0;           // frames since created
    int missedFrames = 0;  // consecutive frames without a match
};

// Minimal SORT-style multi-object tracker: greedy IoU association (highest-IoU
// pairs first), stable integer ids, and age/miss lifecycle. No Kalman filter,
// no external deps — enough for line-crossing / ROI counting at demo quality.
// Replaces the desynced carStatus vector + KCF plumbing that caused the
// out-of-bounds UB in the old dual-camera module.
class IouTracker {
   public:
    explicit IouTracker(float iouThreshold = 0.3F, int maxMissed = 15)
        : iouThreshold_(iouThreshold), maxMissed_(maxMissed) {}

    // Advances tracks by one frame using the current detections; returns the
    // live tracks after association.
    const std::vector<Track>& update(const std::vector<Detection>& detections);

    const std::vector<Track>& tracks() const { return tracks_; }

   private:
    float iouThreshold_;
    int maxMissed_;
    int nextId_ = 0;
    std::vector<Track> tracks_;
};

}  // namespace vision
