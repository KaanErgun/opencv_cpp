#include "vision/tracker.hpp"

#include <algorithm>

namespace vision {
namespace {

float iou(const cv::Rect& a, const cv::Rect& b) {
    const int interArea = (a & b).area();
    const int unionArea = a.area() + b.area() - interArea;
    return unionArea > 0 ? static_cast<float>(interArea) / unionArea : 0.0F;
}

}  // namespace

const std::vector<Track>& IouTracker::update(const std::vector<Detection>& detections) {
    std::vector<bool> trackMatched(tracks_.size(), false);
    std::vector<bool> detMatched(detections.size(), false);

    // Greedy association: repeatedly take the highest-IoU (track, det) pair.
    while (true) {
        float bestIou = iouThreshold_;
        int bestTrack = -1;
        int bestDet = -1;
        for (size_t t = 0; t < tracks_.size(); ++t) {
            if (trackMatched[t]) {
                continue;
            }
            for (size_t d = 0; d < detections.size(); ++d) {
                if (detMatched[d]) {
                    continue;
                }
                const float score = iou(tracks_[t].box, detections[d].box);
                if (score > bestIou) {
                    bestIou = score;
                    bestTrack = static_cast<int>(t);
                    bestDet = static_cast<int>(d);
                }
            }
        }
        if (bestTrack < 0) {
            break;
        }
        trackMatched[bestTrack] = true;
        detMatched[bestDet] = true;
        tracks_[bestTrack].box = detections[bestDet].box;
        tracks_[bestTrack].classId = detections[bestDet].classId;
        tracks_[bestTrack].missedFrames = 0;
        tracks_[bestTrack].age++;
    }

    // Unmatched tracks age out; drop the stale ones.
    for (size_t t = 0; t < tracks_.size(); ++t) {
        if (!trackMatched[t]) {
            tracks_[t].missedFrames++;
        }
    }
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [this](const Track& tr) { return tr.missedFrames > maxMissed_; }),
        tracks_.end());

    // Unmatched detections spawn new tracks.
    for (size_t d = 0; d < detections.size(); ++d) {
        if (!detMatched[d]) {
            Track tr;
            tr.id = nextId_++;
            tr.box = detections[d].box;
            tr.classId = detections[d].classId;
            tracks_.push_back(tr);
        }
    }

    return tracks_;
}

}  // namespace vision
