#include <catch2/catch_test_macros.hpp>

#include "vision/detection.hpp"
#include "vision/tracker.hpp"

using vision::Detection;
using vision::IouTracker;

namespace {
Detection makeDet(int x, int y, int w, int h, int classId = 0) {
    Detection d;
    d.box = cv::Rect(x, y, w, h);
    d.classId = classId;
    d.confidence = 0.9F;
    return d;
}
}  // namespace

TEST_CASE("A new detection spawns exactly one track", "[tracker]") {
    IouTracker tracker;
    const auto& tracks = tracker.update({makeDet(0, 0, 50, 50)});
    REQUIRE(tracks.size() == 1);
    REQUIRE(tracks[0].id == 0);
}

TEST_CASE("An overlapping detection keeps the same track id", "[tracker]") {
    IouTracker tracker;
    tracker.update({makeDet(0, 0, 50, 50)});
    const auto& tracks = tracker.update({makeDet(5, 5, 50, 50)});  // high IoU
    REQUIRE(tracks.size() == 1);
    REQUIRE(tracks[0].id == 0);
    REQUIRE(tracks[0].age == 1);
}

TEST_CASE("A disjoint detection creates a second track", "[tracker]") {
    IouTracker tracker;
    tracker.update({makeDet(0, 0, 50, 50)});
    const auto& tracks = tracker.update({makeDet(500, 500, 50, 50)});
    REQUIRE(tracks.size() == 2);
}

TEST_CASE("A track ages out after maxMissed misses", "[tracker]") {
    IouTracker tracker(0.3F, 2);
    tracker.update({makeDet(0, 0, 50, 50)});
    tracker.update({});  // miss 1
    REQUIRE(tracker.tracks().size() == 1);
    tracker.update({});  // miss 2
    REQUIRE(tracker.tracks().size() == 1);
    tracker.update({});  // miss 3 -> exceeds maxMissed, dropped
    REQUIRE(tracker.tracks().empty());
}
