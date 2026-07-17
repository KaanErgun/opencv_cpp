#include <catch2/catch_test_macros.hpp>

#include "vision/video_source.hpp"

using vision::SourceSpec;

TEST_CASE("SourceSpec parses a webcam index", "[source]") {
    const auto s = SourceSpec::parse("0");
    REQUIRE(s.kind == SourceSpec::Kind::Webcam);
    REQUIRE(s.cameraIndex == 0);
    REQUIRE(s.isLive());
}

TEST_CASE("SourceSpec parses a non-zero webcam index", "[source]") {
    const auto s = SourceSpec::parse("2");
    REQUIRE(s.kind == SourceSpec::Kind::Webcam);
    REQUIRE(s.cameraIndex == 2);
}

TEST_CASE("SourceSpec parses an rtsp url", "[source]") {
    const auto s = SourceSpec::parse("rtsp://user:pass@host:554/stream");
    REQUIRE(s.kind == SourceSpec::Kind::Rtsp);
    REQUIRE(s.path == "rtsp://user:pass@host:554/stream");
    REQUIRE(s.isLive());
}

TEST_CASE("SourceSpec parses a file path", "[source]") {
    const auto s = SourceSpec::parse("clip.mp4");
    REQUIRE(s.kind == SourceSpec::Kind::File);
    REQUIRE(s.path == "clip.mp4");
    REQUIRE_FALSE(s.isLive());
}

TEST_CASE("SourceSpec treats a path with digits and slash as a file", "[source]") {
    const auto s = SourceSpec::parse("videos/01.mp4");
    REQUIRE(s.kind == SourceSpec::Kind::File);
}
