#pragma once

#include <string>

namespace vision::cli {

// Returns the value following `key` (e.g. argValue(argc, argv, "--source")),
// or an empty string if the flag is absent.
inline std::string argValue(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc - 1; ++i) {
        if (key == argv[i]) {
            return argv[i + 1];
        }
    }
    return {};
}

// True if the bare flag `key` (e.g. "--headless") is present.
inline bool hasFlag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (key == argv[i]) {
            return true;
        }
    }
    return false;
}

// argValue with a default when the flag is absent.
inline std::string argOr(int argc, char** argv, const std::string& key,
                         const std::string& fallback) {
    const std::string v = argValue(argc, argv, key);
    return v.empty() ? fallback : v;
}

// Integer argValue with a default; returns fallback on missing or bad input.
inline int argInt(int argc, char** argv, const std::string& key, int fallback) {
    const std::string v = argValue(argc, argv, key);
    if (v.empty()) {
        return fallback;
    }
    try {
        return std::stoi(v);
    } catch (...) {
        return fallback;
    }
}

}  // namespace vision::cli
