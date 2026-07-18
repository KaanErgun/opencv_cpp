cmake_minimum_required(VERSION 3.24)
project(opencv_cpp_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- Dependencies (resolved once, at the top level) -------------------------
# Needs OpenCV >= 4.8 for NMSBoxesBatched. We do NOT pin a version in
# find_package because OpenCV 5.x reports as "5.0" and a `4.8` request is
# rejected as incompatible even though the DNN APIs we use are source-compatible.
# Enforce the floor manually instead.
find_package(OpenCV REQUIRED
  COMPONENTS core imgproc imgcodecs highgui videoio dnn objdetect video)
if(OpenCV_VERSION VERSION_LESS 4.8)
  message(FATAL_ERROR "OpenCV >= 4.8 required, found ${OpenCV_VERSION}")
endif()

# The single-object trackers (CSRT/KCF) live in the tracking contrib module,
# which plain Linux libopencv-dev may lack. app_object_track is skipped then.
find_package(OpenCV QUIET COMPONENTS tracking)
if(TARGET opencv_tracking)
  set(VISION_HAVE_TRACKING TRUE)
else()
  set(VISION_HAVE_TRACKING FALSE)
  message(STATUS "opencv_tracking not found — app_object_track will be skipped")
endif()

# HOGDescriptor lives in the xobjdetect (contrib) module on OpenCV 5.x. Detect
# it so the HOG backend can be gated; it's optional and off if unavailable.
set(VISION_HAVE_HOG FALSE)
if(OpenCV_VERSION VERSION_GREATER_EQUAL 5.0)
  find_package(OpenCV QUIET COMPONENTS xobjdetect)
  if(TARGET opencv_xobjdetect)
    set(VISION_HAVE_HOG TRUE)
    set(VISION_HOG_LIBS opencv_xobjdetect)
  endif()
else()
  set(VISION_HAVE_HOG TRUE)  # objdetect (already required) carries HOG on 4.x
endif()
find_package(Threads REQUIRED)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3)
FetchContent_MakeAvailable(nlohmann_json)

# cpp-httplib: header-only HTTP server + client, used by the ALPR server/client.
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.3)
FetchContent_MakeAvailable(httplib)

# Tesseract OCR (via pkg-config) powers the plate-recognition pipeline. It is a
# system dependency; when absent the ALPR server/client and the OCR library are
# simply not built and the rest of the repo is unaffected.
option(VISION_WITH_OCR "Build the Tesseract-backed ALPR server/client" ON)
set(VISION_HAVE_OCR FALSE)
if(VISION_WITH_OCR)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(TESSERACT IMPORTED_TARGET tesseract lept)
    if(TESSERACT_FOUND)
      set(VISION_HAVE_OCR TRUE)
    else()
      message(STATUS "tesseract/lept not found via pkg-config — ALPR OCR "
                     "targets skipped (install with: brew install tesseract)")
    endif()
  endif()
endif()

# --- Shared warning flags ----------------------------------------------------
include(cmake/CompilerWarnings.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(core)
add_subdirectory(apps)

option(BUILD_TESTS "Build the unit tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
