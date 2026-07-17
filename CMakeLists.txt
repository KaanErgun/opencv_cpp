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
  COMPONENTS core imgproc imgcodecs highgui videoio dnn objdetect)
if(OpenCV_VERSION VERSION_LESS 4.8)
  message(FATAL_ERROR "OpenCV >= 4.8 required, found ${OpenCV_VERSION}")
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
