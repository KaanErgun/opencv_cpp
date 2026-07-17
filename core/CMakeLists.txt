add_library(vision_core STATIC
  src/yolo_detector.cpp
  src/hog_detector.cpp
  src/video_source.cpp
  src/annotator.cpp
  src/tracker.cpp
  src/config.cpp)

target_include_directories(vision_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(vision_core
  PUBLIC
    ${OpenCV_LIBS}
    Threads::Threads
  PRIVATE
    nlohmann_json::nlohmann_json
    project_warnings)

if(VISION_HAVE_HOG AND DEFINED VISION_HOG_LIBS)
  target_link_libraries(vision_core PUBLIC ${VISION_HOG_LIBS})
endif()

target_include_directories(vision_core PUBLIC ${OpenCV_INCLUDE_DIRS})
