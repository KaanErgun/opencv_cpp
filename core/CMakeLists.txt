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

# vision_alpr: OCR + recognition pipeline. Kept as a SEPARATE library so the
# Tesseract dependency stays out of the 14 apps that don't need it. Only built
# when Tesseract was found (VISION_HAVE_OCR).
if(VISION_HAVE_OCR)
  add_library(vision_alpr STATIC
    src/plate_ocr.cpp
    src/alpr_pipeline.cpp)
  target_include_directories(vision_alpr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(vision_alpr
    PUBLIC vision_core
    PRIVATE PkgConfig::TESSERACT project_warnings)
endif()
