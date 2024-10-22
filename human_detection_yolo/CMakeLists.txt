cmake_minimum_required(VERSION 3.10)
project(human_detection_yolo)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

add_executable(human_detection_yolo main.cpp)
target_link_libraries(human_detection_yolo ${OpenCV_LIBS})
