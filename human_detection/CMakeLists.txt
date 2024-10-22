cmake_minimum_required(VERSION 3.10)

# Project name
project(HumanDetection)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add the executable (Replace "human_detection.cpp" with your source file name)
add_executable(HumanDetection human_detection.cpp) # Kaynak dosyanın ismi

# Link OpenCV libraries
target_link_libraries(HumanDetection ${OpenCV_LIBS})

# Include directories for OpenCV
target_include_directories(HumanDetection PRIVATE ${OpenCV_INCLUDE_DIRS})

