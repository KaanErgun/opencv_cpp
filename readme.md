# YOLOv3 Image Processing Samples

This repository showcases a variety of image processing applications developed using the YOLOv3 (You Only Look Once) object detection algorithm, integrated with C++ and OpenCV. Each project is dedicated to different scenarios including object recognition, tracking, and more.

## Projects

- **human_detection**: Recognizes and tracks human figures in video feeds.
- **simple_rtsp**: Handles RTSP video streaming.
- **yolov3**: Core folder for shared YOLOv3 configurations and weights.
- **yolov3_car_detection**: Detects cars in video feeds using YOLOv3.
- **yolov3_cow_detection**: Custom application for detecting cows in agricultural environments.
- **yolov3_human_detection**: Focused on detecting humans, ideal for security applications.

## Getting Started

Each project folder includes source files and a `README` detailing compilation and execution instructions.

### Prerequisites

- CMake 3.0 or newer
- OpenCV 4.0 or newer
- A compatible C++ compiler

### Sample Compile Command

For compiling the cow detection example:
g++ -std=c++17 cow_detection_file.cpp -o cow_detection_file.out -I/usr/include/opencv4 -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_dnn


## Downloading YOLOv3 Weights

Before running the projects, download the necessary YOLOv3 weights:

[Download YOLOv3 weights](https://drive.google.com/drive/folders/1HVAhRrmSAIatjzEXmyS3QTHweT-l2X-s?usp=sharing)

## Contribution

Contributions are welcome! Feel free to fork this repository and submit pull requests. For bug reports, suggestions, or enhancements, please open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
