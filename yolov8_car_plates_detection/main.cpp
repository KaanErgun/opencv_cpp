// g++ -std=c++11 -o main.out main.cpp `pkg-config --cflags --libs opencv4`

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Load the ONNX model
    dnn::Net net = dnn::readNetFromONNX("best.onnx");

    // Open a connection to the webcam (0 is usually the default camera)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open the webcam." << endl;
        return -1;
    }

    // Object names
    vector<string> class_names = {"Unknown", "Plaka"};

    // Main loop to process each frame from the webcam
    while (true) {
        Mat frame;
        // Capture a frame from the webcam
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Could not capture a frame." << endl;
            break;
        }

        // Create a blob from the image
        Mat blob = dnn::blobFromImage(frame, 1/255.0, Size(640, 640), Scalar(0, 0, 0), true, false);

        // Set the input to the network
        net.setInput(blob);

        // Run the forward pass
        Mat detections = net.forward();

        // Process the detections (example for YOLOv5, might need to adjust based on your model)
        for (int i = 0; i < detections.rows; ++i) {
            float confidence = detections.at<float>(i, 4);
            if (confidence > 0.5) {  // Adjust the confidence threshold as needed
                int class_id = static_cast<int>(detections.at<float>(i, 5));
                float x = detections.at<float>(i, 0) * frame.cols;
                float y = detections.at<float>(i, 1) * frame.rows;
                float width = detections.at<float>(i, 2) * frame.cols;
                float height = detections.at<float>(i, 3) * frame.rows;
                
                Rect box((int)(x - width / 2), (int)(y - height / 2), (int)width, (int)height);
                rectangle(frame, box, Scalar(0, 255, 0), 2);
                putText(frame, class_names[class_id], Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

                // Print the detected object name to the terminal
                cout << "Detected: " << class_names[class_id] << endl;
            }
        }

        // Display the frame with detections
        imshow("Detections", frame);

        // Exit if the user presses the 'q' key
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Release the webcam
    cap.release();
    destroyAllWindows();

    return 0;
}
