#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace cv;
using namespace std;

// Normalize confidence score to 0-1 range
float normalizeConfidence(float score) {
    // Apply sigmoid function to map any real number to 0-1 range
    return 1.0f / (1.0f + exp(-score));
}

int main() {
    // Load the ONNX model
    dnn::Net net = dnn::readNetFromONNX("best.onnx");
    if (net.empty()) {
        cerr << "Error: Could not load the ONNX model." << endl;
        return -1;
    }

    // Open a connection to the webcam (0 is usually the default camera)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open the webcam." << endl;
        return -1;
    }

    // Object names
    vector<string> class_names = {"Araba", "Plaka"};

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
        Mat blob = dnn::blobFromImage(frame, 1/255.0, Size(640, 640), Scalar(), true, false);

        // Set the input to the network
        net.setInput(blob);

        // Run the forward pass
        Mat output = net.forward();

        // Debug output
        cout << "Output shape: " << output.size << endl;

        // Process the detections
        vector<Rect> boxes;
        vector<float> confidences;
        vector<int> class_ids;

        int rows = output.size[2];
        int dimensions = output.size[1];

        output = output.reshape(1, rows);

        for (int i = 0; i < rows; ++i) {
            float* row = output.ptr<float>(i);
            
            float confidence = normalizeConfidence(row[4]);
            if (confidence > 0.25) {  // Pre-filtering threshold
                float* classes_scores = row + 5;
                Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
                Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                
                max_class_score = normalizeConfidence(max_class_score);
                
                if (max_class_score > 0.25) {  // Class score threshold
                    confidences.push_back(confidence * max_class_score);  // Combined score
                    class_ids.push_back(class_id.x);
                    
                    float x = row[0];
                    float y = row[1];
                    float w = row[2];
                    float h = row[3];
                    
                    int left = int((x - 0.5 * w) * frame.cols);
                    int top = int((y - 0.5 * h) * frame.rows);
                    int width = int(w * frame.cols);
                    int height = int(h * frame.rows);
                    
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        // Perform Non-Maximum Suppression
        vector<int> indices;
        dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        // Draw detections
        for (int idx : indices) {
            Rect box = boxes[idx];
            int class_id = class_ids[idx];
            float conf = confidences[idx];
            
            rectangle(frame, box, Scalar(0, 255, 0), 2);
            stringstream ss;
            ss << class_names[class_id] << ": " << fixed << setprecision(2) << conf * 100 << "%";
            string label = ss.str();
            putText(frame, label, Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

            // Print the detected object name to the terminal
            cout << "Detected: " << label << endl;
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