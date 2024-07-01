// g++ -std=c++11 -I/usr/local/include/openalpr -o alpr_file.out alpr_file.cpp `pkg-config --cflags --libs opencv4` -L/usr/local/lib -lopenalpr -Wl,-rpath,/usr/local/lib

#include <alpr.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty()) {
        std::cerr << "Error loading image: " << imagePath << std::endl;
        return -1;
    }

    std::string country = "au"; // Australia
    std::string configFile = "/openalpr.conf";
    std::string runtimeDataDir = "/usr/local/share/openalpr/runtime_data";

    alpr::Alpr openalpr(country, configFile, runtimeDataDir);
    if (!openalpr.isLoaded()) {
        std::cerr << "Error loading OpenALPR" << std::endl;
        return -1;
    }

    std::vector<unsigned char> buffer;
    cv::imencode(".jpg", frame, buffer);
    std::vector<char> image_data(buffer.begin(), buffer.end());

    alpr::AlprResults results = openalpr.recognize(image_data);

    if (results.plates.size() > 0) {
        for (int i = 0; i < results.plates.size(); i++) {
            alpr::AlprPlateResult plate = results.plates[i];
            std::cout << "Plate: " << plate.bestPlate.characters 
                      << " Confidence: " << plate.bestPlate.overall_confidence << std::endl;
        }
    } else {
        std::cerr << "No plates detected." << std::endl;
        return -1;
    }

    return 0;
}
