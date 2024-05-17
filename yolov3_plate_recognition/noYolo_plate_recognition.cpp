#include <alpr.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // OpenALPR'yi başlatma
    std::string country = "eu"; // Kullanmak istediğiniz ülke kodunu belirtin
    std::string configFile = "/usr/local/share/openalpr/config/openalpr.conf";
    std::string runtimeDataDir = "/usr/local/share/openalpr/runtime_data";
    alpr::Alpr openalpr(country, configFile, runtimeDataDir);
    if (!openalpr.isLoaded()) {
        std::cerr << "Error loading OpenALPR" << std::endl;
        return -1;
    }

    // Web kamerasını açma
    cv::VideoCapture cap(0); // 0, varsayılan web kamerasını kullanır
    if (!cap.isOpened()) {
        std::cerr << "Cannot open webcam." << std::endl;
        return -1;
    }

    cv::namedWindow("Webcam", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Görüntüyü geçici bir dosyaya kaydetme
        std::string tempFileName = "temp_car_image.jpg";
        cv::imwrite(tempFileName, frame);

        // Plaka tanıma
        alpr::AlprResults results = openalpr.recognize(tempFileName);

        // Sonuçları ekrana yazdırma
        for (int i = 0; i < results.plates.size(); i++) {
            alpr::AlprPlateResult plate = results.plates[i];
            std::cout << "Plate: " << plate.bestPlate.characters 
                      << " Confidence: " << plate.bestPlate.overall_confidence << std::endl;

            // Plaka kutusunu çizin
            cv::rectangle(frame, 
                          cv::Rect(plate.plate_points[0].x, plate.plate_points[0].y, 
                                   plate.plate_points[2].x - plate.plate_points[0].x, 
                                   plate.plate_points[2].y - plate.plate_points[0].y), 
                          cv::Scalar(0, 255, 0), 2);

            // Plaka metnini ekrana yazdırın
            cv::putText(frame, plate.bestPlate.characters, 
                        cv::Point(plate.plate_points[0].x, plate.plate_points[0].y - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        }

        // Görüntüyü gösterme
        cv::imshow("Webcam", frame);

        // Geçici dosyayı temizleme
        std::remove(tempFileName.c_str());

        if (cv::waitKey(30) >= 0) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
