#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int findNextImageIndex(const fs::path& outputDir) {
    int maxIndex = 0;

    if (!fs::exists(outputDir)) {
        return 1;
    }

    for (const fs::directory_entry& entry : fs::directory_iterator(outputDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const fs::path filePath = entry.path();
        if (filePath.extension() != ".jpg") {
            continue;
        }

        const std::string stem = filePath.stem().string();
        const std::string prefix = "image";
        if (stem.rfind(prefix, 0) != 0 || stem.size() <= prefix.size()) {
            continue;
        }

        try {
            const int currentIndex = std::stoi(stem.substr(prefix.size()));
            if (currentIndex > maxIndex) {
                maxIndex = currentIndex;
            }
        } catch (const std::exception&) {
            continue;
        }
    }

    return maxIndex + 1;
}

int main(int argc, char** argv) {
    const std::string videoPath = (argc > 1) ? argv[1] : "SLAM/video.mkv";
    const fs::path outputDir = (argc > 2) ? fs::path(argv[2]) : fs::path("SLAM/captures");

    cv::VideoCapture video(videoPath);
    if (!video.isOpened()) {
        std::cerr << "Impossible d'ouvrir la video : " << videoPath << std::endl;
        return 1;
    }

    std::error_code createDirError;
    fs::create_directories(outputDir, createDirError);
    if (createDirError) {
        std::cerr << "Impossible de creer le dossier de sortie : " << outputDir << std::endl;
        return 1;
    }

    int nextImageIndex = findNextImageIndex(outputDir);

    int waitTimeMs = 30;
    const double fps = video.get(cv::CAP_PROP_FPS);
    if (fps > 0.0) {
        waitTimeMs = static_cast<int>(1000.0 / fps);
        if (waitTimeMs < 1) {
            waitTimeMs = 1;
        }
    }

    std::cout << "Video : " << videoPath << std::endl;
    std::cout << "Dossier de sortie : " << outputDir << std::endl;
    std::cout << "Appuie sur s pour sauver une image, q ou Echap pour quitter." << std::endl;

    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    while (true) {
        if (!video.read(frame) || frame.empty()) {
            std::cout << "Fin de la video." << std::endl;
            break;
        }

        cv::Mat displayFrame = frame.clone();
        cv::putText(
            displayFrame,
            "s: save  q: quit  next: image" + std::to_string(nextImageIndex) + ".jpg",
            cv::Point(20, 35),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 0),
            2
        );

        cv::imshow("Video", displayFrame);

        const int key = cv::waitKey(waitTimeMs);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }

        if (key == 's' || key == 'S') {
            const fs::path outputPath = outputDir / ("image" + std::to_string(nextImageIndex) + ".jpg");
            if (cv::imwrite(outputPath.string(), frame)) {
                std::cout << "Image enregistree : " << outputPath << std::endl;
                ++nextImageIndex;
            } else {
                std::cerr << "Echec de l'enregistrement : " << outputPath << std::endl;
            }
        }
    }

    video.release();
    cv::destroyAllWindows();
    return 0;
}
