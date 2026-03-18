#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    cv::VideoCapture capture;
    int cameraIndex = 0;
    bool useVideoFile = false;
    std::string videoPath;

    if (argc > 1) {
        const std::string sourceArg = argv[1];

        try {
            std::size_t parsedChars = 0;
            cameraIndex = std::stoi(sourceArg, &parsedChars);

            if (parsedChars != sourceArg.size()) {
                useVideoFile = true;
                videoPath = sourceArg;
            }
        } catch (const std::exception&) {
            useVideoFile = true;
            videoPath = sourceArg;
        }
    }

#ifdef __APPLE__
    const int preferredBackend = cv::CAP_AVFOUNDATION;
#else
    const int preferredBackend = cv::CAP_ANY;
#endif

    if (useVideoFile) {
        capture.open(videoPath);
        if (!capture.isOpened()) {
            std::cerr << "Impossible d'ouvrir la video : " << videoPath << std::endl;
            return 1;
        }
    } else {
        capture.open(cameraIndex, preferredBackend);
        if (!capture.isOpened() && preferredBackend != cv::CAP_ANY) {
            capture.open(cameraIndex, cv::CAP_ANY);
        }

        if (!capture.isOpened()) {
            std::cerr << "Impossible d'ouvrir la camera " << cameraIndex << std::endl;
            return 1;
        }

        capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        capture.set(cv::CAP_PROP_FPS, 30);
    }

    int waitTimeMs = 1;
    if (useVideoFile) {
        const double fps = capture.get(cv::CAP_PROP_FPS);
        if (fps > 0.0) {
            waitTimeMs = static_cast<int>(1000.0 / fps);
            if (waitTimeMs < 1) {
                waitTimeMs = 1;
            }
        }
    }

    std::cout
        << (useVideoFile ? "Video ouverte : " + videoPath : "Camera ouverte.")
        << " Appuie sur q ou Echap pour quitter."
        << std::endl;

    cv::namedWindow(useVideoFile ? "Flux video" : "Flux camera", cv::WINDOW_AUTOSIZE);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(3.5);
    clahe->setTilesGridSize(cv::Size(6, 6));
    const float minResponseRatio = 0.0f;

    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        1500,                    // nfeatures
        1.2f,                    // scaleFactor
        8,                       // nlevels
        10,                      // edgeThreshold
        0,                       // firstLevel
        2,                       // WTA_K
        cv::ORB::HARRIS_SCORE,   // scoreType
        15,                      // patchSize
        20                       // fastThreshold
    );

    // feature matching en utilisant FLANN


    cv::Mat frame;
    cv::Mat frameGray;
    cv::Mat frameEnhanced;
    cv::Mat frameWithFeatures;
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> filteredKeypoints;

    while (true) {
        if (!capture.read(frame) || frame.empty()) {
            if (useVideoFile) {
                std::cout << "Fin de la video." << std::endl;
            } else {
                std::cerr << "Lecture frame impossible." << std::endl;
            }
            break;
        }

        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        clahe->apply(frameGray, frameEnhanced);
        orb->detectAndCompute(frameEnhanced, cv::noArray(), keypoints, descriptors);

        const int detectedKeypointsCount = static_cast<int>(keypoints.size());
        if (!keypoints.empty()) {
            float maxResponse = keypoints.front().response;
            for (const cv::KeyPoint& keypoint : keypoints) {
                maxResponse = std::max(maxResponse, keypoint.response);
            }

            const float minAcceptedResponse = maxResponse * minResponseRatio;
            filteredKeypoints.clear();
            filteredKeypoints.reserve(keypoints.size());

            cv::Mat filteredDescriptors;
            for (int i = 0; i < static_cast<int>(keypoints.size()); ++i) {
                if (keypoints[i].response >= minAcceptedResponse) {
                    filteredKeypoints.push_back(keypoints[i]);
                    filteredDescriptors.push_back(descriptors.row(i));
                }
            }

            keypoints = filteredKeypoints;
            descriptors = filteredDescriptors;
        }

        cv::drawKeypoints(
            frameEnhanced,
            keypoints,
            frameWithFeatures,
            cv::Scalar(0, 255, 0),
            cv::DrawMatchesFlags::DEFAULT
        );

        cv::putText(
            frameWithFeatures,
            "ORB gardes: " + std::to_string(keypoints.size()) + " / " + std::to_string(detectedKeypointsCount),
            cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            0.9,
            cv::Scalar(0, 255, 0),
            2
        );

        cv::imshow(useVideoFile ? "Flux video" : "Flux camera", frameWithFeatures);

        const int key = cv::waitKey(waitTimeMs);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();
    return 0;
}
