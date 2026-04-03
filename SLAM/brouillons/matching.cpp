#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
 
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
 
const char* keys =
    "{ help h |                  | Print help message. }"
    "{ input1 | box.png          | Path to input image 1. }"
    "{ input2 | box_in_scene.png | Path to input image 2. }";
 
int main( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys );
    const String input1Path = parser.get<String>("input1");
    const String input2Path = parser.get<String>("input2");

    Mat img1 = imread( input1Path, IMREAD_GRAYSCALE );
    Mat img2 = imread( input2Path, IMREAD_GRAYSCALE );
    Mat img1e;
    Mat img2e;
    Mat img2Gray;
    Mat img1Gray;
    if ( img1.empty() || img2.empty() )
    {
        cout << "Could not open input images:" << endl;
        cout << "  input1 = " << input1Path << endl;
        cout << "  input2 = " << input2Path << "\n" << endl;
        parser.printMessage();
        return -1;
    }
    
    //-- Step 1: Detect the keypoints
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(5);
    clahe->setTilesGridSize(Size(8,8));
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

    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    clahe->apply(img1,img1e);
    clahe->apply(img2,img2e);
    
    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);


    BFMatcher matcher(NORM_HAMMING);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher.knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    


    //-- Draw matches
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
 
    //-- Show detected matches
    imshow("Good Matches", img_matches );
 
    waitKey();
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif
