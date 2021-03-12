#ifndef UTILS_H
#define UTILS_H


#include <opencv2/opencv.hpp>

namespace ORB_SLAM3 {
    const int EDGE_THRESHOLD = 19;
    std::vector<cv::Mat> ComputePyramid(cv::Mat & image, int n_levels, const std::vector<float> & InvScaleFactor);

}; // namespace


#endif
