#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include <opencv2/core/version.hpp>
#if (CV_MAJOR_VERSION > 3)
#include <opencv2/opencv.hpp>
#else
#include <opencv/cv.h>
#endif

namespace ORB_SLAM3
{

class Extractor
{
public:
    Extractor(float scale_factor, int n_levels);
    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    virtual int operator()(cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea)=0;

    virtual int inline GetLevels() { return nlevels;};
    virtual float inline GetScaleFactor() { return scaleFactor; };

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    virtual std::vector<cv::Mat> & getImagePyramid()=0;

    virtual ~Extractor()=default;

protected:
    int nlevels;
    float scaleFactor;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
    std::vector<int> umax;
};

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;

void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& umax);
float IC_Angle(const cv::Mat& image, cv::Point2f pt,  const std::vector<int> & u_max);

} // namespace
#endif // EXTRACTOR_H
