#ifndef EXTRACTOR_H
#define EXTRACTOR_H

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

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    virtual int operator()(cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea)=0;

    virtual int inline GetLevels()=0;
    virtual float inline GetScaleFactor()=0;

    virtual std::vector<float> inline GetScaleFactors()=0;

    virtual std::vector<float> inline GetInverseScaleFactors()=0;

    virtual std::vector<float> inline GetScaleSigmaSquares()=0;

    virtual std::vector<float> inline GetInverseScaleSigmaSquares()=0;

    virtual std::vector<cv::Mat> & getImagePyramid()=0;

    virtual ~Extractor()=default;
};

} // namespace
#endif // EXTRACTOR_H
