#ifndef TORCHEXTRACTOR_H
#define TORCHEXTRACTOR_H

#include "Extractor.h"
#include <torch/torch.h>

namespace ORB_SLAM3 {
class TorchExtractor: public Extractor{

    TorchExtractor(std::string model_path);
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

private:
    torch::jit::script::Module module;

};

} // namespace
#endif TORCHEXTRACTOR
