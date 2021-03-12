#ifndef TORCHEXTRACTOR_H
#define TORCHEXTRACTOR_H

#include "Extractor.h"

#include <torch/torch.h>
#include <torch/script.h>



namespace ORB_SLAM3 {

class TorchExtractor: public Extractor{
public:
    TorchExtractor(std::string model_path, float threshold, float scale_factor, unsigned short n_levels);
    TorchExtractor(torch::jit::script::Module & a_module, float threshold, float scale_factor, unsigned short n_levels);
    virtual int operator()(cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    virtual std::vector<cv::Mat> & getImagePyramid();
    torch::jit::script::Module & getModule();

private:
    torch::jit::script::Module module;
    float threshold;
};

} // namespace

#endif
