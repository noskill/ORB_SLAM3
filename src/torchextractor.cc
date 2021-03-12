#include <iostream>
#include "torchextractor.h"
#include "utils.h"
#include <torch/torch.h>
#include <torch/script.h>

using namespace ORB_SLAM3;

TorchExtractor::TorchExtractor(torch::jit::script::Module & a_module,
		float threshold, float scale_factor, unsigned short n_levels):
	Extractor(scale_factor, n_levels), module(a_module), threshold(threshold) {
}

TorchExtractor::TorchExtractor(std::string model_path, float threshold, float scale_factor, unsigned short n_levels):
       	Extractor(scale_factor, n_levels), threshold(threshold) {
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      this->module = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      throw e;
    }

}

std::vector < std::vector< cv::KeyPoint> > ComputeKeypoints(torch::jit::script::Module & module,
		const std::vector<cv::Mat> & pyramid, float threshold);

int TorchExtractor::operator()(cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea) {
	if(_image.empty())
            return -1;

        cv::Mat image = _image.getMat();
        assert(image.type() == CV_8UC1 );

        // Pre-compute the scale pyramid
	std::vector<cv::Mat> pyramid = ComputePyramid(image, GetLevels(), GetInverseScaleFactors());

	std::vector < std::vector< cv::KeyPoint> > allKeypoints = ComputeKeypoints(module,
			pyramid, threshold);
	return 0;
}

std::vector < std::vector< cv::KeyPoint> > ComputeKeypoints(torch::jit::script::Module & module,
		const std::vector<cv::Mat> & pyramid, float threshold) {
	std::vector < std::vector< cv::KeyPoint> > allKeypoints;
	std::size_t nlevels = pyramid.size();
	allKeypoints.resize(nlevels);
        for (std::size_t level = 0; level < nlevels; ++level) {
            cv::Mat const & img = pyramid[level];
	    assert(img.type() == cv::CV_8U);
	    auto tensor_image = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
	    tensor_image = tensor_image.toType(c10::kFloat) / 255.0;
	    std::vector<c10::IValue> input({tensor_image});
	    auto thres = torch::tensor({threshold});
	    auto outputs = module.forward(input).toTuple();
	    auto points = outputs->elements()[0].toTensor();
	    auto desc = outputs->elements()[1].toTensor();
	    std::cout << points.sizes() << std::endl;
	}
	return allKeypoints;
}


std::vector<cv::Mat> & TorchExtractor::getImagePyramid(){
	throw std::runtime_error("not implemented");
}

torch::jit::script::Module & TorchExtractor::getModule(){
	return module;
}
