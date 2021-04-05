#include <iostream>
#include "torchextractor.h"
#include "utils.h"
#include <tuple>
#include <torch/torch.h>
#include <torch/script.h>

using namespace ORB_SLAM3;

cv::Mat torchTensortoCVMat(torch::Tensor& tensor);
torch::Tensor cvMattoTensor(const cv::Mat& img);

TorchExtractor::TorchExtractor(torch::jit::script::Module & a_module,
		float threshold, float scale_factor, unsigned short n_levels):
	Extractor(scale_factor, n_levels), module(a_module), threshold(threshold) {
}

TorchExtractor::TorchExtractor(std::string model_path, float threshold, float scale_factor, unsigned short n_levels):
       	Extractor(scale_factor, n_levels), threshold(threshold) {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()){
        device = torch::kCUDA;
    }
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      this->module = torch::jit::load(model_path, device);
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      throw e;
    }

}

typedef std::tuple< std::vector < std::vector< cv::KeyPoint> > ,
                    cv::Mat > point_desc_t; 

point_desc_t ComputeKeypoints(torch::jit::script::Module & module,
		const std::vector<cv::Mat> & pyramid, float threshold);



int TorchExtractor::operator()(cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea) {
	if(_image.empty())
            return -1;

        cv::Mat image = _image.getMat();
        assert(image.type() == CV_8UC1 );

        if (GetLevels() != 1) {
           throw std::runtime_error("Only one level is supported");
        }
        // Pre-compute the scale pyramid
	std::vector<cv::Mat> pyramid = ComputePyramid(image, GetLevels(), GetInverseScaleFactors());

        auto [allKeypoints, desc] = ComputeKeypoints(module,
			pyramid, threshold);

        // compute orientations
        for (int level = 0; level < nlevels; ++level)
            computeOrientation(pyramid[level], allKeypoints[level], umax);

        std::size_t nkeypoints = 0;
        for (int level = 0; level < nlevels; ++level) {
            nkeypoints += allKeypoints[level].size();
        }
        cv::Mat descriptors;
        if( nkeypoints == 0 )
            _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, desc.cols, desc.type());
            descriptors = _descriptors.getMat();
        }
        _keypoints = std::vector<cv::KeyPoint>(nkeypoints);

        int i = 0;
        int monoIndex = 0, stereoIndex = nkeypoints-1;
        for (int level = 0; level < nlevels; ++level){
	    auto & keypoints = allKeypoints[level];

            float scale = mvScaleFactor[level];
            for (auto keypoint = keypoints.begin(),
                         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){
 
                // Scale keypoint coordinates
                if (level != 0){
                    keypoint->pt *= scale;
                }
 
                if(keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1]){
                    _keypoints.at(stereoIndex) = (*keypoint);
                    desc.row(i).copyTo(descriptors.row(stereoIndex));
                    stereoIndex--;
                }
                else{
                    _keypoints.at(monoIndex) = (*keypoint);
                    desc.row(i).copyTo(descriptors.row(monoIndex));
                    monoIndex++;
                }
                i++;
            }

        }

	return monoIndex;
}

point_desc_t ComputeKeypoints(torch::jit::script::Module & module,
		const std::vector<cv::Mat> & pyramid, float threshold) {
	std::vector < std::vector< cv::KeyPoint> > allKeypoints;
	std::size_t nlevels = pyramid.size();
	allKeypoints.resize(nlevels);
        cv::Mat desc_mat;
        using namespace cv;
        for (std::size_t level = 0; level < nlevels; ++level) {
            cv::Mat const & img = pyramid[level];
            // cv::imwrite("test.png", img);
            auto tensor_image = cvMattoTensor(img);
            tensor_image = torch::unsqueeze(tensor_image, 0);
            tensor_image = torch::unsqueeze(tensor_image, 0);
	    tensor_image = tensor_image.toType(c10::kFloat) / 255.0;
            assert(tensor_image.max().item<float>() <= 1.0);
            assert(tensor_image.max().item<float>() > 0.05);
	    auto thres = torch::tensor({threshold});
	    std::vector<c10::IValue> input({tensor_image, thres});
	    auto outputs = module.forward(input).toTuple();
	    auto points = outputs->elements()[0].toTensor();
	    auto desc = outputs->elements()[1].toTensor();

	    std::cout << "points " << points.sizes() << std::endl;
            std::cout << "descriptors " << desc.sizes() << std::endl;
            auto & current_level_keypoints = allKeypoints[level];
            for(int k=0; k < points.sizes()[0]; k++){
               auto points_row = points[k];
               float row = points_row[0].item<float>();
               float col = points_row[1].item<float>();
               current_level_keypoints.push_back(cv::KeyPoint(col, row, 1));
            }
/*
            cv::Mat point_img = img.clone();
            cv::drawKeypoints(img, current_level_keypoints, point_img, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::imshow("image points", point_img);
            cv::imshow("gray", img);
            cv::waitKey(3000);
*/          
            desc_mat.push_back(torchTensortoCVMat(desc));
	}
	return std::make_tuple(allKeypoints, desc_mat);
    }


torch::Tensor cvMattoTensor(const cv::Mat& img){
    assert(img.type() == CV_8U);
    assert(img.channels() == 1);
    auto tensor_image = torch::zeros({img.rows, img.cols}, at::kByte);

    for(int r=0; r < img.rows; r++){
    	for(int c=0; c < img.cols; c++){
            tensor_image[r][c] = img.at<unsigned char>(r, c);
        }
    }
#ifndef NDEBUG
    for(int r=0; r < img.rows; r++){
    	for(int c=0; c < img.cols; c++){
            assert(tensor_image.index({r, c}).item().to<unsigned char>() == img.at<unsigned char>(r, c));
        }
    }
#endif
    return tensor_image;
}

cv::Mat torchTensortoCVMat(torch::Tensor& tensor)
{

    static_assert(sizeof(float) * CHAR_BIT == 32, "require 32 bits floats");
    tensor = tensor.squeeze().detach();
    tensor = tensor.contiguous();
    tensor = tensor.to(torch::kFloat32);
    tensor = tensor.to(torch::kCPU);
    if (tensor.sizes().size() != 2){
       std::string err_msg = "only 2D tensor can be converted, found " + std::to_string(tensor.sizes().size()) + "D!";
       throw std::runtime_error(err_msg);
    }
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);
    cv::Mat mat = cv::Mat(cv::Size(width, height), CV_32F, tensor.data_ptr<float>());
#ifndef NDEBUG
    cv::Mat col_means;
    cv::reduce(mat, col_means, 0, CV_REDUCE_AVG);
    assert(col_means.cols == width);
    assert(col_means.rows == 1);
    torch::Tensor t_means = tensor.mean(0);
    assert(abs(t_means[1].item<float>() - col_means.at<float>(0,1)) < 0.0001); 
    for (int r=0; r < height; r++){
        for (int c=0; c < width; c++){
    	    assert(abs(tensor.index({r, c}).item<float>() - mat.at<float>(r, c)) < 0.00001); 
        }
    }
#endif
    return mat.clone();
}


std::vector<cv::Mat> & TorchExtractor::getImagePyramid(){
	throw std::runtime_error("not implemented");
}

torch::jit::script::Module & TorchExtractor::getModule(){
	return module;
}
