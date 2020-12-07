#include <iostream>
#include "torchextractor.h"
#include <torch/script.h>

using namespace ORB_SLAM3;

TorchExtractor::TorchExtractor(std::string model_path){
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      this->module = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      throw e;
    }
}
