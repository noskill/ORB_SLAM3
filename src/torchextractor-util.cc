#include "torchextractor-util.h"
#include "torchextractor.h"

namespace ORB_SLAM3 {

Extractor * createTorchExtractor(std::string model_path, float threshold, float scale_factor, unsigned short n_levels){
	return new TorchExtractor(model_path, threshold, scale_factor, n_levels);
}

Extractor * createTorchExtractor(Extractor * a_module, float threshold, float scale_factor, unsigned short n_levels){
        TorchExtractor * t = dynamic_cast<TorchExtractor*>(a_module);
	if (not t){
		throw std::runtime_error("can't cast Extractor to TorchExtractor");
	}
	return new TorchExtractor(t->getModule(), threshold, scale_factor, n_levels);
}

}
