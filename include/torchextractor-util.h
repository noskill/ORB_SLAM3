#include "Extractor.h"


namespace ORB_SLAM3 {
    Extractor * createTorchExtractor(std::string model_path, float threshold, float scale_factor, unsigned short n_levels);
    Extractor * createTorchExtractor(Extractor * a_module, float threshold, float scale_factor, unsigned short n_levels);
}
