#include "utils.h"

#if (CV_MAJOR_VERSION > 3)
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
using namespace cv;
#define CV_LOAD_IMAGE_UNCHANGED IMREAD_UNCHANGED
#endif


namespace ORB_SLAM3{

std::vector<cv::Mat> ComputePyramid(cv::Mat & image, int n_levels, const std::vector<float> & InvScaleFactor) {
	std::vector<cv::Mat> pyramid;
	pyramid.resize(n_levels);
        for (int level = 0; level < n_levels; ++level)
        {
            float scale = InvScaleFactor[level];
            Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
            Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
            Mat temp(wholeSize, image.type()), masktemp;
            pyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            if( level != 0 )
            {
                resize(pyramid[level-1], pyramid[level], sz, 0, 0, INTER_LINEAR);

                copyMakeBorder(pyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101+BORDER_ISOLATED);
            }
            else
            {
                copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101);
            }
        }
       return pyramid;
    }

}; // namespace
