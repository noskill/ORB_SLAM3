/**
 * File: FGOOD.cpp
 * Author: Anatoly Belikov
 * Description: functions for GOODPoints descriptors
 * License: BSD
 *
 *
 */


#include <vector>
#include <cassert>
#include <string>
#include <sstream>
#include <stdint-gcc.h>
#include <climits>
#include "FGOOD.h"

using namespace std;

// check for 32bit float
char static_assert_float32[1 - (2 * ((sizeof(float) * CHAR_BIT) != 32))];


namespace DBoW2 {

// --------------------------------------------------------------------------

const int FGOOD::L=256;

void FGOOD::meanValue(const std::vector<FGOOD::pDescriptor> &descriptors,
  FGOOD::TDescriptor &mean)
{
	FGOOD::TDescriptor result = cv::Mat::zeros(1, FGOOD::L, CV_32F);
	// iterative mean
	// http://www.heikohoffmann.de/htmlthesis/node134.html
	for(std::size_t i=0; i < descriptors.size(); i++){
		auto x = descriptors[i];
		result = result + (*x - result) * 1.0 / (float(i) + 1.0);
	}
	mean = result;
}

// --------------------------------------------------------------------------

double FGOOD::distance(const FGOOD::TDescriptor &a,
  const FGOOD::TDescriptor &b)
{
	double dist = cv::norm(a, b, cv::NORM_L2);
	return dist;
}

// --------------------------------------------------------------------------

std::string FGOOD::toString(const FGOOD::TDescriptor &a)
{
  stringstream ss;
  const float *p = a.ptr<float>();

  for(int i = 0; i < a.cols; ++i, ++p)
  {
    ss << (float)*p << " ";
  }

  return ss.str();
}

// --------------------------------------------------------------------------

void FGOOD::fromString(FGOOD::TDescriptor &a, const std::string &s)
{
  a.create(1, FGOOD::L, CV_32F);
  float *p = a.ptr<float>();

  stringstream ss(s);
  for(int i = 0; i < FGOOD::L; ++i, ++p)
  {
    float n;
    ss >> n;

    if(!ss.fail())
      *p = n;
  }

}

// --------------------------------------------------------------------------

void FGOOD::toMat32F(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }

  const size_t N = descriptors.size();

  mat.create(N, FGOOD::L, CV_32F);
  float *p;

  for(size_t i = 0; i < N; ++i)
  {
    assert(descriptors[i].type() == mat.type());
    p = mat.ptr<float>(i);
    (*p) = *(descriptors[i].data);
  }
}

// --------------------------------------------------------------------------

} // namespace DBoW2


