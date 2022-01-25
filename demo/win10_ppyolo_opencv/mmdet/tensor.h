#ifndef __TENSOR_H_
#define __TENSOR_H_

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.h"
using namespace std;


namespace mmdet {


class Tensor
{
public:
	int dims;
	int elements;
	vector<int> shape;
	cv::Mat mat;
	Tensor(cv::Mat& mat, vector<int>& shape)
	{
		this->mat = mat;
		this->shape = shape;
	}
	float at(int n, int c, int h, int w) {
		int id = this->mat.step[0] * n + this->mat.step[1] * c + this->mat.step[2] * h + this->mat.step[3] * w;
		float* p = (float*)(this->mat.data + id);
		return *p;
	}
	void set(int n, int c, int h, int w, float value) {
		int id = this->mat.step[0] * n + this->mat.step[1] * c + this->mat.step[2] * h + this->mat.step[3] * w;
		float* p = (float*)(this->mat.data + id);
		*p = value;
	}
};

} // namespace mmdet

#endif // __TENSOR_H_

