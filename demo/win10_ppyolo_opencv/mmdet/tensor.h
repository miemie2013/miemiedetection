#ifndef __TENSOR_H_
#define __TENSOR_H_

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.h"
using namespace std;


namespace mmdet {


template <class T> class Tensor
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
	T at(int n, int c, int h, int w) {
		int id = this->mat.step[0] * n + this->mat.step[1] * c + this->mat.step[2] * h + this->mat.step[3] * w;
		T* p = (T*)(this->mat.data + id);
		return *p;
	}
	void set(int n, int c, int h, int w, T value) {
		int id = this->mat.step[0] * n + this->mat.step[1] * c + this->mat.step[2] * h + this->mat.step[3] * w;
		T* p = (T*)(this->mat.data + id);
		*p = value;
	}
};

} // namespace mmdet

#endif // __TENSOR_H_

