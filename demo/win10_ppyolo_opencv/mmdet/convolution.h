#ifndef __CONVOLUTION_H_
#define __CONVOLUTION_H_

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "tensor.h"
using namespace std;


namespace mmdet {

class Convolution
{
public:
	int stride;
	int padding;
	mmdet::Tensor* kernel;
	Convolution(mmdet::Tensor* kernel, int stride, int padding)
	{
		this->kernel = kernel;
		this->stride = stride;
		this->padding = padding;
	}


	mmdet::Tensor* forward(mmdet::Tensor* x) {
		int N = x->shape.at(0);
		int C = x->shape.at(1);
		int H = x->shape.at(2);
		int W = x->shape.at(3);

		int kH = kernel->shape.at(2);
		int kW = kernel->shape.at(3);
		int out_C = kernel->shape.at(0);

		int out_W = (W + 2 * padding - (kW - 1)) / stride;
		int out_H = (H + 2 * padding - (kH - 1)) / stride;
		printf("N: %d\t", N);
		printf("C: %d\t", C);
		printf("H: %d\t", H);
		printf("W: %d\n", W);
		printf("N: %d\t", N);
		printf("out_C: %d\t", out_C);
		printf("out_H: %d\t", out_H);
		printf("out_W: %d\n", out_W);

		int dstImage_shape_[4] = { N, out_C, out_H, out_W };
		cv::Mat out_(4, dstImage_shape_, CV_32FC1, cv::Scalar(0.0));         // 卷积后的图片
		vector<int> dstImage_shape;
		for (int j = 0; j < 4; j++) {
			dstImage_shape.push_back(dstImage_shape_[j]);
		}
		mmdet::Tensor* out = new mmdet::Tensor(out_, dstImage_shape);



		// 1.先对图片x填充得到填充后的图片pad_x

		// 2.卷积核滑动，只会在H和W两个方向上滑动
		int i, j, r, s, n, in_c, out_c, id;
		for (i = 0; i < out_H; i++) {  // i是纵坐标
			for (j = 0; j < out_W; j++) {  // j是横坐标
				int ori_x = j * stride;   // 卷积核在pad_x中的横坐标，等差数列，公差是stride
				int ori_y = i * stride;   // 卷积核在pad_x中的纵坐标，等差数列，公差是stride

				// aaa
				for (n = 0; n < N; n++) {
					for (out_c = 0; out_c < out_C; out_c++) {
						float sum = 0.0;
						for (r = 0; r < kH; r++) {  // r是纵坐标
							for (s = 0; s < kW; s++) {  // s是横坐标 
								for (in_c = 0; in_c < C; in_c++) {
									float input_value = x->at(n, in_c, ori_y + r, ori_x + s);
									float kernel_value = kernel->at(out_c, in_c, r, s);
									float out_value = input_value * kernel_value;
									sum += out_value;

									printf("%f\n", input_value);
									printf("%f\n", kernel_value);
									printf("%f\n", out_value);
									printf("\n");
								}
							}
						}
						out->set(n, out_c, i, j, sum);
						printf("%f\n", sum);
						printf("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n");
					}
				}
			}
		}
		return out;
	}
	void reshape() {
		;
	}
	int getDims() {
		return this->stride;
	}
};

} // namespace mmdet

#endif // __CONVOLUTION_H_

