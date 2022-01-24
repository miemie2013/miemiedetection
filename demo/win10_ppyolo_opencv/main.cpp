#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "mmdet/tensor.h"

void pretty_print(cv::Mat& img, int max_h = 6, int max_w = 6) {
    double t0 = (double)cv::getTickCount();
    int H = img.rows;
    int W = img.cols;
    int C = img.channels();
    printf("H:%d\t", H);
    printf("W:%d\t", W);
    printf("C:%d\n", C);
    for (int y = 0; y < H; y++) {
        const uchar* ptr = img.ptr(y);
        for (int x = 0; x < W; x++) {
            int a = ptr[0];
            int b = ptr[1];
            int c = ptr[2];
            if ((x < max_w) && (y < max_h)) {
                printf("[%d, %d, %d] ", a, b, c);
            }
            if ((x == max_w) && (y < max_h)) {
                printf("...");
            }
            ptr += 3;
        }

        if (y < max_h) {
            printf("\n");
        }else if (y == max_h) {
            printf("...\n");
        }
    }
    printf("------------------------\n");
    double time = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    printf("Time for pretty_print: %f\n", time);
}

int main(int argc, char** argv)
{
    // 和python版opencv一样，读出来的图片是BGR格式。
	cv::Mat image = cv::imread("../../assets/dog.jpg");
	if (image.empty())
	{
		printf("could not load image…\n");
		return -1;
	}
    pretty_print(image);
	cv::namedWindow("test opencv setup");
	cv::imshow("test opencv setup", image);

    cv::Mat kern = (cv::Mat_<char>(3, 3) << 0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    cv::Mat dstImage;
    cv::filter2D(image, dstImage, image.depth(), kern);
    cv::imshow("dstImage", dstImage);

    mmdet::Tensor* tensor = new mmdet::Tensor(8, 3, 13, 13);
    printf("tensor->getDim0()\n");
    printf("%d", tensor->getDim0());


    cv::waitKey(0);
	return 0;
}
