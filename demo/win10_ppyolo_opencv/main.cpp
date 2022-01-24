#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;

#include <vector>

void pretty_print(cv::Mat& img, int max_h = 6, int max_w = 6) {
    double t0 = (double)cv::getTickCount();
    int height = img.rows;
    int width = img.cols;
    printf("height:%d\t", height);
    printf("width:%d\n", width);
    for (int row = 0; row < height; row++) {
        const uchar* ptr = img.ptr(row);
        for (int col = 0; col < width; col++) {
            int a = ptr[0];
            int b = ptr[1];
            int c = ptr[2];
            ptr += 3;
        }
    }
    printf("------------------------\n");
    double time = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    printf("Time for pretty_print: %f\n", time);
}

int main(int argc, char** argv)
{
	Mat src = imread("D://GitHub/miemiedetection/assets/dog.jpg");
	if (src.empty())
	{
		printf("could not load image¡­\n");
		return -1;
	}
    pretty_print(src);
	namedWindow("test opencv setup");
	imshow("test opencv setup", src);
	waitKey(0);
	return 0;
}
