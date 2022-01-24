#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void pretty_print(cv::Mat& img, int max_h = 6, int max_w = 6) {
    double t0 = (double)cv::getTickCount();
    int height = img.rows;
    int width = img.cols;
    printf("height:%d\t", height);
    printf("width:%d\n", width);
    for (int y = 0; y < height; y++) {
        const uchar* ptr = img.ptr(y);
        for (int x = 0; x < width; x++) {
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
	cv::Mat src = cv::imread("../../assets/dog.jpg");
	if (src.empty())
	{
		printf("could not load image…\n");
		return -1;
	}
    pretty_print(src);
	cv::namedWindow("test opencv setup");
	cv::imshow("test opencv setup", src);
    cv::waitKey(0);
	return 0;
}
