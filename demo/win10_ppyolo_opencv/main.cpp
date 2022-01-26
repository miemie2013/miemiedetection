#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <stdio.h>
#include "mmdet/utils.h"
#include "mmdet/tensor.h"
#include "mmdet/convolution.h"

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
        }
        else if (y == max_h) {
            printf("...\n");
        }
    }
    printf("------------------------\n");
    double time = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    printf("Time for pretty_print: %f\n", time);
}


void pretty_print4D(cv::Mat& img, int* shape, int max_h = 6, int max_w = 6) {
    int N = *shape;
    int C = *(shape + 1);
    int H = *(shape + 2);
    int W = *(shape + 3);
    printf("N: %d\t", N);
    printf("C: %d\t", C);
    printf("H: %d\t", H);
    printf("W: %d\n", W);
    int n, c, h, w, id;
    for (n = 0; n < N; n++) {
        for (c = 0; c < C; c++) {
            for (h = 0; h < H; h++) {
                for (w = 0; w < W; w++) {
                    id = img.step[0] * n + img.step[1] * c + img.step[2] * h + w * img.step[3];
                    //cout << id << endl;
                    float* p = (float*)(img.data + id);
                    printf("%f ", *p);
                }
            }
        }
    }
    printf("\n------------------------\n");
}

void print_shape(cv::Mat& img) {
    int H = img.rows;
    int W = img.cols;
    int C = img.channels();
    printf("H:%d\t", H);
    printf("W:%d\t", W);
    printf("C:%d\n", C);
}

int main(int argc, char** argv)
{
    vector<string> param_names;
    vector<string> param_values;
    mmdet::Utils utils;
    utils.readTxt("conv2d.txt", param_names, param_values);


    // 测试用例1
    int stride = 1;
    int padding = 0;
    int shape_[4] = { 2, 1, 1, 1 };
    cv::Mat imm(4, shape_, CV_32FC1, cv::Scalar(255.0));         // 创建四维Mat对象。32F表示32位浮点数。
    vector<int> shape;
    for (int j = 0; j < 4; j++) {
        shape.push_back(shape_[j]);
    }
    mmdet::Tensor<float>* input_tensor = new mmdet::Tensor<float>(imm, shape);

    int kernel_shape_[4] = { 1, 1, 1, 1 };
    cv::Mat kernel_(4, kernel_shape_, CV_32FC1, cv::Scalar(3.2));         // 卷积核
    vector<int> kernel_shape;
    for (int j = 0; j < 4; j++) {
        kernel_shape.push_back(kernel_shape_[j]);
    }
    mmdet::Tensor<float>* kernel = new mmdet::Tensor<float>(kernel_, kernel_shape);



    vector<string> ss = utils.split(param_values.at(0), ',');
    printf("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr\n");
    printf("%f\n", input_tensor->at(0, 0, 0, 0));
    for (int j = 0; j < ss.size(); j++) {
        // 发现不能正确写入
        float* p = (float*)(input_tensor->mat.data + j);
        float qqq = std::stof(ss.at(j));
        printf("%f\n", qqq);
        //*p = qqq;
        //int n, c, h, w;
        //float value;
        //input_tensor->set(n, c, h, w, value);
    }
    printf("%f\n", input_tensor->at(0, 0, 0, 0));
    printf("%f\n", input_tensor->at(1, 0, 0, 0));






    mmdet::Convolution* conv = new mmdet::Convolution(kernel, stride, padding);
    mmdet::Tensor<float>* dstImage = conv->forward(input_tensor);

    // 结果写进txt
    FILE* fp1;
    errno_t err;
    err = fopen_s(&fp1, "cpp_conv.txt", "w"); //若return 1 , 则将指向这个文件的文件流给fp1

    int aa, bb, cc, dd;
    printf("zzzzzzzzzzzzzzzzzzzzzzzzz\n");
    printf("%d\n", dstImage->shape.at(0));
    printf("%d\n", dstImage->shape.at(1));
    printf("%d\n", dstImage->shape.at(2));
    printf("%d\n", dstImage->shape.at(3));
    for (aa = 0; aa < dstImage->shape[0]; aa++) {
        for (bb = 0; bb < dstImage->shape[1]; bb++) {
            for (cc = 0; cc < dstImage->shape[2]; cc++) {
                for (dd = 0; dd < dstImage->shape[3]; dd++) {
                    fprintf(fp1, "%f,", dstImage->at(aa, bb, cc, dd));
                    printf("%f\n", dstImage->at(aa, bb, cc, dd));
                }
            }
        }
    }
    //关闭文件
    fclose(fp1);


	return 0;
}
