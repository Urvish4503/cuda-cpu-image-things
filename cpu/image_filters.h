#pragma once

#include <opencv2/opencv.hpp>


void colorToGrayScaleCPU(const cv::Mat &input, cv::Mat &output);
void blurCUP(const cv::Mat &input, cv::Mat &output, const int BLUR_SIZE);