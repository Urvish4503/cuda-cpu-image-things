#include "image_filters.h"

#include <cstdint>
#include <opencv2/opencv.hpp>

void colorToGrayScaleCPU(const cv::Mat &input, cv::Mat &output)
{
    int width  = input.cols;
    int height = input.rows;

    output.create(height, width, CV_8UC1);

    for (int y = 0; y < height; y++) {
        const cv::Vec3b *in_row  = input.ptr<cv::Vec3b>(y);
        uint8_t         *out_row = output.ptr<uint8_t>(y);

        for (int x = 0; x < width; x++) {
            cv::Vec3b pixel = in_row[x];

            // pixel = {B, G, R}
            float gray_value = 0.21f * pixel[2]  // R
                             + 0.72f * pixel[1]  // G
                             + 0.07f * pixel[0]; // B

            out_row[x] = static_cast<uint8_t>(gray_value);
        }
    }
}
