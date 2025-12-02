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


void blurCUP(const cv::Mat &input, cv::Mat &output, const int BLUR_SIZE)
{
    int width  = input.cols;
    int height = input.rows;

    output.create(height, width, CV_8UC3);


    // Getting avrerage of the surrounding BLUR_SIZE X BLUR_SIZE
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {

            int r = 0, g = 0, b = 0;
            int pixels = 0;

            for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; blur_row++) {
                for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; blur_col++) {

                    int curr_row = row + blur_row;
                    int curr_col = col + blur_col;


                    if (curr_col >= 0 && curr_col < width && curr_row >= 0 && curr_row < height) {
                        const cv::Vec3b &px = input.at<cv::Vec3b>(curr_row, curr_col);

                        b += px[0]; // Blue
                        g += px[1]; // Green
                        r += px[2]; // Red

                        pixels++;
                    }
                }
            }
            output.at<cv::Vec3b>(row, col) = cv::Vec3b(b / pixels, g / pixels, r / pixels);
        }
    }
}