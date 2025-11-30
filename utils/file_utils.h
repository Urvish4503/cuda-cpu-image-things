#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


struct ImageJob
{
    std::string path;
    cv::Mat     img;
};

std::vector<ImageJob> loadImagesFromDirectory(const std::string &dir_path);
std::string           getOutputFilename(const std::string &input_path, bool is_cup);