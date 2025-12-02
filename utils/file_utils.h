#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


struct ImageJob
{
    std::string path;
    cv::Mat     img;
};

enum class Backend { CPU, GPU };
enum class ProcessType { Blur, Scale };

std::vector<ImageJob> loadImagesFromDirectory(const std::string &dir_path);
std::string           getOutputFilename(const std::string &input_path, ProcessType type, Backend backend);