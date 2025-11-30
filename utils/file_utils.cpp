#include "file_utils.h"

#include <filesystem>
#include <iostream>


std::vector<ImageJob> loadImagesFromDirectory(const std::string &dir_path)
{
    std::vector<ImageJob> images;

    for (const auto &entry : std::filesystem::directory_iterator(dir_path)) {
        if (!entry.is_regular_file())
            continue;

        std::string path = entry.path().string();

        // Check extension
        std::string ext = entry.path().extension().string();
        for (auto &c : ext)
            c = std::tolower(c);

        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp" || ext == ".tiff" || ext == ".webp") {
            cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

            if (img.empty()) {
                std::cerr << "Error loading: " << path << "\n";
                continue;
            }

            images.push_back({path, img});
        }
    }

    return images;
}


std::string getOutputFilename(const std::string &input_path, bool is_cup)
{
    // Get current timestamp
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S") << "_" << std::setfill('0') << std::setw(3)
              << ms.count();

    // Parse path using filesystem
    std::filesystem::path inputFilePath(input_path);
    std::filesystem::path directory = inputFilePath.parent_path();
    if (directory.empty())
        directory = ".";

    std::string base = inputFilePath.stem().string();
    std::string ext  = inputFilePath.extension().string();

    // Create output directory
    std::filesystem::path outputDir = directory / "out";
    std::filesystem::create_directories(outputDir);

    // Construct final output path
    if (is_cup) {
        std::filesystem::path outputPath = outputDir / (base + "_cup_processed_" + timestamp.str() + ext);
        return outputPath.string();
    }
    std::filesystem::path outputPath = outputDir / (base + "_gpu_processed_" + timestamp.str() + ext);
    return outputPath.string();
}