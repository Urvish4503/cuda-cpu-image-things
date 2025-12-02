#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include "cpu/image_filters.h"
#include "cuda/cuda_image_filters.cuh"
#include "utils/file_utils.h"
#include "utils/time_utils.h"


void processBatchOnGPU(const std::vector<ImageJob> &batch, ProcessType type)
{
    std::cout << "\n--- Processing Batch of " << batch.size() << " images on GPU ---\n";

    Timer batch_timer;
    Timer alloc_timer;
    Timer free_timer;

    auto max_job_size = std::max_element(
        batch.begin(), batch.end(), [](const ImageJob &a, const ImageJob &b) { return a.img.total() < b.img.total(); });

    size_t max_image_bytes_rgb  = max_job_size->img.total() * sizeof(uchar3);
    size_t max_image_bytes_gray = max_job_size->img.total() * sizeof(uint8_t);

    // GPU buffers
    uchar3     *d_input;
    uchar3     *d_output_rgb;
    uint8_t    *d_output_gray;
    cudaError_t err;

    // Allocate input
    alloc_timer.reset();
    err                   = cudaMalloc(&d_input, max_image_bytes_rgb);
    double alloc_input_ms = alloc_timer.ms();
    if (err != cudaSuccess) {
        std::cerr << "Malloc Error (input): " << cudaGetErrorString(err) << "\n";
        return;
    }

    // Allocate output depending on process type
    if (type == ProcessType::Scale) {
        alloc_timer.reset();
        err                    = cudaMalloc(&d_output_gray, max_image_bytes_gray);
        double alloc_output_ms = alloc_timer.ms();
        if (err != cudaSuccess) {
            std::cerr << "Malloc Error (gray output): " << cudaGetErrorString(err) << "\n";
            cudaFree(d_input);
            return;
        }
    }
    else {
        alloc_timer.reset();
        err                    = cudaMalloc(&d_output_rgb, max_image_bytes_rgb);
        double alloc_output_ms = alloc_timer.ms();
        if (err != cudaSuccess) {
            std::cerr << "Malloc Error (rgb output): " << cudaGetErrorString(err) << "\n";
            cudaFree(d_input);
            return;
        }
    }

    double total_h2d_ms    = 0;
    double total_kernel_ms = 0;
    double total_d2h_ms    = 0;
    double total_save_ms   = 0;

    for (const auto &job : batch) {
        int width  = job.img.cols;
        int height = job.img.rows;

        size_t image_bytes_rgb  = width * height * sizeof(uchar3);
        size_t image_bytes_gray = width * height * sizeof(uint8_t);

        // Host → Device
        Timer h2d_timer;
        cudaMemcpy(d_input, job.img.ptr<uchar3>(), image_bytes_rgb, cudaMemcpyHostToDevice);
        total_h2d_ms += h2d_timer.ms();

        // Kernel launch
        Timer kernel_timer;
        if (type == ProcessType::Scale) {
            launchGrayScaleCUDAProcess(d_input, d_output_gray, width, height);
        }
        else {
            launchBlurCUDAProcess(d_input, d_output_rgb, width, height);
        }
        cudaDeviceSynchronize();
        total_kernel_ms += kernel_timer.ms();

        // Error check
        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "Kernel Error on " << job.path << ": " << cudaGetErrorString(err) << "\n";

        // Device → Host
        Timer   d2h_timer;
        cv::Mat output_img;
        if (type == ProcessType::Scale) {
            output_img.create(height, width, CV_8UC1);
            cudaMemcpy(output_img.ptr<uint8_t>(), d_output_gray, image_bytes_gray, cudaMemcpyDeviceToHost);
        }
        else {
            output_img.create(height, width, CV_8UC3);
            cudaMemcpy(output_img.ptr<uchar3>(), d_output_rgb, image_bytes_rgb, cudaMemcpyDeviceToHost);
        }
        total_d2h_ms += d2h_timer.ms();

        // Save result
        Timer       save_timer;
        std::string out_path = getOutputFilename(job.path, type, Backend::GPU);
        cv::imwrite(out_path, output_img);
        total_save_ms += save_timer.ms();
    }

    // Free memory
    free_timer.reset();
    cudaFree(d_input);
    if (type == ProcessType::Scale)
        cudaFree(d_output_gray);
    else
        cudaFree(d_output_rgb);
    double free_ms = free_timer.ms();

    std::cout << "\n--- GPU Batch Summary ---\n";
    std::cout << "Total Batch Time   : " << batch_timer.ms() << " ms\n";
    std::cout << "Host→Device Time   : " << total_h2d_ms << " ms\n";
    std::cout << "Kernel Time        : " << total_kernel_ms << " ms\n";
    std::cout << "Device→Host Time   : " << total_d2h_ms << " ms\n";
    std::cout << "Save Time          : " << total_save_ms << " ms\n";
    std::cout << "Free Time          : " << free_ms << " ms\n";
}


void processBatchOnCPU(const std::vector<ImageJob> &batch, ProcessType type)
{
    std::cout << "\n--- Processing Batch of " << batch.size() << " images on CPU ---\n";

    Timer  batch_timer;
    double total_cpu_ms  = 0;
    double total_save_ms = 0;

    for (const auto &job : batch) {
        Timer cpu_timer;

        cv::Mat output_img;

        if (type == ProcessType::Scale) {
            colorToGrayScaleCPU(job.img, output_img);
        }
        else {
            blurCUP(job.img, output_img, 3);
        }

        total_cpu_ms += cpu_timer.ms();

        Timer       save_timer;
        std::string out_path = getOutputFilename(job.path, type, Backend::CPU);
        cv::imwrite(out_path, output_img);
        total_save_ms += save_timer.ms();
    }

    std::cout << "\n--- CPU Batch Summary ---\n";
    std::cout << "Total Batch Time : " << batch_timer.ms() << " ms\n";
    std::cout << "CPU Work Time    : " << total_cpu_ms << " ms\n";
    std::cout << "Save Time        : " << total_save_ms << " ms\n";
}


// int main(int argc, char *argv[])
// {
//     std::string input_dir = "../img";

//     // Load all images from img/
//     std::vector<ImageJob> all_images = loadImagesFromDirectory(input_dir);

//     if (all_images.empty()) {
//         std::cerr << "No images found in folder: " << input_dir << "\n";
//         return 1;
//     }

//     std::vector<ImageJob> current_batch;
//     size_t                current_batch_bytes = 0;


//     // Set a threshold of 512MB to process in chunks to avoid running out of RAM
//     const size_t MAX_BATCH_BYTES = 512 * 1024 * 1024;


//     for (auto &job : all_images) {
//         size_t img_size = job.img.cols * job.img.rows * sizeof(uchar3);


//         // If adding this image exceeds our batch limit, process the current batch first
//         if (current_batch_bytes + img_size > MAX_BATCH_BYTES && !current_batch.empty()) {
//             // Run GPU batch
//             processBatchOnGPU(current_batch,);

//             // Run CPU batch
//             processBatchOnCPU(current_batch);
//             current_batch.clear();
//             current_batch_bytes = 0;
//         }

//         // Add to current batch
//         current_batch.push_back(job);
//         current_batch_bytes += img_size;
//     }

//     // Process any remaining images in the final batch
//     if (!current_batch.empty()) {
//         // Run GPU batch
//         processBatchOnGPU(current_batch);

//         // Run CPU batch
//         processBatchOnCPU(current_batch);
//     }


//     return 0;
// }

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --blur | --grayscale\n";
        return 1;
    }

    ProcessType type;

    std::string flag = argv[1];

    if (flag == "--blur" || flag == "-b") {
        type = ProcessType::Blur;
    }
    else if (flag == "--grayscale" || flag == "-g") {
        type = ProcessType::Scale;
    }
    else {
        std::cerr << "Unknown flag: " << flag << "\n";
        std::cerr << "Use: --blur | -b | --grayscale | -g\n";
        return 1;
    }

    std::string input_dir = "../img";

    std::vector<ImageJob> all_images = loadImagesFromDirectory(input_dir);

    if (all_images.empty()) {
        std::cerr << "No images found in folder: " << input_dir << "\n";
        return 1;
    }

    std::vector<ImageJob> current_batch;
    size_t                current_batch_bytes = 0;
    const size_t          MAX_BATCH_BYTES     = 512 * 1024 * 1024; // 512MB


    for (auto &job : all_images) {
        size_t img_size = job.img.cols * job.img.rows * sizeof(uchar3);

        // If adding this image exceeds batch limit → process the current one
        if (current_batch_bytes + img_size > MAX_BATCH_BYTES && !current_batch.empty()) {

            processBatchOnGPU(current_batch, type);
            processBatchOnCPU(current_batch, type);

            current_batch.clear();
            current_batch_bytes = 0;
        }

        current_batch.push_back(job);
        current_batch_bytes += img_size;
    }

    // Remaining
    if (!current_batch.empty()) {
        processBatchOnGPU(current_batch, type);
        processBatchOnCPU(current_batch, type);
    }

    return 0;
}
