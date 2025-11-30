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


void processBatchOnGPU(const std::vector<ImageJob> &batch)
{

    std::cout << "\n--- Processing Batch of " << batch.size() << " images on GPU ---\n";

    Timer batch_timer;
    Timer alloc_timer;
    Timer free_timer;

    // Find the maximum buffer size needed for this batch.
    // allocating maximum space reqired and using only that for current batch
    auto max_job_size = std::max_element(
        batch.begin(), batch.end(), [](const ImageJob &a, const ImageJob &b) { return a.img.total() < b.img.total(); });

    size_t max_image_bytes = max_job_size->img.total() * max_job_size->img.elemSize();


    // Allocating GPU memory
    // uchar3 is the CUDA cv::Vec3b
    uchar3     *d_input;
    uint8_t    *d_output;
    cudaError_t err;

    alloc_timer.reset();
    err = cudaMalloc(&d_input, max_image_bytes);
    if (err != cudaSuccess) {
        std::cerr << "Malloc Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    double alloc_input_ms = alloc_timer.ms();

    alloc_timer.reset();
    err = cudaMalloc(&d_output, max_image_bytes);
    if (err != cudaSuccess) {
        std::cerr << "Malloc Error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return;
    }
    double alloc_output_ms = alloc_timer.ms();

    double total_h2d_ms    = 0;
    double total_kernel_ms = 0;
    double total_d2h_ms    = 0;
    double total_save_ms   = 0;

    // Processing images
    for (const auto &job : batch) {
        int    width      = job.img.cols;
        int    height     = job.img.rows;
        size_t image_size = width * height * sizeof(uchar3);

        Timer h2d_timer;
        // Data copy from host(System) to device(GPU)
        cudaMemcpy(d_input, job.img.ptr<uchar3>(), image_size, cudaMemcpyHostToDevice);
        total_h2d_ms += h2d_timer.ms();

        Timer kernel_timer;
        // Process Image
        launchGrayScaleCUDAProcess(d_input, d_output, width, height);
        cudaDeviceSynchronize();
        total_kernel_ms += kernel_timer.ms();

        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "Kernel Error on " << job.path << ": " << cudaGetErrorString(err) << std::endl;

        Timer   d2h_timer;
        size_t  gray_size = width * height * sizeof(uint8_t);
        cv::Mat output_img(height, width, CV_8UC1);
        // Copy only the grayscale size
        cudaMemcpy(output_img.ptr<uint8_t>(), d_output, gray_size, cudaMemcpyDeviceToHost);
        total_d2h_ms += d2h_timer.ms();


        // Save Result
        Timer       save_timer;
        std::string output_path = getOutputFilename(job.path, false);
        cv::imwrite(output_path, output_img);
        total_save_ms += save_timer.ms();
    }

    free_timer.reset();
    cudaFree(d_input);
    cudaFree(d_output);
    double total_free_ms = free_timer.ms();

    double total_batch_ms = batch_timer.ms();

    // Final summary
    std::cout << "\n--- GPU Batch Summary ---\n";
    std::cout << "Total GPU Batch Time     : " << total_batch_ms << " ms\n";
    std::cout << "CUDA Malloc Input Time   : " << alloc_input_ms << " ms\n";
    std::cout << "CUDA Malloc Output Time  : " << alloc_output_ms << " ms\n";
    std::cout << "CUDA Free Time           : " << total_free_ms << " ms\n";
    std::cout << "Total Host→Device Time   : " << total_h2d_ms << " ms\n";
    std::cout << "Total Kernel Time        : " << total_kernel_ms << " ms\n";
    std::cout << "Total Device→Host Time   : " << total_d2h_ms << " ms\n";
    std::cout << "Total Image Save Time    : " << total_save_ms << " ms\n";
}


void processBatchOnCPU(const std::vector<ImageJob> &batch)
{
    std::cout << "\n--- Processing Batch of " << batch.size() << " images on CPU ---\n";

    Timer batch_timer;

    double total_cpu_ms  = 0;
    double total_save_ms = 0;

    for (const auto &job : batch) {
        Timer cpu_timer;

        cv::Mat output_img;
        colorToGrayScaleCPU(job.img, output_img);
        total_cpu_ms += cpu_timer.ms();

        Timer       save_timer;
        std::string output_path = getOutputFilename(job.path, true);
        cv::imwrite(output_path, output_img);
        total_save_ms += save_timer.ms();
    }

    double total_batch_ms = batch_timer.ms();

    std::cout << "\n--- CPU Batch Summary ---\n";
    std::cout << "Total CPU Batch Time   : " << total_batch_ms << " ms\n";
    std::cout << "Total CPU Work Time    : " << total_cpu_ms << " ms\n";
    std::cout << "Total Save Time        : " << total_save_ms << " ms\n";
}


int main(int argc, char *argv[])
{
    std::string input_dir = "../img";

    // Load all images from img/
    std::vector<ImageJob> all_images = loadImagesFromDirectory(input_dir);

    if (all_images.empty()) {
        std::cerr << "No images found in folder: " << input_dir << "\n";
        return 1;
    }

    std::vector<ImageJob> current_batch;
    size_t                current_batch_bytes = 0;


    // Set a threshold of 512MB to process in chunks to avoid running out of RAM
    const size_t MAX_BATCH_BYTES = 512 * 1024 * 1024;


    for (auto &job : all_images) {
        size_t img_size = job.img.cols * job.img.rows * sizeof(uchar3);


        // If adding this image exceeds our batch limit, process the current batch first
        if (current_batch_bytes + img_size > MAX_BATCH_BYTES && !current_batch.empty()) {
            // Run GPU batch
            processBatchOnGPU(current_batch);

            // Run CPU batch
            processBatchOnCPU(current_batch);
            current_batch.clear();
            current_batch_bytes = 0;
        }

        // Add to current batch
        current_batch.push_back(job);
        current_batch_bytes += img_size;
    }

    // Process any remaining images in the final batch
    if (!current_batch.empty()) {
        // Run GPU batch
        processBatchOnGPU(current_batch);

        // Run CPU batch
        processBatchOnCPU(current_batch);
    }


    return 0;
}
