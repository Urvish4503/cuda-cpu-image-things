#include <cstdint>
#include <cuda_runtime.h>

#include "cuda_image_filters.cuh"

__global__ void colorToGrayScaleKernel(uchar3 *input, uint8_t *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int    idx       = y * width + x;
    uchar3 pixel     = input[idx];
    float  grayValue = 0.21f * pixel.z + 0.72f * pixel.y + 0.07f * pixel.x;
    output[idx]      = static_cast<uint8_t>(grayValue);
}

__global__ void blurKernal(uchar3 *input, uchar3 *output, int width, int height, const int BLUR_SIZE)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height)
        return;

    int r = 0, g = 0, b = 0;
    int pixels = 0;

    // Getting avrerage of the surrounding BLUR_SIZE X BLUR_SIZE
    for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; blur_row++) {
        for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; blur_col++) {
            int curr_row = row + blur_row;
            int curr_col = col + blur_col;

            if (curr_col >= 0 && curr_col < width && curr_row >= 0 && curr_row < height) {
                uchar3 pixel = input[curr_row * width + curr_col];

                r += pixel.z;
                g += pixel.y;
                b += pixel.x;

                pixels++; // Keep track of number of pixels
            }
        }
    }
    // Write our new pixel value out
    output[row * width + col] = make_uchar3(b / pixels, g / pixels, r / pixels);
}

void launchGrayScaleCUDAProcess(uchar3 *input, uint8_t *output, int width, int height)
{
    dim3 block(32, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    colorToGrayScaleKernel<<<grid, block>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
void launchBlurCUDAProcess(uchar3 *input, uchar3 *output, int width, int height)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    blurKernal<<<grid, block>>>(input, output, width, height, 3);
    cudaDeviceSynchronize();
}