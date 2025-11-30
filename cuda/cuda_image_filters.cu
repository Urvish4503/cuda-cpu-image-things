#include <cstdint>
#include <cuda_runtime.h>

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

void launchGrayScaleCUDAProcess(uchar3 *input, uint8_t *output, int width, int height)
{
    dim3 block(32, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    colorToGrayScaleKernel<<<grid, block>>>(input, output, width, height);
    cudaDeviceSynchronize();
}