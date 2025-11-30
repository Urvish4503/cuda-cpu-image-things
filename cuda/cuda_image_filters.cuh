#ifndef CUDA_IMAGE_FILTERS_CHU
#define CUDA_IMAGE_FILTERS_CHU

#include <cstdint>
#include <cuda_runtime.h>

void launchGrayScaleCUDAProcess(uchar3 *input, uint8_t *output, int width, int height);

#endif