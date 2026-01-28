#include <cuda_runtime.h>
#include "mask-utils.cuh"

__global__ void radial_mask_kernel(float* mask_d, int w, int h)
{
    // TODO
}

void generate_gradient_mask(float* mask_d, int w, int h, GradientDirection dir)
{
    
}