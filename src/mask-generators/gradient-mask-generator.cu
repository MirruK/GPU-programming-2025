#include <cuda_runtime.h>
#include "mask-utils.cuh"


__global__ void gradient_mask_kernel(float* mask_d, int w, int h, GradientDirection dir)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= w || row >= h) return;

    int idx = row * w + col;

    float t = 0.0f;

    switch (dir) {
        case LEFT_TO_RIGHT:
            t = (w > 1) ? (float)col / (float)(w - 1) : 0.0f;
            break;

        case RIGHT_TO_LEFT:
            t = (w > 1) ? 1.0f - ((float)col / (float)(w - 1)) : 0.0f;
            break;

        case TOP_TO_BOTTOM:
            t = (h > 1) ? (float)row / (float)(h - 1) : 0.0f;
            break;

        case BOTTOM_TO_TOP: // Fall-through: BOTTOM_TO_TOP is also the default case.
        default:
            t = (h > 1) ? 1.0f - ((float)row / (float)(h - 1)) : 0.0f;
            break;
    }

    mask_d[idx] = t;
}

inline int int_div_rnd_up(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void generate_gradient_mask(float* mask_d, int w, int h, GradientDirection dir)
{
    dim3 threads(20, 20);
    dim3 blocks(int_div_rnd_up(w, threads.x), int_div_rnd_up(h, threads.y));
    gradient_mask_kernel<<<blocks, threads>>>(mask_d, w, h, dir);
}