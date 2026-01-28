#include "img-utils.cuh"
#include <cuda_runtime.h>

__global__ void apply_mask_kernel(const PPMPixel* base_img_d,
                                  PPMPixel* shaded_img_d,
                                  const float* mask_d,
                                  int w, int h,
                                  int color_depth)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= w || row >= h) return;

    int idx = row * w + col;

    float alpha = mask_d ? mask_d[idx] : 1.0f;

    PPMPixel base_px = base_img_d[idx];
    PPMPixel shaded_px = shaded_img_d[idx];

    shaded_img_d[idx] = apply_opacity_mask(base_px, shaded_px, alpha, color_depth);
}

inline int int_div_rnd_up(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void apply_mask_GPU(const PPMPixel* base_img_d,
                    PPMPixel* shaded_img_d,
                    const float* mask_d,
                    int w, int h,
                    int color_depth)
{
    dim3 threads(20, 20);
    dim3 blocks(int_div_rnd_up(w, threads.x), int_div_rnd_up(h, threads.y));
    printf("Test\n");

    apply_mask_kernel<<<blocks, threads>>>(base_img_d, shaded_img_d, mask_d, w, h, color_depth);
}