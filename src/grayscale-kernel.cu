#include "img-utils.cuh"
#include <cuda_runtime.h>

__global__ void grayscale_kernel(const PPMPixel* img_d, PPMPixel* img_out_d, int w, int h, int color_depth) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= w || row >= h) return;

    PPMPixel px = img_d[row * w + col];

    // Luma approximation Rec. 709
    // https://en.wikipedia.org/wiki/Luma_%28video%29
    float gray = 0.2126f * px.r + 0.7152f * px.g + 0.0722f * px.b;

    int g = (int)(gray + 0.5f);
    if (g < 0) g = 0;
    if (g > color_depth) g = color_depth;

    px.r = (uint16_t)g;
    px.g = (uint16_t)g;
    px.b = (uint16_t)g;

    img_out_d[row * w + col] = px;
}

inline int int_div_rnd_up(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void grayscale_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h, int color_depth) {
    dim3 threads(20, 20);
    dim3 blocks(int_div_rnd_up(w, threads.x), int_div_rnd_up(h, threads.y));

    grayscale_kernel<<<blocks, threads>>>(src_img_d, dst_img_d, w, h, color_depth);

}