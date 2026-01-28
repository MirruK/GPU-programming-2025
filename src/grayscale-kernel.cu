#include "img-utils.cuh"
#include <cuda_runtime.h>

__global__ void grayscale_kernel(const PPMPixel* img_d, PPMPixel* img_out_d, int w, int h, int color_depth) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= w || row >= h) return;

    int idx = row * w + col;
    PPMPixel px = img_d[idx];

    float red_norm = normalize01((float)px.r, color_depth);
    float green_norm = normalize01((float)px.g, color_depth);
    float blue_norm = normalize01((float)px.b, color_depth);

    float red_linear = srgb_to_linear(red_norm);
    float green_linear = srgb_to_linear(green_norm);
    float blue_linear = srgb_to_linear(blue_norm);

    // Luma approximation Rec. 709
    // https://en.wikipedia.org/wiki/Luma_%28video%29
    float gray_linear = 0.2126f * red_linear 
                      + 0.7152f * green_linear
                      + 0.0722f * blue_linear;

    float gray_norm = linear_to_srgb(gray_linear);
    int out = denormalize(gray_norm, color_depth);

    px.r = (uint16_t)out;
    px.g = (uint16_t)out;
    px.b = (uint16_t)out;

    img_out_d[idx] = px;
}

inline int int_div_rnd_up(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void grayscale_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h, int color_depth) {
    dim3 threads(20, 20);
    dim3 blocks(int_div_rnd_up(w, threads.x), int_div_rnd_up(h, threads.y));

    grayscale_kernel<<<blocks, threads>>>(src_img_d, dst_img_d, w, h, color_depth);
}