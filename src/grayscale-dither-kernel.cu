#include <cmath>
#include <cuda_runtime.h>
#include "img-utils.cuh"

#define BAYER_N 8 // set to 4, 8, or 16

/*
The size of the Bayer threshold matrix affects the output image in a subtle way.
The larger it is, the better it smooths out edges and the closer it gets
to the intended luminance values. However, smaller matrices will sometimes retain
more detail. An 8 x 8 matrix seems to be a good middle ground between retaining
detail and smoothing out edges. The matrix sizes we can use are limited to
powers of two because we use a bitwise hack to calculate modulo efficiently in
the bayer_threshold() function.
*/

#if BAYER_N == 4 

__constant__ unsigned char bayer[BAYER_N][BAYER_N] = {
    {  0,  8,  2, 10 },
    { 12,  4, 14,  6 },
    {  3, 11,  1,  9 },
    { 15,  7, 13,  5 }
};

#elif BAYER_N == 8

__constant__ unsigned char bayer[BAYER_N][BAYER_N] = {
    {  0, 48, 12, 60,  3, 51, 15, 63 },
    { 32, 16, 44, 28, 35, 19, 47, 31 },
    {  8, 56,  4, 52, 11, 59,  7, 55 },
    { 40, 24, 36, 20, 43, 27, 39, 23 },
    {  2, 50, 14, 62,  1, 49, 13, 61 },
    { 34, 18, 46, 30, 33, 17, 45, 29 },
    { 10, 58,  6, 54,  9, 57,  5, 53 },
    { 42, 26, 38, 22, 41, 25, 37, 21 }
};

#elif BAYER_N == 16

__constant__ unsigned char bayer[BAYER_N][BAYER_N] = {
    {  0,128, 32,160,  8,136, 40,168,  2,130, 34,162, 10,138, 42,170 },
    {192, 64,224, 96,200, 72,232,104,194, 66,226, 98,202, 74,234,106 },
    { 48,176, 16,144, 56,184, 24,152, 50,178, 18,146, 58,186, 26,154 },
    {240,112,208, 80,248,120,216, 88,242,114,210, 82,250,122,218, 90 },

    { 12,140, 44,172,  4,132, 36,164, 14,142, 46,174,  6,134, 38,166 },
    {204, 76,236,108,196, 68,228,100,206, 78,238,110,198, 70,230,102 },
    { 60,188, 28,156, 52,180, 20,148, 62,190, 30,158, 54,182, 22,150 },
    {252,124,220, 92,244,116,212, 84,254,126,222, 94,246,118,214, 86 },

    {  3,131, 35,163, 11,139, 43,171,  1,129, 33,161,  9,137, 41,169 },
    {195, 67,227, 99,203, 75,235,107,193, 65,225, 97,201, 73,233,105 },
    { 51,179, 19,147, 59,187, 27,155, 49,177, 17,145, 57,185, 25,153 },
    {243,115,211, 83,251,123,219, 91,241,113,209, 81,249,121,217, 89 },

    { 15,143, 47,175,  7,135, 39,167, 13,141, 45,173,  5,133, 37,165 },
    {207, 79,239,111,199, 71,231,103,205, 77,237,109,197, 69,229,101 },
    { 63,191, 31,159, 55,183, 23,151, 61,189, 29,157, 53,181, 21,149 },
    {255,127,223, 95,247,119,215, 87,253,125,221, 93,245,117,213, 85 }
};

#else
#error "Unsupported BAYER_N. Use 4, 8, or 16."
#endif


inline __device__ float bayer_threshold(int x, int y) {
    // Bitwise AND is used as a hack for fast modulo: x & (N-1) == x % N
    // (Works only when N is a power of 2)
    return ((float)bayer[y & (BAYER_N - 1)][x & (BAYER_N - 1)] + 0.5f) / (BAYER_N * BAYER_N);
}

__global__ void grayscale_dither_kernel(const PPMPixel* img_d, PPMPixel* img_out_d, int w, int h, int color_depth, int levels) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= w || row >= h) return;

    PPMPixel px = img_d[row * w + col];

    // Normalize each channel to the range [0..1]
    float red_norm   = normalize01((float)px.r, color_depth);
    float green_norm = normalize01((float)px.g, color_depth);
    float blue_norm  = normalize01((float)px.b, color_depth);

    // Convert to linear light from sRGB
    // (sRGB values are not linearly spaced, which messes with the quantization)
    float red_norm_linear   = srgb_to_linear(red_norm);
    float green_norm_linear = srgb_to_linear(green_norm);
    float blue_norm_linear  = srgb_to_linear(blue_norm);

    // Luma approximation Rec. 709
    // https://en.wikipedia.org/wiki/Luma_%28video%29
    float gray_norm_linear = 0.2126f * red_norm_linear
                           + 0.7152f * green_norm_linear
                           + 0.0722f * blue_norm_linear;

    gray_norm_linear = clamp01(gray_norm_linear);
    
    float v = gray_norm_linear * (levels - 1);  // v is the linear light scaled to the quantized levels
    int base = (int)v;                          // v is almost certainly between two levels, the base is the lower quantization level
    float frac = v - base;                      // the fraction measures how far along we are to the higher quantization level [0,1]

    // Dithering with Bayer threshold map
    // https://en.wikipedia.org/wiki/Ordered_dithering
    float bt = bayer_threshold(col, row);
    int level = base + (frac > bt ? 1 : 0);
    if (level < 0) level = 0;
    if (level > levels - 1) level = levels - 1;

    // Convert quantized level back to linear light, then to normalized sRGB,
    // then de-normalize it to the output range.
    float q_linear = (float)level / (levels - 1);
    float q = linear_to_srgb(q_linear);
    int g = (int)(q * color_depth + 0.5f);
    if (g < 0) g = 0;
    if (g > color_depth) g = color_depth;

    px.r = (uint16_t)g;
    px.g = (uint16_t)g;
    px.b = (uint16_t)g;
    img_out_d[row * w + col] = px;
}

inline int int_div_rnd_up(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void grayscale_dither_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h, int color_depth, int levels) {
    if (levels < 2) levels = 2;
    dim3 threads(16, 16);
    dim3 blocks(int_div_rnd_up(w, threads.x), int_div_rnd_up(h, threads.y));
    grayscale_dither_kernel<<<blocks, threads>>>(src_img_d, dst_img_d, w, h, color_depth, levels);
}