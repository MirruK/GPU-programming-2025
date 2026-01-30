#include "img-utils.cuh"

__global__ void inversion_kernel(PPMPixel* img_d, PPMPixel* img_out_d, int w, int h, int color_depth) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (col >= w || row >= h) return;

  PPMPixel px = img_d[row * w + col];
  
  // Invert colors by subtracting from max color depth
  px.r = color_depth - px.r;
  px.g = color_depth - px.g;
  px.b = color_depth - px.b;
  
  img_out_d[row * w + col] = px;
}

// Round a / b to nearest higher integer value
// Courtesy of: https://github.com/NVIDIA/cuda-samples
inline int int_div_rnd_up(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void invert_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h, int color_depth) {
    dim3 threads(16, 16);
    dim3 blocks(int_div_rnd_up(w, threads.x), int_div_rnd_up(h, threads.y));

    inversion_kernel<<<blocks, threads>>>(src_img_d, dst_img_d, w, h, color_depth);
}
