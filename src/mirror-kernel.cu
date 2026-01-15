#include "img-utils.cuh"

__global__ void mirror_horizontal_kernel(PPMPixel* img_d, PPMPixel* img_out_d, int w, int h) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (col >= w || row >= h) return;

  // Mirror horizontally: flip column position
  int mirrored_col = w - 1 - col;
  PPMPixel px = img_d[row * w + mirrored_col];
  
  img_out_d[row * w + col] = px;
}

// Round a / b to nearest higher integer value
// Courtesy of: https://github.com/NVIDIA/cuda-samples
inline int int_div_rnd_up(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void mirror_image_horizontal_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h) {
    dim3 threads(20, 20);
    dim3 blocks(int_div_rnd_up(w, threads.x), int_div_rnd_up(h, threads.y));

    mirror_horizontal_kernel<<<blocks, threads>>>(src_img_d, dst_img_d, w, h);
}
