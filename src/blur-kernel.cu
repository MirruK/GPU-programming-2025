#include "img-utils.cuh"
#define FILTER_SIZE 3
#define FILTER_RADIUS 1

__constant__ float blur_filter[FILTER_SIZE][FILTER_SIZE];

__global__ void blur_kernel(PPMPixel* img_d, PPMPixel* img_out_d, int w, int h, int color_depth) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (col < 0 || col >= w || row < 0 || row >= h) return; 
  PPMPixel px;
  for(int i = -FILTER_RADIUS; i < FILTER_RADIUS; i++){
    for(int j = -FILTER_RADIUS; j < FILTER_RADIUS; j++){
      if ((col + j > 0) && (col + j < w) && (row + i > 0) && (row + i < h)){
	auto px_filtered = px_scale(img_d[(row+i)*w + (col + j)], blur_filter[i][j], color_depth);
	  px = px_add(px_filtered, px, color_depth);
      }
    }
  }
  img_out_d[row*w + col] = px;
}

// Round a / b to nearest higher integer value
// Courtesy of: https://github.com/NVIDIA/cuda-samples
inline int int_div_rnd_up(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }


void blur_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h, int color_depth) {
    dim3 threads(16,12);
    dim3 blocks(int_div_rnd_up(w, threads.x), int_div_rnd_up(h, threads.y));

    blur_kernel<<<blocks, threads>>>(src_img_d, dst_img_d, w, h, color_depth);
}
