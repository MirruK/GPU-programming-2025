#include "img-utils.cuh"

__constant__ float blur_filter[FILTER_SIZE][FILTER_SIZE];

cudaError_t set_blur_filter(const float h_filter[FILTER_SIZE][FILTER_SIZE]) {
    return cudaMemcpyToSymbol(blur_filter, h_filter, FILTER_SIZE*FILTER_SIZE*sizeof(float));
}

__global__ void blur_kernel(PPMPixel* img_d, PPMPixel* img_out_d, int w, int h, int color_depth) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (col >= w || row >= h) return; 
  float3 acc = {0.0f,0.0f,0.0f};
  PPMPixel px = {0,0,0};
  for(int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++){
    for(int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++){
      int x = col + j;
      int y = row + i;
      if ((x >= 0) && (x < w) && (y >= 0) && (y < h)){
        auto curr = img_d[y*w + x];
        acc.x += (float)curr.r * blur_filter[i+FILTER_RADIUS][j+FILTER_RADIUS];
        acc.y += (float)curr.g * blur_filter[i+FILTER_RADIUS][j+FILTER_RADIUS];
        acc.z += (float)curr.b * blur_filter[i+FILTER_RADIUS][j+FILTER_RADIUS];
        // Gives different result because every add and multiplication is quantized, check it out
        // auto scaled_px = px_scale(curr, blur_filter[i+FILTER_RADIUS][j+FILTER_RADIUS], color_depth);
        // px = px_add(px, scaled_px, color_depth);
      }
    }
  }
  px.r = min(color_depth, (int)(acc.x + 0.5f)); 
  px.g = min(color_depth, (int)(acc.y + 0.5f)); 
  px.b = min(color_depth, (int)(acc.z + 0.5f)); 
  img_out_d[row*w + col] = px;
}

// Round a / b to nearest higher integer value
// Courtesy of: https://github.com/NVIDIA/cuda-samples
inline int int_div_rnd_up(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }


void blur_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h, int color_depth) {
    dim3 threads(20,20);
    dim3 blocks(int_div_rnd_up(w, threads.x), int_div_rnd_up(h, threads.y));

    blur_kernel<<<blocks, threads>>>(src_img_d, dst_img_d, w, h, color_depth);
}
