#include "img-utils.cuh"
#include <cmath>

__constant__ float filter[FILTER_SIZE][FILTER_SIZE];

cudaError_t set_blur_filter(const float h_filter[FILTER_SIZE][FILTER_SIZE]) {
    return cudaMemcpyToSymbol(filter, h_filter, FILTER_SIZE*FILTER_SIZE*sizeof(float));
}

__global__ void convolution_kernel(PPMPixel* img_d, PPMPixel* img_out_d, int w, int h, int color_depth) {
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
        acc.x += (float)curr.r * filter[i+FILTER_RADIUS][j+FILTER_RADIUS];
        acc.y += (float)curr.g * filter[i+FILTER_RADIUS][j+FILTER_RADIUS];
        acc.z += (float)curr.b * filter[i+FILTER_RADIUS][j+FILTER_RADIUS];
        // Gives different result because every add and multiplication is quantized, check it out
        // auto scaled_px = px_scale(curr, filter[i+FILTER_RADIUS][j+FILTER_RADIUS], color_depth);
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

void init_motion_filter_horizontal(){
  float h_filter[FILTER_SIZE][FILTER_SIZE];

  for (int i = 0; i < FILTER_SIZE; ++i) {
    for (int j = 0; j < FILTER_SIZE; ++j) {
      h_filter[i][j] = 0.0f;
    }
  }

  int mid = FILTER_RADIUS;
  float val = 1.0f / FILTER_SIZE;
  for (int j = 0; j < FILTER_SIZE; ++j) {
    h_filter[mid][j] = val;
  }

  cudaError_t err = set_blur_filter(h_filter);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to set motion blur filter: %s\n",
            cudaGetErrorString(err));
  }
}

void init_gaussian_filter(float sigma)
{
  float h_filter[FILTER_SIZE][FILTER_SIZE];

  float sum = 0.0f;
  int r = FILTER_RADIUS;

  for (int i = -r; i <= r; ++i) {
    for (int j = -r; j <= r; ++j) {
      float x = static_cast<float>(j);
      float y = static_cast<float>(i);
      float value = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
      h_filter[i + r][j + r] = value;
      sum += value;
    }
  }

  // normalize
  float inv_sum = 1.0f / sum;
  for (int i = 0; i < FILTER_SIZE; ++i) {
    for (int j = 0; j < FILTER_SIZE; ++j) {
      h_filter[i][j] *= inv_sum;
    }
  }

  cudaError_t err = set_blur_filter(h_filter);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to set Gaussian blur filter: %s\n",
            cudaGetErrorString(err));
  }
}

void init_box_filter()
{
  float h_filter[FILTER_SIZE][FILTER_SIZE];

  float val = 1.0f / (FILTER_SIZE * FILTER_SIZE);  // all equal
  for (int i = 0; i < FILTER_SIZE; ++i) {
    for (int j = 0; j < FILTER_SIZE; ++j) {
      h_filter[i][j] = val;
    }
  }

  cudaError_t err = set_blur_filter(h_filter);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to set box blur filter: %s\n",
            cudaGetErrorString(err));
  }
}

void select_blur_filter(BlurType type) {
  switch (type) {
    case BLUR_BOX:
      init_box_filter();
      break;
    case BLUR_GAUSSIAN:
      init_gaussian_filter(1.0f);
      break;
    case BLUR_MOTION:
      init_motion_filter_horizontal();
      break;
  }
}

void convolve_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h, int color_depth) {
    dim3 threads(20,20);
    dim3 blocks(int_div_rnd_up(w, threads.x), int_div_rnd_up(h, threads.y));

    convolution_kernel<<<blocks, threads>>>(src_img_d, dst_img_d, w, h, color_depth);
}
