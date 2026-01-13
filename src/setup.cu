#include <cstdio>
#include <string>
#include "img.hpp"
#include "img-utils.cuh"

void init_blur_filter(float filter[FILTER_SIZE][FILTER_SIZE]) {
  for(int i = 0; i < FILTER_SIZE; i++) {
    for(int j = 0; j < FILTER_SIZE; j++) {
      filter[i][j] = 1.0f/((float)(FILTER_SIZE*FILTER_SIZE));
    }
  }
}

void init_sharpen_filter(float filter[3][3]) {
  float temp[3][3] = {{0,-1,0},{-1,5,-1}, {0,-1,0}};
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      filter[i][j] = temp[i][j];
    }
  }
}

void run_blur_kernel_test(std::string outfile, BlurType blur_type){
  printf("Running kernel test\n");
  FILE* in_fp = stdin;
  // Allocate image on host
  auto img = PPMImage::from_file(in_fp);
  fclose(in_fp);
  int w = img.width;
  int h = img.height;
  int color_depth = img.color_depth;
  PPMPixel* in_pixels_h = img.pixels.data();
  printf("Width: %d, Height: %d\nColor depth: %d\n", w, h, color_depth);
  // Allocate output image on host
  PPMImage out_img = PPMImage();
  out_img.width = w;
  out_img.height = h;
  out_img.color_depth = color_depth;
  out_img.pixels = std::vector<PPMPixel>(w*n);
  PPMPixel* out_pixels_h = out_img.pixels.data();
  // Allocate device vars
  PPMPixel *px_in_d;
  PPMPixel *px_out_d;
  cudaMalloc((void**)&px_in_d, w*h*sizeof(PPMPixel));
  cudaMalloc((void**)&px_out_d, w*h*sizeof(PPMPixel));
  // memcopy into device
  float filter_host[FILTER_SIZE][FILTER_SIZE];
  // Select the blur filter type and check for errors
  select_blur_filter(blur_type);
  // Copy input image to device
  cudaMemcpy(px_in_d, in_pixels_h, w * h * sizeof(PPMPixel), cudaMemcpyHostToDevice);

  // Run kernel
  blur_image_GPU(px_in_d, px_out_d, w, h, color_depth);
  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    printf("Error when running kernel: %s\n", cudaGetErrorString(cuda_error));
  }

  printf("Reading back GPU results...\n");
  cudaMemcpy(out_pixels_h, px_out_d, w * h * sizeof(PPMPixel), cudaMemcpyDeviceToHost);
  FILE* out;
  if (outfile.length() > 0) {
    out = fopen(outfile.c_str(), "wb");
  } else {
    out = fopen("output.ppm", "wb");
  }
  out_img.to_file(out);
  fclose(out);

  // free device memory
  cudaFree(px_in_d);
  cudaFree(px_out_d);
}

