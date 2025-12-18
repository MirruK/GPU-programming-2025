#include <cstdio>
#include "img.hpp"
#include "img-utils.cuh"

void run_blur_kernel_test(){
  printf("Running kernel test\n");
  FILE* in_fp = stdin;
  // Allocate image on host
  // TODO: Take this information from cmdline args (allow either stdin or file(s))
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
  out_img.pixels = std::vector<PPMPixel>();
  out_img.pixels.reserve(w*h);
  PPMPixel* out_pixels_h = out_img.pixels.data();
  // Allocate device vars
  PPMPixel *px_in_d;
  PPMPixel *px_out_d;
  cudaMalloc((void**)&px_in_d, w*h*sizeof(PPMPixel));
  cudaMalloc((void**)&px_out_d, w*h*sizeof(PPMPixel));
  // memcopy into device
  float blur_filter_host[FILTER_SIZE][FILTER_SIZE];
  for(int i = 0; i < FILTER_SIZE; i++) {
    for(int j = 0; j < FILTER_SIZE; j++) {
      blur_filter_host[i][j] = 1.0f/((float)(FILTER_SIZE*FILTER_SIZE));
    }
  }
  printf("Blur filter: w: %d, h: %d\n Contents:\n", FILTER_SIZE, FILTER_SIZE);
  for(int i = 0; i < FILTER_SIZE; i++) {
    for(int j = 0; j < FILTER_SIZE; j++) {
      printf("%f, ", blur_filter_host[i][j]);
    }
    printf("\n");
  }
  cudaError_t cuda_error = set_blur_filter(blur_filter_host);
  if (cuda_error != cudaSuccess) {
      printf("MemcpyToSymbol failed: %s\n", cudaGetErrorString(cuda_error));
  }
  cudaMemcpy(px_in_d, in_pixels_h, w*h*sizeof(PPMPixel), cudaMemcpyHostToDevice);
  blur_image_GPU(px_in_d, px_out_d, w, h, color_depth);
  cuda_error = cudaDeviceSynchronize();
  if(cuda_error != cudaSuccess){
    printf("Error when running kernel: %s\n", cudaGetErrorString(cuda_error));
  }

  printf("Reading back GPU results...\n");
  cudaMemcpy(out_pixels_h, px_out_d, w * h * sizeof(PPMPixel), cudaMemcpyDeviceToHost);
  FILE* out = fopen("output.ppm", "wb");
  out_img.to_file(out);
  fclose(out);
}
