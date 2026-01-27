#include <cstdio>
#include <string>
#include "common.hpp"
#include "img.hpp"
#include "img-utils.cuh"

void run_dither(std::string outfile){}

void run_grayscale(std::string outfile){
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
  out_img.pixels = std::vector<PPMPixel>(w*h);
  PPMPixel* out_pixels_h = out_img.pixels.data();
  // Allocate device vars
  PPMPixel *px_in_d;
  PPMPixel *px_out_d;
  cudaMalloc((void**)&px_in_d, w*h*sizeof(PPMPixel));
  cudaMalloc((void**)&px_out_d, w*h*sizeof(PPMPixel));
  // Select the blur filter type and check for errors
  select_blur_filter(blur_type);
  // Copy input image to device
  cudaMemcpy(px_in_d, in_pixels_h, w * h * sizeof(PPMPixel), cudaMemcpyHostToDevice);

  // Run kernel
  grayscale_image_GPU(px_in_d, px_out_d, w, h, color_depth);
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

