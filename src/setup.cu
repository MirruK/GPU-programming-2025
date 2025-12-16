#include <cstdio>
#include "img.hpp"
#include "img-utils.cuh"

void run_blur_kernel_test(){
  printf("Running kernel test\n");
  FILE* in_fp = stdin;
  // Allocate image on host
  auto img = PPMImage::from_file(in_fp);
  fclose(in_fp);
  int w = img.width;
  int h = img.height;
  int color_depth = img.color_depth;
  PPMPixel* in_pixels_h = img.pixels.data();
  // Allocate output image on host
  auto out_img = PPMImage();
  out_img.width = w;
  out_img.height = h;
  out_img.color_depth = color_depth;
  out_img.pixels = std::vector<PPMPixel>();
  out_img.pixels.reserve(w*h);
  PPMPixel* out_pixels_h = out_img.pixels.data();
  // Allocate device vars
  PPMPixel *px_in_d, *px_out_d;
  cudaMalloc(&px_in_d, w*h*sizeof(PPMPixel));
  cudaMalloc(&px_out_d, w*h*sizeof(PPMPixel));
  // memcopy into device
  cudaMemcpy(&px_in_d, &in_pixels_h, w*h*sizeof(PPMPixel), cudaMemcpyHostToDevice);
  // TODO: RUN KERNEL ON INPUT IMG
  blur_image_GPU(px_in_d, px_out_d, w, h, color_depth);
  cudaDeviceSynchronize();

  printf("Reading back GPU results...\n");
  cudaMemcpy(out_pixels_h, px_out_d, w * h * sizeof(PPMPixel), cudaMemcpyDeviceToHost);
  
  // TODO: WRITE RESULT TO OUT IMG
  FILE* out = fopen("output.ppm", "wb");
  out_img.to_file(out);
  fclose(out);
}
