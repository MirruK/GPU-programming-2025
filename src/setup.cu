#include <cstdio>
#include <string>
#include "common.hpp"
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

void write_outfile(std::string outfile, PPMImage out_img){
  FILE* out;
  if (outfile.length() > 0) {
    out = fopen(outfile.c_str(), "wb");
  } else {
    out = fopen("output.ppm", "wb");
  }
  out_img.to_file(out);
  fclose(out);
}

void dispatch_blur_kernel(ShaderType shader_type, PPMPixel* px_in_d, PPMPixel* px_out_d, int w, int h, int color_depth) {
  // Select the blur filter type and check for errors
  select_blur_filter(shader_type);
  // Run kernel
  blur_image_GPU(px_in_d, px_out_d, w, h, color_depth);
}

// Forward declarations for new kernels implemented in separate .cu files.
void invert_image_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h, int color_depth);
void mirror_image_horizontal_GPU(PPMPixel* src_img_d, PPMPixel* dst_img_d, int w, int h);


void run_kernel(std::string infile, std::string outfile, ShaderType shader_type){
  printf("Running kernel test\n");

  FILE* in_fp = fopen(infile.c_str(), "rb");
  if (!in_fp) {
    printf("Failed to open input file '%s'\n", infile.c_str());
    return;
  }

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
  // Copy input image to device
  cudaMemcpy(px_in_d, in_pixels_h, w * h * sizeof(PPMPixel), cudaMemcpyHostToDevice);

  //Dispatch the correct kernel
  switch (shader_type) {
  case ShaderType::BLUR_BOX:
  case ShaderType::BLUR_GAUSSIAN:
  case ShaderType::BLUR_MOTION: dispatch_blur_kernel(shader_type, px_in_d, px_out_d, w, h, color_depth);
    break;
  case ShaderType::DITHER: {
      // Placeholder levels value; will be made configurable later
      int levels = 4;
      grayscale_dither_image_GPU(px_in_d, px_out_d, w, h, color_depth, levels);
    }
    break;
  case ShaderType::GRAYSCALE: {
      // Optional opacity mask not yet wired; pass nullptr for full effect
      grayscale_image_GPU(px_in_d, px_out_d, w, h, color_depth);
    }
    break;
  case ShaderType::INVERSION: {
      invert_image_GPU(px_in_d, px_out_d, w, h, color_depth);
    }
    break;
  case ShaderType::MIRROR: {
      mirror_image_horizontal_GPU(px_in_d, px_out_d, w, h);
    }
    break;
  case ShaderType::BLUR_SOBEL: {
    PPMPixel* grayscale_d;
    float* gx_d;
    float* gy_d;
    cudaMalloc((void**)&grayscale_d, w*h*sizeof(PPMPixel));
    cudaMalloc((void**)&gx_d, w*h*sizeof(float));
    cudaMalloc((void**)&gy_d, w*h*sizeof(float));

    sobel_image_GPU(px_in_d, px_out_d, grayscale_d, gx_d, gy_d, w, h, color_depth);

    cudaFree(grayscale_d);
    cudaFree(gx_d);
    cudaFree(gy_d);
    }
    break;
  }

  apply_mask_GPU(px_in_d, px_out_d, nullptr, w, h, color_depth);

  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    printf("Error when running kernel: %s\n", cudaGetErrorString(cuda_error));
  }

  printf("Reading back GPU results...\n");
  cudaMemcpy(out_pixels_h, px_out_d, w * h * sizeof(PPMPixel), cudaMemcpyDeviceToHost);
  write_outfile(outfile, out_img);

  // free device memory
  cudaFree(px_in_d);
  cudaFree(px_out_d);
}
