#include "img.hpp"
#include <algorithm>
#include <cstdio>
#include <string>
#define FILTER_SIZE 3
#define FILTER_RADIUS 1

void convolve_image_CPU(const PPMPixel* src, PPMPixel* dst, int w, int h, int color_depth,
                        const float filter[FILTER_SIZE][FILTER_SIZE]) {
  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;
      for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i) {
        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j) {
          int x = col + j;
          int y = row + i;
          if (x >= 0 && x < w && y >= 0 && y < h) {
            const PPMPixel &p = src[y * w + x];
            float f = filter[i + FILTER_RADIUS][j + FILTER_RADIUS];
            acc_r += p.r * f;
            acc_g += p.g * f;
            acc_b += p.b * f;
          }
        }
      }
      PPMPixel out;
      out.r = std::min(color_depth, (int)(acc_r + 0.5f));
      out.g = std::min(color_depth, (int)(acc_g + 0.5f));
      out.b = std::min(color_depth, (int)(acc_b + 0.5f));
      dst[row * w + col] = out;
    }
  }
}

// This duplicate code is here to avoid depending on any cuda files for the cpu only code
static void init_box_filter(float filter[FILTER_SIZE][FILTER_SIZE]) {
  const float val = 1.0f / (float)(FILTER_SIZE * FILTER_SIZE);
  for (int i = 0; i < FILTER_SIZE; ++i) {
    for (int j = 0; j < FILTER_SIZE; ++j) {
      filter[i][j] = val;
    }
  }
}

static void write_outfile(const std::string &outfile, PPMImage &out_img){
  FILE* out;
  if (outfile.length() > 0) {
    out = fopen(outfile.c_str(), "wb");
  } else {
    out = fopen("output.ppm", "wb");
  }
  out_img.to_file(out);
  fclose(out);
}

void run_cpu_convolution(std::string outfile) {
  FILE* in_fp = stdin;
  auto img = PPMImage::from_file(in_fp);
  fclose(in_fp);

  int w = img.width;
  int h = img.height;
  int color_depth = img.color_depth;
  PPMPixel* in_pixels_h = img.pixels.data();

  PPMImage out_img;
  out_img.width = w;
  out_img.height = h;
  out_img.color_depth = color_depth;
  out_img.pixels = std::vector<PPMPixel>(w*h);
  PPMPixel* out_pixels_h = out_img.pixels.data();

  float filter[FILTER_SIZE][FILTER_SIZE];
  init_box_filter(filter);

  convolve_image_CPU(in_pixels_h, out_pixels_h, w, h, color_depth, filter);
  write_outfile(outfile, out_img);
}
