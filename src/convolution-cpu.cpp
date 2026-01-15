#include "img-utils.cuh"
#include <algorithm>

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
