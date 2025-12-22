#pragma once
#include <vector>
#include <cstdio>
#include <cstdint>
#include <algorithm>

// PPM File format documentation at: https://netpbm.sourceforge.net/doc/ppm.html

// Note: Possible optimization is to restrict r, g and b types to be 1 byte
// If color_depth is <256, this would cut image size in half inside memory
typedef struct {
  uint16_t r;
  uint16_t g;
  uint16_t b;  
}PPMPixel;

class PPMImage {
public:
  int width;
  int height;
  /* Number of distict values for each color 0-65536 */
  uint16_t color_depth;
  std::vector<PPMPixel> pixels;
  PPMImage();
  static PPMImage from_file(std::FILE* fp);
  void to_file(std::FILE* fp);
};
