#include "img.hpp"
#include <cstdio>

void run_blur_kernel_test();

int main () {
 //  FILE* f = stdin;
 //  auto img = PPMImage::from_file(f);
 //  std::printf("Width: %d\nHeight: %d\nColor Depth: %d\n;;;;;;;;\n", img.width, img.height, img.color_depth);
 // // for(auto px : img.pixels){
 // //   std::printf("px: {%d,%d,%d} ", px.r, px.g, px.b);
 // // }
 //  FILE* out = fopen("output.ppm", "wb");
 //  img.to_file(out);
 //  fclose(out);
 //  return 0;
  run_blur_kernel_test();
}
