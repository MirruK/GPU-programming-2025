#include "img.hpp"
#include <cstdio>
#include <string>

void run_convolution_kernel_test(std::string outfile);

int main (int argc, char* argv[]) {
  std::string outfile = "";
  if (argc > 1) {outfile = std::string(argv[1]);}
  run_convolution_kernel_test(outfile);
}
