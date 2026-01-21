#pragma once
#include <string>

enum class BlurType {
  BLUR_BOX,
  BLUR_GAUSSIAN,
  BLUR_MOTION,
  BLUR_SOBEL
};

void run_blur_kernel_test(std::string outfile, BlurType blur_type);
