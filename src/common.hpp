#pragma once
#include <string>

enum class BlurType {
  BLUR_BOX,
  BLUR_GAUSSIAN,
  BLUR_MOTION
};

void run_blur_kernel_test(std::string outfile, BlurType blur_type);
