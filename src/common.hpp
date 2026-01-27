#pragma once
#include <string>

enum class ShaderType{
  BLUR_BOX,
  BLUR_GAUSSIAN,
  BLUR_MOTION,
  BLUR_SOBEL,
  DITHER,
  GRAYSCALE,
  INVERSION,
  MIRROR
};

// Kernel runner (defined in setup.cu)
void run_kernel(std::string outfile, ShaderType shader_type);
